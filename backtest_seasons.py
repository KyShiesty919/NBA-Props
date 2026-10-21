"""
Walk-forward backtest of the projection model over several seasons.

For every player-game (player actually played 10+ min, so a prop would have existed):
  * project him using ONLY data available before that date, with the production code
    (model.project_player -> minutes, blended rates, variance; model.prob_over -> P(over))
  * opponent / pace adjustments use season-to-date team stats as of that date, shrunk
    toward the prior season exactly as run_daily does
  * compare with a NAIVE projection: season-to-date per-game average (prior-season
    average until he has 5 games)

We do NOT have historical sportsbook lines, so we can't compute real profit. Instead:
  1. accuracy: model MAE vs naive MAE per stat
  2. calibration: P(over) buckets vs actual hit rate, Brier score
  3. SYNTHETIC betting sim: the "line" is the naive projection rounded to a half-number,
     both sides -110. Betting when the model's P(over) is far enough from 50%. This answers
     "does the model know more than a naive bettor?" — NOT "does it beat DraftKings?".
     Real books are sharper than a naive projection, so treat these ROIs as an upper bound.

    python backtest_seasons.py            # all four seasons, ~10 min
    python backtest_seasons.py --quick    # 2025-26 only
"""
import sys
import numpy as np
import pandas as pd

import config
from fetch_stats import player_game_logs, team_game_rows, teams
from model import project_player, prob_over, STATS, OPP_COL

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
ALL = STATS + ["PRA"]
WARMUP_GAMES = 5          # skip a player's first N games of a season (model won't have a rate yet w/o prior)


def team_rel_by_date(season, prior_season):
    """(team_abbr, date) -> dict of REL factors as of BEFORE that date, blended with prior season."""
    cur = team_game_rows(season).copy()
    pri = team_game_rows(prior_season) if prior_season else pd.DataFrame()
    t = teams().set_index("TEAM_ID")["ABBR"]
    cols = {"PACE": "POSS", "OPP_PTS": "OPP_PTS", "OPP_REB": "OPP_REB", "OPP_AST": "OPP_AST", "OPP_FG3M": "OPP_FG3M"}
    cur["GAME_DATE"] = pd.to_datetime(cur["GAME_DATE"])
    cur = cur.sort_values(["TEAM_ID", "GAME_DATE"])
    if not pri.empty:
        pfull = pri.groupby("TEAM_ID").agg({v: "mean" for v in cols.values()})
        league_prior = {k: pri[v].mean() for k, v in cols.items()}
    else:
        pfull, league_prior = None, None
    out = {}
    for tid, g in cur.groupby("TEAM_ID"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            gp = i
            rel = {}
            for k, v in cols.items():
                cur_mean = g.loc[:i - 1, v].mean() if i > 0 else np.nan
                prior_val = pfull.loc[tid, v] if pfull is not None and tid in pfull.index else np.nan
                w = gp / (gp + config.TEAM_SHRINK_K)
                if np.isnan(prior_val):
                    val = cur_mean
                    league = cur[v].mean()
                elif np.isnan(cur_mean):
                    val = prior_val
                    league = league_prior[k]
                else:
                    val = w * cur_mean + (1 - w) * prior_val
                    league = w * cur[v].mean() + (1 - w) * league_prior[k]
                rel[k + "_REL"] = (val / league) if (val == val and league) else 1.0
            out[(t.get(tid), g.loc[i, "GAME_DATE"].normalize())] = rel
    return out


def run_season(season, prior_season, quick=False):
    cur = player_game_logs(season)
    prior = player_game_logs(prior_season) if prior_season else cur.iloc[0:0]
    rel = team_rel_by_date(season, prior_season)
    team_dates = cur.groupby("TEAM_ABBREVIATION")["GAME_DATE"].apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    prior_by_p = {p: d for p, d in prior.groupby("PLAYER_ID")}
    for pid, pl in cur.groupby("PLAYER_ID"):
        pl = pl.sort_values("GAME_DATE").reset_index(drop=True)
        pp = prior_by_p.get(pid, prior.iloc[0:0])
        pp_played = pp[pp["MIN"] > 0]
        for i in range(len(pl)):
            g = pl.loc[i]
            if g["MIN"] < 10:
                continue
            pc = pl.iloc[:i]
            n_cur = int((pc["MIN"] > 0).sum())
            if n_cur < WARMUP_GAMES and len(pp_played) < 10:
                continue
            base = project_player(pc, pp)
            if base is None or base["minutes"] < 15:
                continue
            team, opp, home, date = g["TEAM_ABBREVIATION"], g["OPP"], bool(g["HOME"]), g["GAME_DATE"]
            r = rel.get((opp, date), {})
            # back-to-back
            td = team_dates.get(team, [])
            b2b = any((date - d).days == 1 for d in td[-400:]) if td else False
            minutes = base["minutes"] * (config.B2B_MINUTES_MULT if b2b else 1.0)
            pace_adj = 1 + config.PACE_ADJ_STRENGTH * (r.get("PACE_REL", 1.0) - 1)
            row = {"season": season, "player_id": pid, "date": date, "n_cur": n_cur, "minutes_proj": minutes,
                   "minutes_actual": g["MIN"]}
            mu_sum = 0.0
            pc_played = pc[pc["MIN"] > 0]
            for s in STATS:
                opp_adj = 1 + config.OPP_ADJ_STRENGTH * (r.get(OPP_COL[s] + "_REL", 1.0) - 1)
                mu = minutes * base[f"rate_{s}"] * opp_adj * pace_adj
                if s == "PTS" and home:
                    mu *= 1 + config.HOME_PTS_BUMP
                row[f"mu_{s}"], row[f"var_{s}"], row[f"act_{s}"] = mu, base[f"var_{s}"], g[s]
                # naive: season-to-date per-game mean, prior-season mean until 5 games
                row[f"naive_{s}"] = (pc_played[s].mean() if n_cur >= 5 else
                                     (pp_played[s].mean() if len(pp_played) else pc_played[s].mean()))
                mu_sum += mu if s != "FG3M" else 0
            row["mu_PRA"], row["var_PRA"], row["act_PRA"] = mu_sum, base["var_PRA"], g["PRA"]
            row["naive_PRA"] = (pc_played["PRA"].mean() if n_cur >= 5 else
                                (pp_played["PRA"].mean() if len(pp_played) else pc_played["PRA"].mean()))
            rows.append(row)
    df = pd.DataFrame(rows)
    # synthetic line = naive projection to nearest half; model P(over)
    for s in ALL:
        df[f"line_{s}"] = np.floor(df[f"naive_{s}"]) + 0.5
        df[f"p_over_{s}"] = [prob_over(s, m, v, l)[0] for m, v, l in zip(df[f"mu_{s}"], df[f"var_{s}"], df[f"line_{s}"])]
        df[f"over_{s}"] = (df[f"act_{s}"] > df[f"line_{s}"]).astype(int)
    return df


def report(df: pd.DataFrame):
    pd.set_option("display.width", 220)
    print("\n=== 1. ACCURACY: mean absolute error, model vs naive (lower is better) ===")
    acc = []
    for season, d in list(df.groupby("season")) + [("ALL", df)]:
        row = {"season": season, "n": len(d), "min_mae": (d["minutes_proj"] - d["minutes_actual"]).abs().mean()}
        for s in ALL:
            row[f"{s}_model"] = (d[f"mu_{s}"] - d[f"act_{s}"]).abs().mean()
            row[f"{s}_naive"] = (d[f"naive_{s}"] - d[f"act_{s}"]).abs().mean()
            row[f"{s}_bias"] = (d[f"mu_{s}"] - d[f"act_{s}"]).mean()
        acc.append(row)
    acc = pd.DataFrame(acc).set_index("season").round(2)
    print(acc.to_string())

    print("\n=== 2. CALIBRATION: when the model says P(over)=x, how often does it go over? ===")
    bins = [0, 0.35, 0.42, 0.47, 0.53, 0.58, 0.65, 1.0]
    for s in ALL:
        d = df[[f"p_over_{s}", f"over_{s}"]].rename(columns={f"p_over_{s}": "p", f"over_{s}": "y"})
        d["bucket"] = pd.cut(d["p"], bins)
        c = d.groupby("bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        c["gap"] = c["actual"] - c["predicted"]
        brier = ((d["p"] - d["y"]) ** 2).mean()
        print(f"\n{s}: Brier {brier:.4f} (coin flip = 0.2500; lower is better)")
        print(c.round(3).to_string())

    print("\n=== 3. SYNTHETIC BETTING SIM (line = naive projection, -110 both sides; UPPER BOUND, not real books) ===")
    print("Blended prob = 0.35*0.5 + 0.65*model (same market weight as production at full season).")
    dec = 100 / 110 + 1
    sims = []
    for s in ALL:
        p = 0.35 * 0.5 + 0.65 * df[f"p_over_{s}"]
        y = df[f"over_{s}"]
        for edge in (0.04, 0.06, 0.08):
            bet_over, bet_under = p >= 0.5 + edge, p <= 0.5 - edge
            n = int(bet_over.sum() + bet_under.sum())
            if n == 0:
                continue
            wins = int((bet_over & (y == 1)).sum() + (bet_under & (y == 0)).sum())
            profit = wins * (dec - 1) - (n - wins)
            sims.append({"stat": s, "min_edge": edge, "bets": n, "hit_rate": wins / n, "roi": profit / n,
                         "bets_per_season": n / df["season"].nunique()})
    sims = pd.DataFrame(sims)
    print(sims.round(3).to_string(index=False))
    print("\nBreak-even hit rate at -110 is 52.4%.")
    # by season at 6% edge
    print("\nBy season, all stats pooled, min edge 6%:")
    by = []
    for season, d in df.groupby("season"):
        n = w = 0
        for s in ALL:
            p = 0.35 * 0.5 + 0.65 * d[f"p_over_{s}"]
            y = d[f"over_{s}"]
            bo, bu = p >= 0.56, p <= 0.44
            n += int(bo.sum() + bu.sum())
            w += int((bo & (y == 1)).sum() + (bu & (y == 0)).sum())
        by.append({"season": season, "bets": n, "hit_rate": w / n if n else np.nan,
                   "roi": (w * (dec - 1) - (n - w)) / n if n else np.nan})
    print(pd.DataFrame(by).round(3).to_string(index=False))
    return acc, sims


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    seasons = SEASONS[-1:] if quick else SEASONS
    frames = []
    for i, s in enumerate(SEASONS):
        if s not in seasons:
            continue
        prior = SEASONS[i - 1] if i > 0 else None
        print(f"running {s} (prior {prior}) ...", flush=True)
        frames.append(run_season(s, prior))
        print(f"  {len(frames[-1])} player-games")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(config.OUT_DIR / "backtest_seasons.parquet") if hasattr(df, "to_parquet") else None
    acc, sims = report(df)
    with pd.ExcelWriter(config.OUT_DIR / "backtest_seasons.xlsx", engine="openpyxl") as xw:
        acc.to_excel(xw, sheet_name="Accuracy")
        sims.to_excel(xw, sheet_name="Synthetic sim", index=False)
        df.sample(min(len(df), 50000), random_state=1).to_excel(xw, sheet_name="Sample rows", index=False)
    print(f"\nSaved {config.OUT_DIR / 'backtest_seasons.xlsx'}")
