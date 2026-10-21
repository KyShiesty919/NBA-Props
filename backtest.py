"""
Backtest the usage-redistribution and minutes rules on last season's game logs.

For every team-game in PRIOR_SEASON:
  * trailing stats for each player come ONLY from games before that date (no leakage)
  * "absent rotation players" = guys who played >= 15 min in >= 3 of the team's last 5 games
    but have no line in this one  (i.e. a known DNP at tip-off, the case the usage model handles)
  * teammates who played >= 10 min are projected two ways:
        A) baseline: trailing minutes x trailing per-minute rate
        B) redistributed: usage.redistribute() with a given ALPHA / minutes rule
    and compared with actual PTS / REB / AST.

Outputs bias (mean projected - actual) and MAE for each setting, overall and for the
players the model boosted most (the ones whose props you'd actually bet on).
Also compares minutes rules on ALL games: last-10-played EWMA vs per-team-game mean.

    python backtest.py            # ~2-4 min
"""
import sys
import numpy as np
import pandas as pd

import config
import usage
from fetch_stats import player_game_logs, rosters
from injuries import pos_group
from model import _wmean

STATS = ["PTS", "REB", "AST"]
TRAIL = 15          # games of history per player for rates
MIN_LOOK = 10       # games for minutes EWMA


def team_games(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.groupby(["TEAM_ABBREVIATION", "GAME_ID"]).agg(date=("GAME_DATE", "first")).reset_index()
    return g.sort_values(["TEAM_ABBREVIATION", "date"]).reset_index(drop=True)


def trailing(player_logs: pd.DataFrame, before: pd.Timestamp) -> dict | None:
    h = player_logs[(player_logs["GAME_DATE"] < before) & (player_logs["MIN"] > 0)].tail(TRAIL)
    if len(h) < 5:
        return None
    out = {"minutes": _wmean(h["MIN"].tail(MIN_LOOK).to_numpy(), config.MINUTES_DECAY)}
    for s in STATS:
        out[f"rate_{s}"] = h[s].sum() / h["MIN"].sum()
    return out


def run(logs: pd.DataFrame, pos_map: dict, alpha_scale: float, fill_minutes: bool, empirical: bool,
        max_games: int | None = None, fill_fraction: float = 1.0) -> pd.DataFrame:
    # patch the knobs under test
    usage.ALPHA = {k: v * alpha_scale for k, v in BASE_ALPHA.items()}
    usage.FILL_FRACTION = fill_fraction
    rows = []
    tg = team_games(logs)
    by_team = {t: d for t, d in logs.groupby("TEAM_ABBREVIATION")}
    n_done = 0
    for team, games in tg.groupby("TEAM_ABBREVIATION"):
        tl = by_team[team]
        by_player = {p: d for p, d in tl.groupby("PLAYER_ID")}
        games = games.reset_index(drop=True)
        for i in range(6, len(games)):
            gid, date = games.loc[i, "GAME_ID"], games.loc[i, "date"]
            last5 = set(games.loc[i - 5:i - 1, "GAME_ID"])
            played_today = tl[tl["GAME_ID"] == gid]
            today_ids = set(played_today["PLAYER_ID"])
            recent = tl[tl["GAME_ID"].isin(last5) & (tl["MIN"] >= 15)]
            rot = recent.groupby("PLAYER_ID").size()
            rot = set(rot[rot >= 3].index)
            absent = rot - today_ids
            if not absent:
                continue
            # build team rows from trailing stats
            avail, outs = [], []
            for pid in rot | today_ids:
                tr = trailing(by_player[pid], date)
                if tr is None:
                    continue
                row = {**tr, "player_id": pid, "player": str(pid), "pos_group": pos_map.get(pid, "F"),
                       "var_PTS": 0, "var_REB": 0, "var_AST": 0, "var_PRA": 0, "rate_FG3M": 0}
                (outs if pid in absent else avail).append(row)
            if not avail or not outs:
                continue
            adf, odf = pd.DataFrame(avail), pd.DataFrame(outs)
            hist = logs[logs["GAME_DATE"] < date] if empirical else None
            red = usage.redistribute(adf, odf, hist if empirical else logs.iloc[0:0],
                                     ) if fill_minutes else _rates_only(adf, odf)
            red = red.set_index("player_id")
            base = adf.set_index("player_id")
            for _, a in played_today[played_today["MIN"] >= 10].iterrows():
                pid = a["PLAYER_ID"]
                if pid not in red.index:
                    continue
                r = {"team": team, "game": gid, "date": date, "player_id": pid, "n_out": len(outs),
                     "out_minutes": odf["minutes"].sum(), "min_actual": a["MIN"],
                     "min_base": base.loc[pid, "minutes"], "min_red": red.loc[pid, "minutes"]}
                for s in STATS:
                    r[f"{s}_actual"] = a[s]
                    r[f"{s}_base"] = base.loc[pid, "minutes"] * base.loc[pid, f"rate_{s}"]
                    r[f"{s}_red"] = red.loc[pid, "minutes"] * red.loc[pid, f"rate_{s}"]
                rows.append(r)
            n_done += 1
            if max_games and n_done >= max_games:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _rates_only(adf, odf):
    """Redistribute rates but not minutes (to isolate the two effects)."""
    saved = usage.TEAM_MINUTES
    usage.TEAM_MINUTES = 0        # deficit never positive -> no minute fill
    try:
        return usage.redistribute(adf, odf, adf.iloc[0:0])
    finally:
        usage.TEAM_MINUTES = saved


def summarize(df: pd.DataFrame, label: str) -> dict:
    boosted = df[(df["min_red"] - df["min_base"]) > 2]
    out = {"setting": label, "n_player_games": len(df), "n_boosted": len(boosted)}
    for s in STATS:
        out[f"{s}_bias_base"] = (df[f"{s}_base"] - df[f"{s}_actual"]).mean()
        out[f"{s}_bias_red"] = (df[f"{s}_red"] - df[f"{s}_actual"]).mean()
        out[f"{s}_mae_base"] = (df[f"{s}_base"] - df[f"{s}_actual"]).abs().mean()
        out[f"{s}_mae_red"] = (df[f"{s}_red"] - df[f"{s}_actual"]).abs().mean()
        if len(boosted):
            out[f"{s}_boosted_bias_base"] = (boosted[f"{s}_base"] - boosted[f"{s}_actual"]).mean()
            out[f"{s}_boosted_bias_red"] = (boosted[f"{s}_red"] - boosted[f"{s}_actual"]).mean()
            out[f"{s}_boosted_mae_base"] = (boosted[f"{s}_base"] - boosted[f"{s}_actual"]).abs().mean()
            out[f"{s}_boosted_mae_red"] = (boosted[f"{s}_red"] - boosted[f"{s}_actual"]).abs().mean()
    out["min_bias_base"] = (df["min_base"] - df["min_actual"]).mean()
    out["min_bias_red"] = (df["min_red"] - df["min_actual"]).mean()
    out["min_mae_base"] = (df["min_base"] - df["min_actual"]).abs().mean()
    out["min_mae_red"] = (df["min_red"] - df["min_actual"]).abs().mean()
    return out


def minutes_rules(logs: pd.DataFrame) -> pd.DataFrame:
    """On all games (player played >= 10 min): which trailing minutes rule predicts tonight best?"""
    rows = []
    tg = team_games(logs)
    tg_idx = {(t, g): i for i, (t, g) in enumerate(zip(tg["TEAM_ABBREVIATION"], tg["GAME_ID"]))}
    for pid, pl in logs.groupby("PLAYER_ID"):
        pl = pl.sort_values("GAME_DATE").reset_index(drop=True)
        team = pl["TEAM_ABBREVIATION"].iloc[-1]
        team_sched = tg[tg["TEAM_ABBREVIATION"] == team].reset_index(drop=True)
        played = pl[pl["TEAM_ABBREVIATION"] == team]
        if len(played) < 25:
            continue
        mins_by_game = dict(zip(played["GAME_ID"], played["MIN"]))
        sched_ids = team_sched["GAME_ID"].tolist()
        for i in range(20, len(sched_ids)):
            gid = sched_ids[i]
            actual = mins_by_game.get(gid)
            if actual is None or actual < 10:
                continue
            prev = sched_ids[max(0, i - 15):i]
            per_team_game = np.mean([mins_by_game.get(g, 0.0) for g in prev])
            played_prev = [mins_by_game[g] for g in prev if g in mins_by_game and mins_by_game[g] > 0][-MIN_LOOK:]
            if len(played_prev) < 5:
                continue
            ewma = _wmean(np.array(played_prev), config.MINUTES_DECAY)
            mean_played = float(np.mean(played_prev))
            rows.append({"actual": actual, "ewma10": ewma, "mean_played": mean_played,
                         "per_team_game15": per_team_game, "blend": 0.5 * ewma + 0.5 * per_team_game})
    d = pd.DataFrame(rows)
    res = {}
    for c in ("ewma10", "mean_played", "per_team_game15", "blend"):
        res[c] = {"bias": (d[c] - d["actual"]).mean(), "mae": (d[c] - d["actual"]).abs().mean()}
    return pd.DataFrame(res).T.round(2), len(d)


BASE_ALPHA = dict(usage.ALPHA)

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    prior = player_game_logs(config.PRIOR_SEASON)
    ros = rosters()
    pos_map = ros.set_index("PLAYER_ID")["POS"].map(pos_group).to_dict()
    print(f"{len(prior)} logs, {prior['GAME_ID'].nunique()} games")

    print("\n=== MINUTES RULES (all games, player played 10+ min) ===")
    mr, n = minutes_rules(prior)
    print(mr.to_string(), f"\n(n = {n} player-games)")

    print("\n=== USAGE REDISTRIBUTION (games with a rotation player out) ===")
    settings = [("no redistribution", 0.0, False, False, 1.0),
                ("rates only, alpha x2.0", 2.0, False, False, 1.0),
                ("fill 100% + alpha x2.0 (original, pre-calibration)", 2.0, True, False, 1.0),
                ("fill 40% + alpha x0.5", 0.5, True, False, 0.4),
                ("fill 40% + alpha x1.0 (current)", 1.0, True, False, 0.4),
                ("fill 30% + alpha x1.5", 1.5, True, False, 0.3),
                ("fill 50% + alpha x1.0", 1.0, True, False, 0.5)]
    results = []
    for label, a, fill, emp, ff in settings:
        df = run(prior, pos_map, a, fill, emp, max_games=300 if quick else 900, fill_fraction=ff)
        results.append(summarize(df, label))
        print(f"  done: {label} ({len(df)} player-games)")
    res = pd.DataFrame(results).set_index("setting")
    pd.set_option("display.width", 250)
    cols = ["n_player_games", "n_boosted", "min_bias_red", "min_mae_red",
            "PTS_bias_red", "PTS_mae_red", "REB_bias_red", "REB_mae_red", "AST_bias_red", "AST_mae_red"]
    print("\nALL teammates in those games (bias = projected - actual; lower MAE is better):")
    print(res[cols].round(2).to_string())
    bcols = ["n_boosted", "PTS_boosted_bias_base", "PTS_boosted_bias_red", "PTS_boosted_mae_base", "PTS_boosted_mae_red",
             "AST_boosted_bias_red", "AST_boosted_mae_red", "REB_boosted_bias_red", "REB_boosted_mae_red"]
    print("\nPlayers the model BOOSTED by 2+ minutes (the ones you'd bet overs on):")
    print(res[[c for c in bcols if c in res.columns]].round(2).to_string())
    # empirical override check on a subsample (slow)
    print("\n=== EMPIRICAL with/without override (subsample) ===")
    saved = usage.USE_EMPIRICAL
    d0 = run(prior, pos_map, 1.0, True, False, max_games=250, fill_fraction=0.4)
    usage.USE_EMPIRICAL = True
    d1 = run(prior, pos_map, 1.0, True, True, max_games=250, fill_fraction=0.4)
    usage.USE_EMPIRICAL = saved
    for s in STATS:
        print(f"  {s}: MAE generic {(d0[f'{s}_red'] - d0[f'{s}_actual']).abs().mean():.2f}  "
              f"with empirical {(d1[f'{s}_red'] - d1[f'{s}_actual']).abs().mean():.2f}")
    res.round(3).to_csv(config.OUT_DIR / "backtest_usage.csv")
    print(f"\nSaved {config.OUT_DIR / 'backtest_usage.csv'}")
