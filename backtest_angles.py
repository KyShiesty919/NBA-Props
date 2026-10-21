"""
Where could edge actually live? Tests on the cached historical snapshots.

  A. MORNING LINES: run the production model on ~10am lines, grade at morning prices, and
     measure closing-line value (CLV) against the 2h-pre-tip consensus. If model picks
     consistently close better than they opened, morning betting has edge the close-line
     backtest couldn't see.
  B. BLOWOUT UNDERS / REST: do unders do better in games with a big spread? Back-to-backs?
     (uses every graded bet from the main backtest plus the spreads snapshot)
  C. ALT LINES: price every alt-line rung with the model's distribution; does the model find
     mispriced rungs, and would they have paid?
  D. DOUBLE-DOUBLES: model P(DD) via the joint of PTS/REB(/AST) vs the book's price.

    python backtest_angles.py
"""
import gzip
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import config
from backtest_lines import (HIST, PRIOR, SEASON_END, load_day, run_day, team_stats_asof,
                            _with_game_team_ids)
from fetch_lines import flatten, devig, norm_name, american_to_prob
from fetch_stats import player_game_logs, teams, rosters
from injuries import pos_group
from model import prob_over, pick_bets

pd.set_option("display.width", 230)
OUT = config.OUT_DIR


def _days(season, tag):
    suf = ".json.gz" if tag == "close" else f"_{tag}.json.gz"
    return sorted(date.fromisoformat(p.name[:10]) for p in (HIST / season).glob(f"??????????{suf}"))


def _season_ctx(season):
    t = teams()
    abbr_to_id = t.set_index("ABBR")["TEAM_ID"].to_dict()
    cur_all = _with_game_team_ids(player_game_logs(season), abbr_to_id)
    prior_all = _with_game_team_ids(player_game_logs(PRIOR[season]), abbr_to_id)
    t_prior = team_stats_asof(PRIOR[season], pd.Timestamp(SEASON_END[PRIOR[season]]) + pd.Timedelta(days=30))
    ros = rosters()
    pos_map = ros.set_index("PLAYER_ID")["POS"].map(pos_group).to_dict()
    return cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map


def _summ(g):
    g = g[g["result"].isin(["win", "loss"])]
    n = len(g)
    if n == 0:
        return pd.Series({"bets": 0})
    pr = g["profit_flat"]
    return pd.Series({"bets": n, "hit": (g["result"] == "win").mean(), "roi": pr.mean(),
                      "z": pr.mean() / (pr.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                      "avg_clv": g["clv"].mean() if "clv" in g else np.nan,
                      "clv_pos": (g["clv"] > 0).mean() if "clv" in g else np.nan})


# ------------------------------------------------------------------ A. morning lines
def morning_test(season="2025-26"):
    ctx = _season_ctx(season)
    cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map = ctx
    rows_bets, rows_all = [], []
    for d in _days(season, "morning"):
        raw_close = load_day(season, d, "close")
        if not raw_close:
            continue
        close = devig(flatten(raw_close, datetime.now(timezone.utc)))
        close_c = (close.groupby(["event_id", "stat", "player_key", "line"])["mkt_fair_over"].first()
                   .rename("close_fair_over").reset_index())
        # monkeypatch load_day to serve the morning file to run_day
        import backtest_lines as bl
        orig = bl.load_day
        bl.load_day = lambda s, dd, tag="close": orig(s, dd, "morning")
        try:
            evald, bets = run_day(season, d, cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map)
        finally:
            bl.load_day = orig
        if evald is None:
            continue
        for df in (evald, bets):
            m = df.merge(close_c, on=["event_id", "stat", "player_key", "line"], how="left")
            close_side = np.where(m["side"] == "over", m["close_fair_over"], 1 - m["close_fair_over"])
            m["clv"] = close_side - m["mkt_fair_prob"]
            m["moved_to_us"] = m["clv"] > 0
            (rows_bets if df is bets else rows_all).append(m)
    bets = pd.concat(rows_bets, ignore_index=True)
    allp = pd.concat(rows_all, ignore_index=True)
    bets.to_pickle(OUT / "bt_morning_bets.pkl")
    print(f"\n==== A. MORNING LINES ({season}, {bets['date'].nunique()} days) ====")
    print("Production settings (mw .75, edge 3%, 10+ games, no usage-model bets):")
    print(_summ(bets).round(4).to_string())
    print("\nBy side:");  print(bets.groupby("side").apply(_summ).round(4).to_string())
    print("\nBy prop:");  print(bets.groupby("stat").apply(_summ).round(4).to_string())
    # looser settings for comparison: the original mw .35 / edge 4%
    import model as M
    saved = (M.MARKET_WEIGHT, M.MIN_EDGE, M.MIN_N_CUR, M.SKIP_USAGE_MODEL_BETS)
    # recompute blended with mw .35 from stored model/mkt probs
    a = allp[allp["mkt_fair_prob"].notna() & allp["actual"].notna()].copy()
    for mw, me, label in ((0.35, 0.04, "original mw .35 / edge 4%"), (0.75, 0.03, "mw .75 / edge 3%, no n_cur filter"),
                          (0.75, 0.03, "mw .75 / edge 3% / n_cur>=10 / no inj")):
        p = mw * a["mkt_fair_prob"] + (1 - mw) * a["model_prob"]
        dec = np.where(a["price"] > 0, a["price"] / 100 + 1, 100 / -a["price"] + 1)
        d = a.assign(p=p, edge=p - a["mkt_fair_prob"], ev_=p * (dec - 1) - (1 - p))
        d = d[(d["edge"] >= me) & (d["ev_"] > 0) & ~d["suspicious"] & d["book"].isin(config.BETTABLE_BOOKS) & (d["n_books"] >= 2)]
        if "n_cur>=10" in label:
            d = d[(d["n_cur"] >= 10) & (d["inj_note"].fillna("").astype(str).str.len() == 0)]
        d = d.sort_values("ev_", ascending=False).drop_duplicates(["date", "player_key", "stat"])
        print(f"\n{label}:"); print(_summ(d).round(4).to_string())
    print("\nCLV distribution on ALL morning lines the model leaned on (edge>=3%, any book):")
    lean = a[(0.75 * a["mkt_fair_prob"] + 0.25 * a["model_prob"] - a["mkt_fair_prob"]) >= 0.03]
    print(f"  n={len(lean)}  avg CLV {lean['clv'].mean():+.4f}  share moved toward model {lean['moved_to_us'].mean():.3f}")
    base = a.drop_duplicates(["date", "event_id", "stat", "player_key", "line", "side"])
    print(f"  baseline, all lines: avg |move| {base['clv'].abs().mean():.4f}; share that moved at all {(base['clv'] != 0).mean():.3f}")
    return bets


# ------------------------------------------------------------------ B. blowouts / rest
def spreads_table():
    rows = []
    for season in PRIOR:
        for tag in ("extras", "spreads"):
            for d in _days(season, tag):
                raw = load_day(season, d, tag)
                for item in raw or []:
                    ev = item["event"]
                    for bk in item["odds"].get("bookmakers", []):
                        for m in bk.get("markets", []):
                            if m["key"] != "spreads":
                                continue
                            for o in m["outcomes"]:
                                rows.append({"event_id": ev["id"], "book": bk["key"], "team": o["name"], "spread": o["point"]})
    s = pd.DataFrame(rows)
    return s.groupby("event_id")["spread"].apply(lambda x: x.abs().median()).rename("abs_spread").reset_index()


def blowout_rest_test():
    bets = pd.concat([pd.read_pickle(p) for p in OUT.glob("bt_lines_*_bets.pkl")], ignore_index=True)
    sp = spreads_table()
    b = bets.merge(sp, on="event_id", how="left")
    print(f"\n==== B. BLOWOUT UNDERS & REST (all graded bets from the main backtest; {b['abs_spread'].notna().sum()} with a spread) ====")
    b["spread_bucket"] = pd.cut(b["abs_spread"], [0, 4, 8, 12, 30])
    print("\nBy spread bucket and side:")
    print(b.groupby(["spread_bucket", "side"], observed=True).apply(_summ).round(4).to_string())
    print("\nBack-to-back:")
    print(b.groupby(["b2b", "side"]).apply(_summ).round(4).to_string())
    # unders on big-spread games, by prop
    big = b[(b["abs_spread"] >= 10) & (b["side"] == "under")]
    print("\nUnders in 10+ point spread games, by prop:")
    print(big.groupby("stat").apply(_summ).round(4).to_string())
    return b


# ------------------------------------------------------------------ C. alt lines & D. double-doubles
def alt_lines_test(season="2025-26"):
    cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map = _season_ctx(season)
    rows, dd_rows = [], []
    for d in _days(season, "extras"):
        raw = load_day(season, d, "extras")
        if not raw:
            continue
        # projections as of this day: reuse run_day on the close file for the projection table
        evald, _ = run_day(season, d, cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map)
        if evald is None:
            continue
        proj = evald.drop_duplicates("player_key").set_index("player_key")
        act = cur_all[cur_all["GAME_DATE"] == pd.Timestamp(d)].drop_duplicates(["PLAYER_ID", "OPP"]).set_index(["PLAYER_ID", "OPP"])
        for item in raw:
            ev = item["event"]
            for bk in item["odds"].get("bookmakers", []):
                if bk["key"] not in config.BETTABLE_BOOKS:
                    continue
                for m in bk.get("markets", []):
                    if m["key"] == "player_points_alternate":
                        for o in m["outcomes"]:
                            pk = norm_name(o.get("description", ""))
                            if pk not in proj.index or o["name"].lower() != "over":
                                continue
                            r = proj.loc[pk]
                            if r["n_cur"] < 10 or str(r.get("inj_note", "")) not in ("", "nan"):
                                continue
                            # need mu/var for PTS: evald rows don't carry var; recompute from projection + stored var? use mu & disagree
                            mu = r["projection"] if r["stat"] == "PTS" else np.nan
                            rows.append({"date": d, "player": o["description"], "book": bk["key"], "line": o["point"],
                                         "price": o["price"], "player_id": r["player_id"], "opp": r["opp"], "pk": pk})
                    if m["key"] == "player_double_double":
                        for o in m["outcomes"]:
                            pk = norm_name(o.get("description", ""))
                            if pk not in proj.index:
                                continue
                            r = proj.loc[pk]
                            dd_rows.append({"date": d, "player": o["description"], "book": bk["key"], "side": o["name"].lower(),
                                            "price": o["price"], "player_id": r["player_id"], "opp": r["opp"]})
    # We need PTS mu/var per player-day: pull from the evald rows (stat==PTS) which carry projection; var not stored.
    # Rebuild var cheaply: use the player's last-25 game PTS variance before that date.
    alt = pd.DataFrame(rows)
    if alt.empty:
        print("\n==== C. ALT LINES: no data ===="); return
    logs = cur_all.sort_values("GAME_DATE")
    pri = prior_all.sort_values("GAME_DATE")
    def mu_var(pid, d):
        h = pd.concat([pri[pri["PLAYER_ID"] == pid], logs[(logs["PLAYER_ID"] == pid) & (logs["GAME_DATE"] < pd.Timestamp(d))]])
        h = h[h["MIN"] > 0].tail(25)
        return float(np.var(h["PTS"], ddof=1)) if len(h) > 3 else np.nan
    # projection mu for PTS from the close-run evald per day: recompute by rerunning is costly; approximate with evald cache
    ev_all = pd.read_pickle(OUT / f"bt_lines_{season}_all.pkl")
    mu_tab = ev_all[ev_all["stat"] == "PTS"].drop_duplicates(["date", "player_key"]).set_index(["date", "player_key"])["projection"]
    alt["mu"] = [mu_tab.get((d.isoformat(), pk), np.nan) for d, pk in zip(alt["date"], alt["pk"])]
    alt = alt[alt["mu"].notna()].copy()
    alt["var"] = [mu_var(p, d) for p, d in zip(alt["player_id"], alt["date"])]
    alt["p_model"] = [prob_over("PTS", m, v, l)[0] for m, v, l in zip(alt["mu"], alt["var"], alt["line"])]
    alt["p_book"] = alt["price"].apply(american_to_prob)
    alt["dec"] = np.where(alt["price"] > 0, alt["price"] / 100 + 1, 100 / -alt["price"] + 1)
    # alt ladders are one-sided; book's implied includes ~5-8% hold on that side. Shade model toward book heavily.
    alt["p_blend"] = 0.75 * alt["p_book"] + 0.25 * alt["p_model"]
    alt["ev"] = alt["p_blend"] * (alt["dec"] - 1) - (1 - alt["p_blend"])
    actual = cur_all[["PLAYER_ID", "OPP", "GAME_DATE", "PTS"]].copy()
    actual["date"] = actual["GAME_DATE"].dt.date
    alt = alt.merge(actual[["PLAYER_ID", "OPP", "date", "PTS"]], left_on=["player_id", "opp", "date"],
                    right_on=["PLAYER_ID", "OPP", "date"], how="left")
    alt = alt[alt["PTS"].notna()]
    alt["won"] = alt["PTS"] > alt["line"]
    alt["profit"] = np.where(alt["won"], alt["dec"] - 1, -1.0)
    print(f"\n==== C. ALT POINTS LINES ({season}, {alt['date'].nunique()} days, {len(alt)} rungs priced) ====")
    alt["dist"] = pd.cut((alt["line"] - alt["mu"]) / alt["mu"].clip(lower=1), [-1, -0.3, -0.15, 0, 0.15, 0.3, 1])
    print("Calibration of raw model vs book by rung distance from projection:")
    print(alt.groupby("dist", observed=True).agg(n=("won", "size"), p_model=("p_model", "mean"), p_book=("p_book", "mean"),
                                                  actual=("won", "mean")).round(3).to_string())
    for thr in (0.0, 0.02, 0.04, 0.06):
        g = alt[alt["ev"] >= thr].sort_values("ev", ascending=False).drop_duplicates(["date", "pk"])
        if len(g):
            print(f"  EV>={thr:.2f}: n={len(g)}, hit {g['won'].mean():.3f}, ROI {g['profit'].mean():+.4f}, avg price {g['price'].mean():+.0f}")
    alt.to_pickle(OUT / "bt_alt_points.pkl")
    # D. double-doubles — only a rough check: use book price vs naive frequency from last 25 games
    dd = pd.DataFrame(dd_rows)
    if not dd.empty:
        def dd_rate(pid, d):
            h = pd.concat([pri[pri["PLAYER_ID"] == pid], logs[(logs["PLAYER_ID"] == pid) & (logs["GAME_DATE"] < pd.Timestamp(d))]])
            h = h[h["MIN"] > 0].tail(25)
            if len(h) < 8:
                return np.nan
            return ((h[["PTS", "REB", "AST"]] >= 10).sum(axis=1) >= 2).mean()
        dd["p_hist"] = [dd_rate(p, d) for p, d in zip(dd["player_id"], dd["date"])]
        dd = dd[dd["p_hist"].notna()]
        dd["p_book"] = dd["price"].apply(american_to_prob)
        dd["dec"] = np.where(dd["price"] > 0, dd["price"] / 100 + 1, 100 / -dd["price"] + 1)
        dd["p_side"] = np.where(dd["side"] == "yes", dd["p_hist"], 1 - dd["p_hist"])
        dd["p_blend"] = 0.6 * dd["p_book"] + 0.4 * dd["p_side"]
        dd["ev"] = dd["p_blend"] * (dd["dec"] - 1) - (1 - dd["p_blend"])
        a2 = cur_all[["PLAYER_ID", "OPP", "GAME_DATE", "PTS", "REB", "AST"]].copy()
        a2["date"] = a2["GAME_DATE"].dt.date
        a2["dd"] = ((a2[["PTS", "REB", "AST"]] >= 10).sum(axis=1) >= 2)
        dd = dd.merge(a2[["PLAYER_ID", "OPP", "date", "dd"]], left_on=["player_id", "opp", "date"],
                      right_on=["PLAYER_ID", "OPP", "date"], how="left")
        dd = dd[dd["dd"].notna()]
        dd["won"] = np.where(dd["side"] == "yes", dd["dd"], ~dd["dd"].astype(bool))
        dd["profit"] = np.where(dd["won"], dd["dec"] - 1, -1.0)
        print(f"\n==== D. DOUBLE-DOUBLE ({len(dd)} prices; model = last-25-game frequency, blended 60/40 toward book) ====")
        for thr in (0.0, 0.03, 0.06):
            g = dd[dd["ev"] >= thr].sort_values("ev", ascending=False).drop_duplicates(["date", "player", "side"])
            if len(g):
                print(f"  EV>={thr:.2f}: n={len(g)}, hit {g['won'].mean():.3f}, ROI {g['profit'].mean():+.4f}, avg price {g['price'].mean():+.0f}")


if __name__ == "__main__":
    morning_test()
    blowout_rest_test()
    alt_lines_test()
