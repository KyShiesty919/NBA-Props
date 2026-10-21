"""
Re-simulate betting rules on the cached real-line backtest (outputs/bt_lines_*_all.pkl)
with different market weights / edge thresholds / filters. No API calls.

Fit on 2023-24 + 2024-25, validate blind on 2025-26.
"""
import itertools
import numpy as np
import pandas as pd

import config

pd.set_option("display.width", 220)
FIT, TEST = ["2023-24", "2024-25"], ["2025-26"]


def load():
    ev = pd.concat([pd.read_pickle(p) for p in config.OUT_DIR.glob("bt_lines_*_all.pkl")], ignore_index=True)
    ev = ev[ev["actual"].notna() & ev["mkt_fair_prob"].notna()].copy()
    ev["dec"] = np.where(ev["price"] > 0, ev["price"] / 100 + 1, 100 / -ev["price"] + 1)
    ev["won"] = np.where(ev["side"] == "over", ev["actual"] > ev["line"], ev["actual"] < ev["line"])
    ev["push"] = ev["actual"] == ev["line"]
    ev["inj"] = ev["inj_note"].fillna("").astype(str).str.len() > 0
    ev["book_fair"] = ev["book_implied"]        # this book's price as prob (with vig)
    return ev


def simulate(ev, mw, min_edge, stats=None, sides=None, no_inj=False, min_n_cur=0, min_books=2,
             max_disagree=None, price_vs_consensus=None):
    d = ev
    if stats:
        d = d[d["stat"].isin(stats)]
    if sides:
        d = d[d["side"].isin(sides)]
    if no_inj:
        d = d[~d["inj"]]
    d = d[(d["n_cur"] >= min_n_cur) & (d["n_books"] >= min_books) & d["book"].isin(config.BETTABLE_BOOKS)]
    if max_disagree is not None:
        d = d[d["disagree"] <= max_disagree]
    else:
        d = d[~d["suspicious"]]
    p = mw * d["mkt_fair_prob"] + (1 - mw) * d["model_prob"]
    edge = p - d["mkt_fair_prob"]
    ev_ = p * (d["dec"] - 1) - (1 - p)
    d = d.assign(p=p, edge=edge, ev_=ev_)
    d = d[(d["edge"] >= min_edge) & (d["ev_"] > 0)]
    if price_vs_consensus is not None:
        # only bet where THIS book's implied prob is at least X below the de-vigged consensus (a soft price)
        d = d[(d["mkt_fair_prob"] - d["book_fair"]) >= price_vs_consensus]
    # one bet per player/stat per day: best EV, never both sides
    d = d.sort_values("ev_", ascending=False).drop_duplicates(["date", "player_key", "stat"])
    g = d[~d["push"]]
    n = len(g)
    if n == 0:
        return {"bets": 0}
    profit = np.where(g["won"], g["dec"] - 1, -1.0)
    roi = profit.mean()
    z = roi / (profit.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"bets": n, "per_season": n / g["season"].nunique(), "hit": g["won"].mean(), "roi": roi, "z": z,
            "avg_price": g["price"].mean(), "avg_edge": g["edge"].mean()}


if __name__ == "__main__":
    ev = load()
    fit, test = ev[ev["season"].isin(FIT)], ev[ev["season"].isin(TEST)]
    print(f"fit rows {len(fit)}, test rows {len(test)}")

    print("\n=== 1. MARKET WEIGHT x MIN EDGE (all props, production filters) — FIT seasons ===")
    rows = []
    for mw, me in itertools.product([0.35, 0.5, 0.65, 0.75, 0.85, 0.9], [0.02, 0.03, 0.04, 0.06]):
        rows.append({"mw": mw, "min_edge": me, **simulate(fit, mw, me)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 2. FILTERS at mw=0.75, min_edge=0.03 — FIT seasons ===")
    rows = []
    for label, kw in [("all", {}), ("REB+FG3M only", {"stats": ["REB", "FG3M"]}),
                      ("REB+FG3M+AST", {"stats": ["REB", "FG3M", "AST"]}), ("unders only", {"sides": ["under"]}),
                      ("no injury bets", {"no_inj": True}), ("n_cur>=10", {"min_n_cur": 10}),
                      ("3+ books", {"min_books": 3}), ("disagree<=0.15", {"max_disagree": 0.15}),
                      ("soft price >=2%", {"price_vs_consensus": 0.02}),
                      ("soft price >=3%", {"price_vs_consensus": 0.03}),
                      ("REB+FG3M, no inj, n_cur>=10", {"stats": ["REB", "FG3M"], "no_inj": True, "min_n_cur": 10}),
                      ("soft>=2% + no inj + n_cur>=10", {"price_vs_consensus": 0.02, "no_inj": True, "min_n_cur": 10})]:
        rows.append({"filter": label, **simulate(fit, 0.75, 0.03, **kw)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 3. PURE LINE-SHOPPING baseline (mw=1.0: ignore the model, bet soft prices vs consensus) — FIT ===")
    rows = []
    for thr in (0.02, 0.03, 0.04, 0.05):
        rows.append({"soft_price>=": thr, **simulate(fit, 1.0, -1, price_vs_consensus=thr)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 4. BLIND TEST on 2025-26 of the candidates ===")
    cands = [("production (mw .35, edge .04)", dict(mw=0.35, min_edge=0.04)),
             ("mw .75, edge .03", dict(mw=0.75, min_edge=0.03)),
             ("mw .75, edge .03, no inj, n_cur>=10", dict(mw=0.75, min_edge=0.03, no_inj=True, min_n_cur=10)),
             ("mw .75, edge .03, REB+FG3M", dict(mw=0.75, min_edge=0.03, stats=["REB", "FG3M"])),
             ("mw .75, edge .02, soft price>=2%", dict(mw=0.75, min_edge=0.02, price_vs_consensus=0.02)),
             ("mw .85, edge .02, soft price>=2%, no inj", dict(mw=0.85, min_edge=0.02, price_vs_consensus=0.02, no_inj=True)),
             ("line shopping only, soft>=3%", dict(mw=1.0, min_edge=-1, price_vs_consensus=0.03))]
    rows = []
    for label, kw in cands:
        rows.append({"candidate": label, "set": "FIT", **simulate(fit, **kw)})
        rows.append({"candidate": label, "set": "TEST 2025-26", **simulate(test, **kw)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
