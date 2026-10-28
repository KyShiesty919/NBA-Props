"""
Grade the picks ledger against actual box scores, compute hit rate, flat ROI,
calibration, and closing-line value (CLV) when a 'close' snapshot exists.

    python grade.py            # grade everything ungraded, print report
    python grade.py --demo     # grade using synthetic results (pipeline test)
"""
import os
import sys
if "--demo" in sys.argv:
    os.environ["LINES_SUBDIR"] = "lines_demo"
from datetime import date, datetime

import numpy as np
import pandas as pd

from config import LEDGER, OUT_DIR
from fetch_lines import load_snapshot, devig, american_to_prob


def _result(row, actual):
    line = row["line"]
    if actual == line:
        return "push"
    won = actual > line if row["side"] == "over" else actual < line
    return "win" if won else "loss"


def _profit(row):
    if row["result"] == "push" or pd.isna(row["result"]):
        return 0.0
    dec = (row["price"] / 100 + 1) if row["price"] > 0 else (100 / -row["price"] + 1)
    return row["stake"] * (dec - 1) if row["result"] == "win" else -row["stake"]


def grade_ledger(box_fetcher, demo: bool = False) -> pd.DataFrame:
    if not LEDGER.exists():
        print("No ledger yet.")
        return pd.DataFrame()
    led = pd.read_csv(LEDGER)
    for c in ("result",):
        led[c] = led[c].astype(object)
    for c in ("actual", "close_price", "close_fair_prob", "clv", "profit"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    todo = led[led["result"].isna() & (pd.to_datetime(led["date"]).dt.date < date.today() if not demo else True)]
    for day in sorted(todo["date"].unique()):
        d = datetime.fromisoformat(day).date()
        try:
            box = box_fetcher(d)
        except Exception as e:  # noqa: BLE001
            print(f"  could not fetch box scores for {day}: {e}")
            continue
        actual = box.set_index("PLAYER_ID")
        idx = led.index[(led["date"] == day) & led["result"].isna()]
        for i in idx:
            r = led.loc[i]
            if r["player_id"] in actual.index:
                a = float(actual.loc[r["player_id"], r["stat"]])
                led.loc[i, "actual"] = a
                led.loc[i, "result"] = _result(r, a)
            else:
                led.loc[i, "result"] = "void"   # DNP
        # CLV: compare to closing snapshot if present
        close = load_snapshot(day, "close")
        if close is not None and not close.empty:
            close = devig(close)
            for i in idx:
                r = led.loc[i]
                m = close[(close["event_id"] == r["event_id"]) & (close["stat"] == r["stat"]) &
                          (close["player_key"] == r["player_key"]) & (close["side"] == r["side"]) &
                          (close["line"] == r["line"])]
                if m.empty:
                    continue
                mk = m["mkt_fair_over"].iloc[0]
                close_fair = mk if r["side"] == "over" else 1 - mk
                same_book = m[m["book"] == r["book"]]
                led.loc[i, "close_price"] = int(same_book["price"].iloc[0]) if not same_book.empty else np.nan
                led.loc[i, "close_fair_prob"] = round(close_fair, 4)
                led.loc[i, "clv"] = round(close_fair - r["mkt_fair_prob"], 4)
    led["profit"] = led.apply(_profit, axis=1)
    led.to_csv(LEDGER, index=False)
    return led


def report(led: pd.DataFrame) -> pd.DataFrame:
    g = led[led["result"].isin(["win", "loss", "push"])].copy()
    if g.empty:
        print("Nothing graded yet.")
        return pd.DataFrame()
    g["win"] = (g["result"] == "win").astype(int)
    g["dec"] = g["result"] != "push"

    def summarise(x):
        n = int(x["dec"].sum())
        w = int(x["win"].sum())
        return pd.Series({
            "bets": n, "wins": w, "hit_rate": round(w / n, 3) if n else np.nan,
            "staked": round(x["stake"].sum(), 2), "profit": round(x["profit"].sum(), 2),
            "roi": round(x["profit"].sum() / x["stake"].sum(), 4) if x["stake"].sum() else np.nan,
            "avg_blended_prob": round(x["blended_prob"].mean(), 3),
            "avg_clv": round(x["clv"].mean(), 4) if "clv" in x and x["clv"].notna().any() else np.nan,
            "clv_pos_rate": round((x["clv"] > 0).mean(), 3) if "clv" in x and x["clv"].notna().any() else np.nan,
        })

    if "status" in g.columns:
        g["status"] = g["status"].fillna("")
        live = g[~g["status"].str.startswith("dropped")]
        overall = pd.concat([summarise(live).to_frame("ALL (excl. dropped)").T,
                             summarise(g[g["status"].str.startswith("dropped")]).to_frame("dropped by re-score").T])
        g = live
    else:
        overall = summarise(g).to_frame("ALL").T
    by_stat = g.groupby("stat").apply(summarise)
    g["tier_base"] = g["tier"].fillna("").str.replace(" (early)", "", regex=False) if "tier" in g.columns else ""
    by_tier = g.groupby("tier_base").apply(summarise)
    by_side = g.groupby("side").apply(summarise)
    # Calibration: does a 58% blended prob win 58% of the time?
    bins = [0, 0.52, 0.55, 0.58, 0.62, 0.66, 1.0]
    g["bucket"] = pd.cut(g["blended_prob"], bins)
    calib = g[g["dec"]].groupby("bucket", observed=True).agg(
        n=("win", "size"), predicted=("blended_prob", "mean"), actual=("win", "mean")).round(3)
    calib["gap"] = (calib["actual"] - calib["predicted"]).round(3)

    # Significance: flat-stake z-test on profit vs 0 (rough; needs 200+ bets to mean much)
    n = int(g["dec"].sum())
    pr = g.loc[g["dec"], "profit"] / g.loc[g["dec"], "stake"]
    z = pr.mean() / (pr.std(ddof=1) / np.sqrt(n)) if n > 1 and pr.std() > 0 else np.nan

    print("\n=== OVERALL ===");   print(overall.to_string())
    print("\n=== BY TIER ===");   print(by_tier.to_string())
    print("\n=== BY PROP ===");   print(by_stat.to_string())
    print("\n=== BY SIDE ===");   print(by_side.to_string())
    print("\n=== CALIBRATION ==="); print(calib.to_string())
    print(f"\nz-score of ROI vs 0: {z:.2f}  (|z| > 2 with 200+ bets before believing it)")
    if n < 200:
        print(f"Only {n} graded bets — treat every number above as noise.")
    out = OUT_DIR / "performance_report.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        overall.to_excel(xw, sheet_name="Overall")
        by_tier.to_excel(xw, sheet_name="By tier")
        by_stat.to_excel(xw, sheet_name="By prop")
        by_side.to_excel(xw, sheet_name="By side")
        calib.to_excel(xw, sheet_name="Calibration")
        g.drop(columns=["win", "dec", "bucket", "tier_base"], errors="ignore").to_excel(xw, sheet_name="Graded bets", index=False)
    print(f"\nReport -> {out}")
    return overall


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    if demo:
        from demo_data import demo_box_scores as fetcher
    else:
        from fetch_stats import box_scores_for_date as fetcher
    report(grade_ledger(fetcher, demo=demo))
