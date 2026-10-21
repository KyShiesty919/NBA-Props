"""
Daily driver.

    python run_daily.py              # pull stats + lines, project, write outputs/YYYY-MM-DD_props.xlsx
    python run_daily.py --close      # ONLY re-pull lines and save a closing snapshot (run ~15 min before first tip)
    python run_daily.py --demo       # synthetic data, no network — pipeline test
    python run_daily.py --no-ledger  # don't append tonight's picks to the ledger
    python run_daily.py --reuse-lines # re-run the model on today's already-saved lines (no API credits)

Typical day:
   morning:   python run_daily.py          -> bet from the "Bets" sheet
   pre-tip:   python run_daily.py --close  -> captures closing lines for CLV
   next day:  python grade.py              -> hit rate / ROI / calibration / CLV
"""
import sys
from datetime import date

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

import config
from config import LEDGER, OUT_DIR, SEASON, PRIOR_SEASON
from fetch_lines import devig
from model import blend_team_stats, build_projections, evaluate, pick_bets

LEDGER_COLS = ["date", "game", "player_id", "player", "player_key", "team", "opp", "stat", "side",
               "line", "book", "price", "projection", "model_prob", "mkt_fair_prob", "blended_prob",
               "edge", "ev_per_dollar", "stake", "event_id", "fetched_utc",
               "actual", "result", "close_price", "close_fair_prob", "clv", "profit"]


def main():
    args = set(sys.argv[1:])
    demo = "--demo" in args
    today = date.today()

    if "--close" in args and not demo:
        from fetch_lines import fetch_lines
        fetch_lines("close")
        print("Closing snapshot saved. Run grade.py tomorrow.")
        return

    # ---- data ------------------------------------------------------------
    if demo:
        from demo_data import make_demo
        cur, prior, t_cur, t_prior, lines = make_demo(today)
        print("DEMO MODE: synthetic data.")
    else:
        from fetch_stats import load_all
        from fetch_lines import fetch_lines
        print(f"Pulling game logs {SEASON} + {PRIOR_SEASON} and team stats ...")
        cur, prior, t_cur, t_prior = load_all()
        from fetch_lines import load_snapshot
        lines = load_snapshot(today.isoformat(), "open") if "--reuse-lines" in args else None
        if lines is None:
            print("Pulling lines from The Odds API ...")
            lines = fetch_lines("open")
        else:
            print("Reusing today's saved open lines.")
    if lines.empty:
        print("No lines returned — no games today, or all games already tipped.")
        return

    # ---- model -------------------------------------------------------------
    lines = devig(lines)
    slate = lines[["home", "away"]].drop_duplicates()
    teams = blend_team_stats(t_cur, t_prior)
    proj = build_projections(cur, prior, teams, slate, today)
    # Manual injury/rest overrides: data/out.csv with a 'player' column (one name per row)
    out_file = config.DATA_DIR / "out.csv"
    if out_file.exists():
        from fetch_lines import norm_name
        out_keys = {norm_name(n) for n in pd.read_csv(out_file)["player"].dropna()}
        proj = proj[~proj["player_key"].isin(out_keys)]
        print(f"Excluded {len(out_keys)} players listed in data/out.csv")
    print(f"{len(proj)} players projected across {len(slate)} games.")
    evald = evaluate(proj, lines)
    bets = pick_bets(evald)
    print(f"{len(evald)} priced lines evaluated, {len(bets)} clear the {config.MIN_EDGE:.0%} edge bar.")

    # ---- outputs -----------------------------------------------------------
    out = OUT_DIR / f"{today.isoformat()}_props{'_demo' if demo else ''}.xlsx"
    write_xlsx(out, bets, evald, proj, lines)
    print(f"Wrote {out}")

    if "--no-ledger" not in args and not bets.empty:
        if LEDGER.exists():   # re-run same day: replace today's picks
            old = pd.read_csv(LEDGER)
            old[old["date"] != today.isoformat()].to_csv(LEDGER, index=False)
        append_ledger(bets, today)
        print(f"Appended {len(bets)} picks to {LEDGER}")

    if not bets.empty:
        cols = ["player", "stat", "side", "line", "book", "price", "projection", "blended_prob", "edge", "stake"]
        print("\nTOP BETS\n" + bets[cols].head(15).to_string(index=False))


def append_ledger(bets: pd.DataFrame, today: date):
    rows = bets.copy()
    rows.insert(0, "date", today.isoformat())
    for c in LEDGER_COLS:
        if c not in rows.columns:
            rows[c] = pd.NA
    rows = rows[LEDGER_COLS]
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        # don't double-log the same pick if you run twice in a day
        key = ["date", "player_key", "stat", "side"]
        rows = rows.merge(old[key].drop_duplicates(), on=key, how="left", indicator=True)
        rows = rows[rows["_merge"] == "left_only"].drop(columns="_merge")
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(LEDGER, index=False)


def write_xlsx(path, bets, evald, proj, lines):
    bet_cols = ["game", "player", "team", "opp", "stat", "side", "line", "book", "price",
                "projection", "minutes", "model_prob", "mkt_fair_prob", "blended_prob", "edge",
                "ev_per_dollar", "fair_price", "stake", "n_books", "n_cur", "new_team", "b2b",
                "market_weight", "disagree", "suspicious"]
    proj_cols = ["game", "player", "team", "opp", "home", "b2b", "minutes", "n_cur", "new_team",
                 "mu_PTS", "mu_REB", "mu_AST", "mu_FG3M", "mu_PRA"]
    knobs = pd.DataFrame({"setting": ["MARKET_WEIGHT", "MIN_EDGE", "MIN_BOOKS", "KELLY_FRACTION",
                                      "MAX_BET_PCT", "BANKROLL", "PRIOR_SHRINK_K", "OPP_ADJ_STRENGTH"],
                          "value": [config.MARKET_WEIGHT, config.MIN_EDGE, config.MIN_BOOKS,
                                    config.KELLY_FRACTION, config.MAX_BET_PCT, config.BANKROLL,
                                    config.PRIOR_SHRINK_K, config.OPP_ADJ_STRENGTH]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        (bets[[c for c in bet_cols if c in bets.columns]] if not bets.empty else pd.DataFrame(columns=bet_cols)
         ).to_excel(xw, sheet_name="Bets", index=False)
        evald[[c for c in bet_cols if c in evald.columns]].to_excel(xw, sheet_name="All lines", index=False)
        proj[[c for c in proj_cols if c in proj.columns]].sort_values(["game", "mu_PTS"], ascending=[True, False]
                                                                       ).to_excel(xw, sheet_name="Projections", index=False)
        lines.to_excel(xw, sheet_name="Raw lines", index=False)
        knobs.to_excel(xw, sheet_name="Settings", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3A5F")
            for i, col in enumerate(ws.columns, 1):
                w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[get_column_letter(i)].width = min(max(w + 2, 8), 28)
            ws.auto_filter.ref = ws.dimensions
            for name, fmt in (("model_prob", "0.0%"), ("mkt_fair_prob", "0.0%"), ("blended_prob", "0.0%"),
                              ("edge", "+0.0%"), ("ev_per_dollar", "+0.0%"), ("stake", "$#,##0.00")):
                for j, c in enumerate(ws[1], 1):
                    if c.value == name:
                        for cell in list(ws.columns)[j - 1][1:]:
                            cell.number_format = fmt


if __name__ == "__main__":
    main()
