"""
Injury report from ESPN + usage redistribution when teammates are out.

Statuses ESPN uses: "Out", "Day-To-Day" (and occasionally "Doubtful"/"Questionable").
  Out            -> removed from projections; minutes & usage redistributed to teammates.
  Day-To-Day     -> projected normally, flagged `dtd`; stake halved in pick_bets.
data/out.csv (column `player`) still works as a manual override for late scratches.
"""
from datetime import date

import pandas as pd
import requests

from config import DATA_DIR
from fetch_lines import norm_name
from fetch_stats import SITE, HDR, teams

OUT_STATUSES = {"out", "doubtful", "injured reserve", "suspension", "suspended", "not with team"}
DTD_STATUSES = {"day-to-day", "questionable", "probable", "game time decision"}

# Position groups for minutes redistribution and per-position matchup adjustments
POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "GF": "F", "FG": "F",
             "C": "C", "FC": "C", "CF": "C"}


def fetch_injuries() -> pd.DataFrame:
    """One row per injured player: player_key, team, status, is_out, is_dtd, return_date, note."""
    cache = DATA_DIR / "injuries_latest.csv"
    try:
        js = requests.get(f"{SITE}/injuries", headers=HDR, timeout=30).json()
    except Exception as e:  # noqa: BLE001
        print(f"  injury feed failed ({e}); using cached copy if present")
        return pd.read_csv(cache) if cache.exists() else pd.DataFrame(
            columns=["player", "player_key", "team", "pos", "status", "is_out", "is_dtd", "return_date", "note"])
    name_to_abbr = teams().set_index("TEAM_NAME")["ABBR"].to_dict()
    rows = []
    for t in js.get("injuries", []):
        abbr = name_to_abbr.get(t.get("displayName"), "")
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            status = (i.get("status") or "").strip()
            sl = status.lower()
            rows.append({
                "player": a.get("displayName", ""), "player_key": norm_name(a.get("displayName", "")),
                "team": abbr, "pos": a.get("position", {}).get("abbreviation", ""),
                "status": status, "is_out": sl in OUT_STATUSES, "is_dtd": sl in DTD_STATUSES,
                "return_date": (i.get("details") or {}).get("returnDate", ""),
                "note": (i.get("shortComment") or "")[:160], "as_of": date.today().isoformat(),
            })
    df = pd.DataFrame(rows)
    # manual overrides
    manual = DATA_DIR / "out.csv"
    if manual.exists():
        for n in pd.read_csv(manual)["player"].dropna():
            df = pd.concat([df, pd.DataFrame([{"player": n, "player_key": norm_name(n), "team": "", "pos": "",
                                               "status": "Out (manual)", "is_out": True, "is_dtd": False,
                                               "return_date": "", "note": "data/out.csv", "as_of": date.today().isoformat()}])])
    df.to_csv(cache, index=False)
    return df.reset_index(drop=True)


def pos_group(pos: str) -> str:
    return POS_GROUP.get((pos or "").upper(), "F")
