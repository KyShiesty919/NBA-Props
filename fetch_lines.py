"""
Pull NBA player-prop lines from The Odds API and flatten to one row per
(event, book, market, player, side). Every pull is snapshotted to data/lines/
so you can grade closing-line value later: run once in the morning (bet-time
snapshot) and once again ~15 min before the first tip (closing snapshot).
"""
import gzip
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import ODDS_API_KEY, ODDS_REGIONS, MARKETS, EXCLUDE_BOOKS, TEAM_ABBR, DATA_DIR

API = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
LINES_DIR = DATA_DIR / "lines"
LINES_DIR.mkdir(exist_ok=True)


def norm_name(s: str) -> str:
    """'Nikola Jokić' -> 'nikola jokic'; strips suffixes/punctuation for matching."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split())


def american_to_prob(price: float) -> float:
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def prob_to_american(p: float) -> int:
    if p <= 0 or p >= 1:
        return 0
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def fetch_events() -> list:
    r = requests.get(f"{API}/sports/{SPORT}/events", params={"apiKey": ODDS_API_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_event_props(event_id: str) -> dict:
    r = requests.get(
        f"{API}/sports/{SPORT}/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "regions": ODDS_REGIONS,
                "markets": ",".join(MARKETS), "oddsFormat": "american"},
        timeout=30,
    )
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining")
    if remaining:
        print(f"  Odds API credits remaining: {remaining}", file=sys.stderr)
    return r.json()


CLOSE_WINDOW_MIN = 75   # --close grabs games tipping within this many minutes


def fetch_lines(tag: str = "open") -> pd.DataFrame:
    """
    tag='open'  : pull every game on today's slate that hasn't tipped (bet-time snapshot).
    tag='close' : pull only games tipping within CLOSE_WINDOW_MIN that aren't already
                  in today's close file, and merge them in. Safe to run hourly all
                  afternoon/evening — each game gets captured once, right before tip.
    """
    if not ODDS_API_KEY:
        raise SystemExit("ODDS_API_KEY not set. export ODDS_API_KEY=... (see README)")
    events = fetch_events()
    now = datetime.now(timezone.utc)
    today_local = datetime.now().date()
    snap = LINES_DIR / f"{today_local.isoformat()}_{tag}.json.gz"
    existing = _read_snap(snap)["data"] if (tag == "close" and snap.exists()) else []
    have = {item["event"]["id"] for item in existing}
    raw = []
    for ev in events:
        tip = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if tip.astimezone().date() != today_local or tip < now or ev["id"] in have:
            continue
        if tag == "close" and (tip - now).total_seconds() > CLOSE_WINDOW_MIN * 60:
            continue
        raw.append({"event": ev, "odds": fetch_event_props(ev["id"])})
        time.sleep(0.3)
    merged = existing + raw
    if raw or not snap.exists():
        _write_snap(snap, {"fetched_utc": now.isoformat(), "data": merged})
    print(f"{tag}: captured {len(raw)} new events ({len(merged)} total) -> {snap}", file=sys.stderr)
    return flatten(raw if tag == "open" else merged, now)


def flatten(raw: list, fetched: datetime) -> pd.DataFrame:
    rows = []
    for item in raw:
        ev, odds = item["event"], item["odds"]
        home, away = ev["home_team"], ev["away_team"]
        h, a = TEAM_ABBR.get(home, home[:3].upper()), TEAM_ABBR.get(away, away[:3].upper())
        for bk in odds.get("bookmakers", []):
            if bk["key"] in EXCLUDE_BOOKS:
                continue
            for m in bk.get("markets", []):
                stat = MARKETS.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    rows.append({
                        "event_id": ev["id"], "game": f"{a} @ {h}", "home": h, "away": a,
                        "tip_utc": ev["commence_time"], "book": bk["key"], "stat": stat,
                        "player": o.get("description", ""), "player_key": norm_name(o.get("description", "")),
                        "side": o["name"].lower(), "line": float(o["point"]), "price": int(o["price"]),
                        "implied": american_to_prob(o["price"]), "fetched_utc": fetched.isoformat(),
                    })
    return pd.DataFrame(rows)


def _read_snap(p: Path) -> dict:
    return json.loads(gzip.decompress(p.read_bytes()).decode())


def _write_snap(p: Path, obj: dict):
    p.write_bytes(gzip.compress(json.dumps(obj).encode()))


def load_snapshot(day: str, tag: str) -> pd.DataFrame | None:
    p = LINES_DIR / f"{day}_{tag}.json.gz"
    if not p.exists():
        return None
    js = _read_snap(p)
    return flatten(js["data"], datetime.fromisoformat(js["fetched_utc"]))


def devig(lines: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative de-vig per (book, player, stat, line): fair_over = imp_over/(imp_over+imp_under).
    Then market consensus = median fair prob across books AT THE SAME LINE,
    and the 'best' price per side across books.
    """
    if lines.empty:
        return lines
    key = ["event_id", "book", "stat", "player_key", "line"]
    piv = lines.pivot_table(index=key, columns="side", values="implied", aggfunc="first").reset_index()
    piv = piv.dropna(subset=["over", "under"])
    tot = piv["over"] + piv["under"]
    piv["fair_over"] = piv["over"] / tot
    piv["hold"] = tot - 1
    fair = piv[key + ["fair_over", "hold"]]
    lines = lines.merge(fair, on=key, how="left")
    cons = (fair.groupby(["event_id", "stat", "player_key", "line"])
                .agg(mkt_fair_over=("fair_over", "median"), n_books=("book", "nunique"))
                .reset_index())
    return lines.merge(cons, on=["event_id", "stat", "player_key", "line"], how="left")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "open"
    df = fetch_lines(tag)
    print(df.head(20).to_string())
