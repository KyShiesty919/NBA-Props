"""
Player game logs + team stats from ESPN's public JSON API.

(stats.nba.com blocks datacenter IPs, so ESPN is the source. No key needed.)

Player logs: one call per rostered player per season (~500/season), threaded.
Team stats : scoreboard per calendar day (~170/season), team box totals -> opponent
             allowed PTS/REB/AST/FG3M and a possessions proxy for pace.
Prior season is cached forever; current season refreshes once per day.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from pathlib import Path

import pandas as pd
import requests

from config import SEASON, PRIOR_SEASON, DATA_DIR

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
# ESPN abbreviations that differ from the common ones used by books
ESPN_TO_STD = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
SEASON_START = {"2022-23": date(2022, 10, 18), "2023-24": date(2023, 10, 24), "2024-25": date(2024, 10, 22),
                "2025-26": date(2025, 10, 21), "2026-27": date(2026, 10, 20)}


def _year(season: str) -> int:        # "2026-27" -> 2027 (ESPN uses the end year)
    return int(season[:4]) + 1


def _get(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HDR, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _stamp(p):
    return Path(str(p) + ".fetched")


def _mark_fetched(p):
    _stamp(p).write_text(date.today().isoformat())


def _cache_is_today(p) -> bool:
    """Freshness by an explicit stamp, NOT mtime: git pull rewrites files and would make stale data look fresh."""
    p = Path(p)
    if not p.exists():
        return False
    st = _stamp(p)
    return st.exists() and st.read_text().strip() == date.today().isoformat()


def std(abbr: str) -> str:
    return ESPN_TO_STD.get(abbr, abbr)


# ------------------------------------------------------------------ teams & rosters
def teams() -> pd.DataFrame:
    p = DATA_DIR / "espn_teams.csv"
    if p.exists():
        return pd.read_csv(p)
    js = _get(f"{SITE}/teams", {"limit": 50})
    rows = [{"TEAM_ID": int(t["team"]["id"]), "TEAM_NAME": t["team"]["displayName"],
             "ABBR": std(t["team"]["abbreviation"])} for t in js["sports"][0]["leagues"][0]["teams"]]
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return df


def rosters(force=False) -> pd.DataFrame:
    p = DATA_DIR / "espn_rosters.csv"
    if _cache_is_today(p) and not force:
        return pd.read_csv(p)
    rows = []
    for _, t in teams().iterrows():
        js = _get(f"{SITE}/teams/{t['TEAM_ID']}/roster")
        for a in (js or {}).get("athletes", []):
            rows.append({"PLAYER_ID": int(a["id"]), "PLAYER_NAME": a["displayName"],
                         "TEAM_ID": t["TEAM_ID"], "TEAM_ABBREVIATION": t["ABBR"],
                         "POS": a.get("position", {}).get("abbreviation", "")})
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    _mark_fetched(p)
    return df


# ------------------------------------------------------------------ player game logs
def _et_date(iso: str) -> str:
    try:
        from datetime import datetime as _dt
        t = _dt.fromisoformat(iso.replace("Z", "+00:00"))
        return (t - timedelta(hours=5)).date().isoformat()
    except Exception:  # noqa: BLE001
        return iso[:10]


def _parse_gamelog(pid, name, js, season) -> list:
    if not js or "seasonTypes" not in js:
        return []
    names = js.get("names", [])
    idx = {n: i for i, n in enumerate(names)}
    need = ["minutes", "totalRebounds", "assists", "points", "threePointFieldGoalsMade-threePointFieldGoalsAttempted"]
    if any(n not in idx for n in need):
        return []
    events = js.get("events", {})
    rows = []
    for st in js["seasonTypes"]:
        if "Regular Season" not in st.get("displayName", ""):
            continue
        for cat in st.get("categories", []):
            for ev in cat.get("events", []):
                s = ev["stats"]
                meta = events.get(ev["eventId"], {})
                try:
                    rows.append({
                        "PLAYER_ID": pid, "PLAYER_NAME": name,
                        "TEAM_ABBREVIATION": std(meta.get("team", {}).get("abbreviation", "")),
                        # ESPN stamps UTC; shift 5h so a 10:30pm ET tip lands on the ET calendar day
                        "GAME_ID": ev["eventId"], "GAME_DATE": _et_date(meta.get("gameDate", "")),
                        "HOME": meta.get("atVs") == "vs",
                        "OPP": std(meta.get("opponent", {}).get("abbreviation", "")),
                        "MIN": float(s[idx["minutes"]]), "PTS": int(s[idx["points"]]),
                        "REB": int(s[idx["totalRebounds"]]), "AST": int(s[idx["assists"]]),
                        "FG3M": int(s[idx[need[4]]].split("-")[0]), "SEASON": season,
                    })
                except (ValueError, IndexError):
                    continue
    return rows


def player_game_logs(season: str, force=False) -> pd.DataFrame:
    cache = DATA_DIR / f"gamelogs_{season}.csv"
    fresh = cache.exists() and (season != SEASON or _cache_is_today(cache))
    if fresh and not force:
        return _finish(pd.read_csv(cache))
    ros = rosters()
    yr = _year(season)
    rows = []

    def one(r):
        js = _get(f"{WEB}/athletes/{r.PLAYER_ID}/gamelog", {"season": yr})
        return _parse_gamelog(r.PLAYER_ID, r.PLAYER_NAME, js, season)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, r) for r in ros.itertuples()]
        for f in as_completed(futs):
            rows.extend(f.result())
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
                                   "HOME", "OPP", "MIN", "PTS", "REB", "AST", "FG3M", "SEASON"])
    # TEAM_ID from roster (current team) — used for continuity with model.py
    df = df.merge(ros[["PLAYER_ID", "TEAM_ID"]], on="PLAYER_ID", how="left")
    df.to_csv(cache, index=False)
    _mark_fetched(cache)
    return _finish(df)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


# ------------------------------------------------------------------ team stats via scoreboard
def _scoreboard_day(d: date) -> list:
    js = _get(f"{SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 50})
    rows = []
    for e in (js or {}).get("events", []):
        if e.get("season", {}).get("type") != 2:          # 2 = regular season
            continue
        if e["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        comp = e["competitions"][0]
        cs = comp["competitors"]
        if len(cs) != 2:
            continue
        box = {}
        for c in cs:
            st = {s["name"]: float(s.get("displayValue", 0) or 0) for s in c.get("statistics", [])}
            box[c["homeAway"]] = {"TEAM_ID": int(c["team"]["id"]), "ABBR": std(c["team"]["abbreviation"]),
                                  "PTS": float(c.get("score", 0) or 0), "REB": st.get("rebounds", 0),
                                  "AST": st.get("assists", 0), "FG3M": st.get("threePointFieldGoalsMade", 0),
                                  "FGA": st.get("fieldGoalsAttempted", 0), "FTA": st.get("freeThrowsAttempted", 0)}
        if "home" not in box or "away" not in box:
            continue
        for side, opp in (("home", "away"), ("away", "home")):
            t, o = box[side], box[opp]
            rows.append({"GAME_ID": e["id"], "GAME_DATE": d.isoformat(), "TEAM_ID": t["TEAM_ID"], "ABBR": t["ABBR"],
                         "PTS": t["PTS"], "OPP_PTS": o["PTS"], "OPP_REB": o["REB"], "OPP_AST": o["AST"],
                         "OPP_FG3M": o["FG3M"],
                         "POSS": (t["FGA"] + 0.44 * t["FTA"] + o["FGA"] + 0.44 * o["FTA"]) / 2})
    return rows


def team_game_rows(season: str, force=False) -> pd.DataFrame:
    """Per team-game rows for a season, incrementally cached."""
    cache = DATA_DIR / f"teamgames_{season}.csv"
    start = SEASON_START.get(season, date(_year(season) - 1, 10, 20))
    end = min(date.today() - timedelta(days=1), date(_year(season), 4, 20))
    have = pd.read_csv(cache) if cache.exists() and not force else pd.DataFrame()
    done = set(pd.to_datetime(have["GAME_DATE"]).dt.date) if not have.empty else set()
    # Re-check the last 2 fetched days in case a late game wasn't final yet.
    days = [d for d in (start + timedelta(i) for i in range((end - start).days + 1))
            if d not in done or (date.today() - d).days <= 2]
    if days:
        with ThreadPoolExecutor(max_workers=6) as ex:
            new = [r for rows in ex.map(_scoreboard_day, days) for r in rows]
        new = pd.DataFrame(new)
        if not have.empty and not new.empty:
            have = have[~have["GAME_ID"].astype(str).isin(new["GAME_ID"].astype(str))]
        have = pd.concat([have, new], ignore_index=True)
        have["GAME_ID"] = have["GAME_ID"].astype(str)
        have = have.drop_duplicates(["GAME_ID", "TEAM_ID"], keep="last")
        have.to_csv(cache, index=False)
    return have


def team_stats(season: str, force=False) -> pd.DataFrame:
    """Shape model.py expects: TEAM_ID, TEAM_NAME, GP, PACE, OPP_PTS, OPP_REB, OPP_AST, OPP_FG3M."""
    tg = team_game_rows(season, force)
    t = teams()
    if tg.empty:
        out = t[["TEAM_ID", "TEAM_NAME", "ABBR"]].copy()
        out["GP"] = 0
        for c in ("PACE", "OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M"):
            out[c] = float("nan")
        return out
    agg = tg.groupby("TEAM_ID").agg(GP=("GAME_ID", "count"), PACE=("POSS", "mean"), OPP_PTS=("OPP_PTS", "mean"),
                                    OPP_REB=("OPP_REB", "mean"), OPP_AST=("OPP_AST", "mean"),
                                    OPP_FG3M=("OPP_FG3M", "mean")).reset_index()
    return t[["TEAM_ID", "TEAM_NAME", "ABBR"]].merge(agg, on="TEAM_ID", how="left").fillna({"GP": 0})


# ------------------------------------------------------------------ grading
def box_scores_for_date(game_date: date) -> pd.DataFrame:
    """Player lines for one date, from each game's summary endpoint."""
    js = _get(f"{SITE}/scoreboard", {"dates": game_date.strftime("%Y%m%d"), "limit": 50})
    rows = []
    for e in (js or {}).get("events", []):
        if e["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        summ = _get(f"{SITE}/summary", {"event": e["id"]})
        for team in (summ or {}).get("boxscore", {}).get("players", []):
            for grp in team.get("statistics", []):
                keys = grp.get("keys", [])
                k = {n: i for i, n in enumerate(keys)}
                if "points" not in k:
                    continue
                for a in grp.get("athletes", []):
                    s = a.get("stats", [])
                    if not s or a.get("didNotPlay"):
                        continue
                    try:
                        rows.append({"PLAYER_ID": int(a["athlete"]["id"]), "MIN": float(s[k["minutes"]] or 0),
                                     "PTS": int(s[k["points"]]), "REB": int(s[k["rebounds"]]),
                                     "AST": int(s[k["assists"]]),
                                     "FG3M": int(s[k["threePointFieldGoalsMade-threePointFieldGoalsAttempted"]].split("-")[0])})
                    except (ValueError, IndexError, KeyError):
                        continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


def load_all(force=False):
    cur = player_game_logs(SEASON, force)
    prior = player_game_logs(PRIOR_SEASON, force)
    return cur, prior, team_stats(SEASON, force), team_stats(PRIOR_SEASON, force)
