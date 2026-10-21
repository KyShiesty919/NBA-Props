"""
Synthetic NBA data for testing the pipeline without network access.
Not for betting. Player names are fake.
"""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import TEAM_ABBR, DATA_DIR
from fetch_lines import norm_name, flatten, LINES_DIR

RNG = np.random.default_rng(7)
TEAMS = sorted(set(TEAM_ABBR.values()))
NAME_TO_ABBR = {k: v for k, v in TEAM_ABBR.items() if k != "LA Clippers"}
ABBR_TO_NAME = {v: k for k, v in NAME_TO_ABBR.items()}
FIRST = ["Jalen", "Marcus", "Devin", "Tyrese", "Zion", "Cade", "Evan", "Paolo", "Scoot", "Amen"]
LAST = ["Carter", "Reed", "Holloway", "Banks", "Okafor", "Nguyen", "Kim", "Suarez", "Patel", "Brooks"]


def _players():
    rows, pid = [], 1000
    for ti, t in enumerate(TEAMS):
        for k in range(9):  # 9-man rotation
            pid += 1
            minutes = [34, 33, 31, 29, 26, 22, 18, 14, 11][k]
            rows.append({"PLAYER_ID": pid, "PLAYER_NAME": f"{FIRST[(ti + k) % 10]} {LAST[(ti * 3 + k) % 10]} {t}",
                         "TEAM_ID": 1610612737 + ti, "TEAM": t, "min_mu": minutes,
                         "pts_rate": RNG.uniform(0.35, 0.95), "reb_rate": RNG.uniform(0.08, 0.35),
                         "ast_rate": RNG.uniform(0.04, 0.28), "fg3_rate": RNG.uniform(0.02, 0.12)})
    return pd.DataFrame(rows)


def _season_logs(players, start: date, n_games: int, season_tag: str):
    rows = []
    for _, p in players.iterrows():
        for g in range(n_games):
            d = start + timedelta(days=int(g * 2.2))
            opp = RNG.choice([t for t in TEAMS if t != p["TEAM"]])
            home = RNG.random() < 0.5
            m = max(0, RNG.normal(p["min_mu"], 4))
            rows.append({"PLAYER_ID": p["PLAYER_ID"], "PLAYER_NAME": p["PLAYER_NAME"], "TEAM_ID": p["TEAM_ID"],
                         "TEAM_ABBREVIATION": p["TEAM"], "GAME_ID": f"{season_tag}{g:04d}", "GAME_DATE": d,
                         "MATCHUP": f"{p['TEAM']} {'vs.' if home else '@'} {opp}", "MIN": round(m, 1),
                         "PTS": RNG.poisson(m * p["pts_rate"]), "REB": RNG.poisson(m * p["reb_rate"]),
                         "AST": RNG.poisson(m * p["ast_rate"]), "FG3M": RNG.poisson(m * p["fg3_rate"])})
    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["HOME"] = ~df["MATCHUP"].str.contains("@")
    df["OPP"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*([A-Z]{3})")[0]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


def _team_stats(gp):
    return pd.DataFrame({"TEAM_ID": [1610612737 + i for i in range(len(TEAMS))],
                         "TEAM_NAME": [ABBR_TO_NAME[t] for t in TEAMS], "GP": gp,
                         "PACE": RNG.normal(99, 2.5, len(TEAMS)).round(1),
                         "OPP_PTS": RNG.normal(113, 4, len(TEAMS)).round(1),
                         "OPP_REB": RNG.normal(44, 2, len(TEAMS)).round(1),
                         "OPP_AST": RNG.normal(26, 2, len(TEAMS)).round(1),
                         "OPP_FG3M": RNG.normal(13, 1.2, len(TEAMS)).round(1)})


def _lines(players, slate, today):
    """Books post lines near the player's true mean with noise; 3 books, -110ish."""
    stat_map = {"PTS": ("player_points", "pts_rate"), "REB": ("player_rebounds", "reb_rate"),
                "AST": ("player_assists", "ast_rate"), "FG3M": ("player_threes", "fg3_rate")}
    tip = datetime.combine(today, datetime.min.time()).replace(hour=23, minute=30, tzinfo=timezone.utc)
    raw = []
    for i, (home, away) in enumerate(slate):
        ev = {"id": f"demo{i}", "home_team": ABBR_TO_NAME[home], "away_team": ABBR_TO_NAME[away],
              "commence_time": tip.isoformat().replace("+00:00", "Z")}
        bks = []
        for bk in ("draftkings", "fanduel", "betmgm"):
            mkts = []
            for s, (mkey, rate) in list(stat_map.items()) + [("PRA", ("player_points_rebounds_assists", None))]:
                outs = []
                for _, p in players[players["TEAM"].isin([home, away])].head(14).iterrows():
                    if p["min_mu"] < 18:
                        continue
                    if s == "PRA":
                        mu = p["min_mu"] * (p["pts_rate"] + p["reb_rate"] + p["ast_rate"])
                    else:
                        mu = p["min_mu"] * p[rate]
                    line = np.floor(mu + RNG.normal(0, 0.8 if s != "FG3M" else 0.3)) + 0.5
                    if line < 0.5:
                        continue
                    lean = int(RNG.normal(0, 8))
                    outs += [{"name": "Over", "description": p["PLAYER_NAME"], "price": -110 - lean, "point": line},
                             {"name": "Under", "description": p["PLAYER_NAME"], "price": -110 + lean, "point": line}]
                mkts.append({"key": mkey, "outcomes": outs})
            bks.append({"key": bk, "markets": mkts})
        raw.append({"event": ev, "odds": {"bookmakers": bks}})
    now = datetime.now(timezone.utc)
    (LINES_DIR / f"{today.isoformat()}_open.json").write_text(json.dumps({"fetched_utc": now.isoformat(), "data": raw}, default=str))
    open_df = flatten(raw, now)
    # fake closing snapshot: shift a few prices so CLV has something to show
    for item in raw:
        for bk in item["odds"]["bookmakers"]:
            for m in bk["markets"]:
                for o in m["outcomes"]:
                    o["price"] += int(RNG.choice([-10, 0, 0, 10]))
    (LINES_DIR / f"{today.isoformat()}_close.json").write_text(json.dumps({"fetched_utc": now.isoformat(), "data": raw}, default=str))
    return open_df


def make_demo(today: date):
    players = _players()
    players.to_csv(DATA_DIR / "demo_players.csv", index=False)
    prior = _season_logs(players, today - timedelta(days=330), 60, "P")
    cur = _season_logs(players, today - timedelta(days=12), 4, "C")
    slate = [("BOS", "NYK"), ("LAL", "DEN"), ("OKC", "DAL"), ("MIA", "ORL"), ("GSW", "PHX"), ("MIL", "CLE")]
    lines = _lines(players, slate, today)
    return cur, prior, _team_stats(4), _team_stats(82), lines


def demo_box_scores(game_date: date) -> pd.DataFrame:
    """Simulate tonight's results from the same true rates."""
    players = pd.read_csv(DATA_DIR / "demo_players.csv")
    rows = []
    for _, p in players.iterrows():
        m = max(0, RNG.normal(p["min_mu"], 4))
        rows.append({"PLAYER_ID": p["PLAYER_ID"], "MIN": m, "PTS": RNG.poisson(m * p["pts_rate"]),
                     "REB": RNG.poisson(m * p["reb_rate"]), "AST": RNG.poisson(m * p["ast_rate"]),
                     "FG3M": RNG.poisson(m * p["fg3_rate"])})
    df = pd.DataFrame(rows)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df
