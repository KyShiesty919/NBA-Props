"""
REAL-LINE backtest: run the production model walk-forward against historical sportsbook
player-prop lines from The Odds API, and grade the bets it would have placed.

  * Sampled game-days per season (evenly spaced Nov -> early Apr).
  * For each game: one odds snapshot 2 hours before tip (near-closing lines: the hard test).
    Cost: 50 credits per game (5 markets x 10). Snapshots are cached in data/hist_lines/
    so re-running the model costs nothing.
  * Projections use ONLY logs before that date, season-to-date team stats shrunk toward the
    prior season, and the production model code (build_projections / evaluate / pick_bets),
    with the current config knobs (market weight by games played, suspicious filter,
    bettable books, Kelly sizing).
  * "Out" players are inferred the way a bettor could at snapshot time: a rotation player
    (15+ min in 3 of the team's last 5) with no prop posted. That feeds the usage model.
  * Grading: actual box score from the ESPN logs. No log that day = void.

    python backtest_lines.py fetch 2024-25 35     # pull lines for 35 days (costs credits)
    python backtest_lines.py run 2024-25          # project + grade from cache (free)
    python backtest_lines.py report               # all cached seasons, summary + xlsx
"""
import gzip
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
from fetch_lines import flatten, devig, norm_name, american_to_prob
from fetch_stats import player_game_logs, team_game_rows, teams, rosters, SEASON_START
from injuries import pos_group
from model import blend_team_stats, build_projections, evaluate, pick_bets

API = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
HIST = config.DATA_DIR / "hist_lines"
HIST.mkdir(exist_ok=True)
PRIOR = {"2023-24": "2022-23", "2024-25": "2023-24", "2025-26": "2024-25"}
SEASON_END = {"2022-23": date(2023, 4, 9), "2023-24": date(2024, 4, 14), "2024-25": date(2025, 4, 13), "2025-26": date(2026, 4, 12)}
ET = -5  # hours; close enough for choosing dates (DST ignored on purpose)


# ------------------------------------------------------------------ fetching
def sample_days(season: str, n: int) -> list[date]:
    start = SEASON_START[season] + timedelta(days=14)
    end = SEASON_END[season] - timedelta(days=5)
    span = (end - start).days
    return sorted({start + timedelta(days=int(round(i * span / (n - 1)))) for i in range(n)})


def _get(url, params):
    for i in range(3):
        r = requests.get(url, params={**params, "apiKey": config.ODDS_API_KEY}, timeout=30)
        if r.status_code == 200:
            return r.json(), r.headers.get("x-requests-remaining")
        if r.status_code == 429:
            time.sleep(5)
            continue
        if r.status_code in (404, 422):
            return None, r.headers.get("x-requests-remaining")
        time.sleep(2)
    return None, None


SNAPS = {
    # tag: (markets, how to pick the snapshot time)
    "close":   (list(config.MARKETS), "tip-2h"),
    "morning": (list(config.MARKETS), "10am"),
    "extras":  (["player_points_alternate", "player_double_double", "spreads"], "tip-2h"),   # 30 credits/game
    "spreads": (["spreads"], "tip-2h"),                                                      # 10 credits/game
}


def _snap_time(tip: datetime, rule: str) -> datetime:
    if rule == "10am":
        local = tip + timedelta(hours=ET)
        return datetime(local.year, local.month, local.day, 15, 0, tzinfo=timezone.utc)   # ~10-11am ET
    return (tip - timedelta(hours=2)).replace(second=0, microsecond=0)


def fetch_day(season: str, d: date, tag: str = "close") -> int:
    """Fetch and cache all games on ET date d. Returns credits spent (approx)."""
    out = HIST / season / (f"{d.isoformat()}.json.gz" if tag == "close" else f"{d.isoformat()}_{tag}.json.gz")
    if out.exists():
        return 0
    markets, rule = SNAPS[tag]
    out.parent.mkdir(parents=True, exist_ok=True)
    # events list as of noon ET that day
    at = datetime(d.year, d.month, d.day, 17, 0, tzinfo=timezone.utc)
    js, rem = _get(f"{API}/events", {"date": at.isoformat().replace("+00:00", "Z")})
    if not js:
        out.write_bytes(gzip.compress(json.dumps({"date": d.isoformat(), "data": []}).encode()))
        return 1
    raw, spent = [], 1
    for ev in js.get("data", []):
        tip = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if (tip + timedelta(hours=ET)).date() != d:
            continue
        snap_at = _snap_time(tip, rule)
        odds, rem = _get(f"{API}/events/{ev['id']}/odds", {
            "regions": "us", "markets": ",".join(markets), "oddsFormat": "american",
            "date": snap_at.isoformat().replace("+00:00", "Z")})
        spent += 10 * len(markets)
        if odds and odds.get("data"):
            raw.append({"event": ev, "odds": odds["data"], "snapshot": odds.get("timestamp")})
        time.sleep(0.2)
    out.write_bytes(gzip.compress(json.dumps({"date": d.isoformat(), "data": raw}).encode()))
    print(f"  {d} [{tag}]: {len(raw)} games cached, ~{spent} credits, {rem} remaining", flush=True)
    return spent


def load_day(season: str, d: date, tag: str = "close"):
    p = HIST / season / (f"{d.isoformat()}.json.gz" if tag == "close" else f"{d.isoformat()}_{tag}.json.gz")
    if not p.exists():
        return None
    js = json.loads(gzip.decompress(p.read_bytes()).decode())
    return js["data"]


# ------------------------------------------------------------------ modelling as-of a date
def team_stats_asof(season: str, before: pd.Timestamp) -> pd.DataFrame:
    tg = team_game_rows(season)
    tg = tg[pd.to_datetime(tg["GAME_DATE"]) < before]
    t = teams()
    if tg.empty:
        out = t[["TEAM_ID", "TEAM_NAME", "ABBR"]].copy()
        out["GP"] = 0
        for c in ("PACE", "OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M"):
            out[c] = np.nan
        return out
    agg = tg.groupby("TEAM_ID").agg(GP=("GAME_ID", "count"), PACE=("POSS", "mean"), OPP_PTS=("OPP_PTS", "mean"),
                                    OPP_REB=("OPP_REB", "mean"), OPP_AST=("OPP_AST", "mean"),
                                    OPP_FG3M=("OPP_FG3M", "mean")).reset_index()
    return t[["TEAM_ID", "TEAM_NAME", "ABBR"]].merge(agg, on="TEAM_ID", how="left").fillna({"GP": 0})


def _with_game_team_ids(logs: pd.DataFrame, abbr_to_id: dict) -> pd.DataFrame:
    """Historical mode: TEAM_ID must be the team he played for in that game, not today's roster."""
    logs = logs.copy()
    logs["TEAM_ID"] = logs["TEAM_ABBREVIATION"].map(abbr_to_id)
    return logs


def run_day(season, d, cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map):
    raw = load_day(season, d)
    if not raw:
        return None, None
    before = pd.Timestamp(d)
    lines = flatten(raw, datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc))
    if lines.empty:
        return None, None
    lines = devig(lines)
    slate = lines[["home", "away"]].drop_duplicates()
    cur = cur_all[cur_all["GAME_DATE"] < before]
    teams_bl = blend_team_stats(team_stats_asof(season, before), t_prior)
    # infer OUT: rotation players (15+ min in 3 of team's last 5 games) with no prop posted
    posted = set(lines["player_key"])
    inj_rows = []
    for team in set(slate["home"]) | set(slate["away"]):
        tl = cur[cur["TEAM_ABBREVIATION"] == team]
        last5 = sorted(tl["GAME_ID"].unique(), key=lambda g: tl[tl["GAME_ID"] == g]["GAME_DATE"].iloc[0])[-5:]
        rec = tl[tl["GAME_ID"].isin(last5) & (tl["MIN"] >= 15)].groupby(["PLAYER_ID", "PLAYER_NAME"]).size()
        for (pid, name), n in rec.items():
            if n >= 3 and norm_name(name) not in posted:
                inj_rows.append({"player": name, "player_key": norm_name(name), "team": team, "pos": "",
                                 "status": "Out (no props posted)", "is_out": True, "is_dtd": False})
    inj = pd.DataFrame(inj_rows) if inj_rows else pd.DataFrame(columns=["player_key", "team", "is_out", "is_dtd", "status"])
    proj = build_projections(cur, prior_all, teams_bl, slate, d, ros, inj)
    if proj.empty:
        return None, None
    evald = evaluate(proj, lines)
    bets = pick_bets(evald)
    # grade
    act = cur_all[cur_all["GAME_DATE"] == before].drop_duplicates(["PLAYER_ID", "OPP"]).set_index(["PLAYER_ID", "OPP"])
    for df in (evald, bets):
        if df.empty:
            continue
        df["date"] = d.isoformat()
        df["season"] = season
        df["actual"] = [float(act.loc[(p, o), s]) if (p, o) in act.index else np.nan
                        for p, o, s in zip(df["player_id"], df["opp"], df["stat"])]
        won = np.where(df["side"] == "over", df["actual"] > df["line"], df["actual"] < df["line"])
        push = df["actual"] == df["line"]
        df["result"] = np.where(df["actual"].isna(), "void", np.where(push, "push", np.where(won, "win", "loss")))
        dec = np.where(df["price"] > 0, df["price"] / 100 + 1, 100 / -df["price"] + 1)
        df["profit_flat"] = np.where(df["result"] == "win", dec - 1, np.where(df["result"] == "loss", -1.0, 0.0))
        df["profit_staked"] = df["profit_flat"] * df["stake"]
    return evald, bets


def run_season(season: str):
    prior_season = PRIOR[season]
    t = teams()
    abbr_to_id = t.set_index("ABBR")["TEAM_ID"].to_dict()
    cur_all = _with_game_team_ids(player_game_logs(season), abbr_to_id)
    prior_all = _with_game_team_ids(player_game_logs(prior_season), abbr_to_id)
    # prior-season TEAM_ID should be the team he ENDED that season with, for roster continuity
    t_prior = team_stats_asof(prior_season, pd.Timestamp(SEASON_END[prior_season] + timedelta(days=30)))
    ros = rosters()
    pos_map = ros.set_index("PLAYER_ID")["POS"].map(pos_group).to_dict()
    days = sorted(date.fromisoformat(p.name[:10]) for p in (HIST / season).glob("??????????.json.gz")) if (HIST / season).exists() else []
    ev_all, bet_all = [], []
    for d in days:
        t0 = time.time()
        e, b = run_day(season, d, cur_all, prior_all, t_prior, ros, abbr_to_id, pos_map)
        if e is None:
            continue
        ev_all.append(e)
        bet_all.append(b)
        g = b[b["result"].isin(["win", "loss"])] if not b.empty else b
        print(f"  {d}: {len(e)} lines, {len(b)} bets, {int((g['result']=='win').sum()) if len(g) else 0}-"
              f"{int((g['result']=='loss').sum()) if len(g) else 0}, {time.time()-t0:.0f}s", flush=True)
    ev = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
    bets = pd.concat(bet_all, ignore_index=True) if bet_all else pd.DataFrame()
    ev.to_pickle(config.OUT_DIR / f"bt_lines_{season}_all.pkl")
    bets.to_pickle(config.OUT_DIR / f"bt_lines_{season}_bets.pkl")
    return ev, bets


# ------------------------------------------------------------------ report
def summarize(b: pd.DataFrame) -> pd.Series:
    g = b[b["result"].isin(["win", "loss"])]
    n, w = len(g), int((g["result"] == "win").sum())
    return pd.Series({"bets": n, "wins": w, "hit_rate": w / n if n else np.nan,
                      "roi_flat": g["profit_flat"].sum() / n if n else np.nan,
                      "profit_staked": g["profit_staked"].sum(), "staked": g["stake"].sum(),
                      "roi_staked": g["profit_staked"].sum() / g["stake"].sum() if g["stake"].sum() else np.nan,
                      "avg_edge": g["edge"].mean(), "avg_blended_prob": g["blended_prob"].mean(),
                      "avg_mkt_prob": g["mkt_fair_prob"].mean()})


def report():
    pd.set_option("display.width", 220)
    bets = pd.concat([pd.read_pickle(p) for p in config.OUT_DIR.glob("bt_lines_*_bets.pkl")], ignore_index=True)
    ev = pd.concat([pd.read_pickle(p) for p in config.OUT_DIR.glob("bt_lines_*_all.pkl")], ignore_index=True)
    bets["inj"] = bets["inj_note"].fillna("").astype(str).str.len() > 0
    bets["boosted"] = bets["min_boost"].fillna(0) > 1
    g = bets[bets["result"].isin(["win", "loss"])]
    print(f"\n{len(ev)} priced lines evaluated, {len(bets)} bets placed, {len(g)} graded "
          f"({(bets['result']=='void').sum()} void, {(bets['result']=='push').sum()} push)\n")
    print("=== OVERALL ===");        print(summarize(bets).round(4).to_string())
    print("\n=== BY SEASON ===");    print(bets.groupby("season").apply(summarize).round(4).to_string())
    print("\n=== BY PROP ===");      print(bets.groupby("stat").apply(summarize).round(4).to_string())
    print("\n=== BY SIDE ===");      print(bets.groupby("side").apply(summarize).round(4).to_string())
    print("\n=== BY BOOK ===");      print(bets.groupby("book").apply(summarize).round(4).to_string())
    print("\n=== INJURY / USAGE-MODEL BETS vs OTHERS ===")
    print(bets.groupby("inj").apply(summarize).round(4).to_string())
    print("\n=== BY EDGE BUCKET (does more edge = more profit? it should) ===")
    bets["edge_bucket"] = pd.cut(bets["edge"], [0.04, 0.05, 0.06, 0.08, 0.10, 1.0])
    print(bets.groupby("edge_bucket", observed=True).apply(summarize).round(4).to_string())
    print("\n=== BY GAMES PLAYED THIS SEASON (early-season reliance on last year) ===")
    bets["n_bucket"] = pd.cut(bets["n_cur"], [-1, 5, 15, 30, 50, 100])
    print(bets.groupby("n_bucket", observed=True).apply(summarize).round(4).to_string())
    print("\n=== CALIBRATION of blended prob on graded bets ===")
    g = g.copy()
    g["win"] = (g["result"] == "win").astype(int)
    g["bucket"] = pd.cut(g["blended_prob"], [0, 0.5, 0.54, 0.58, 0.62, 0.66, 1.0])
    c = g.groupby("bucket", observed=True).agg(n=("win", "size"), predicted=("blended_prob", "mean"),
                                               market=("mkt_fair_prob", "mean"), actual=("win", "mean"))
    c["gap_vs_model"] = c["actual"] - c["predicted"]
    c["gap_vs_market"] = c["actual"] - c["market"]
    print(c.round(3).to_string())
    # raw-model calibration on ALL priced lines (not just bets): is model_prob informative at all?
    print("\n=== RAW MODEL vs MARKET on ALL priced lines (over side only, one row per book-line) ===")
    o = ev[(ev["side"] == "over") & ev["actual"].notna()].copy()
    o["over"] = (o["actual"] > o["line"]).astype(int)
    for col in ("model_prob", "mkt_fair_prob", "blended_prob"):
        brier = ((o[col] - o["over"]) ** 2).mean()
        print(f"  Brier {col:14s}: {brier:.4f}   (lower is better; 0.25 = coin flip)")
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean()
    print(f"  LogLoss model {ll(o['model_prob'], o['over']):.4f}  market {ll(o['mkt_fair_prob'], o['over']):.4f}  "
          f"blended {ll(o['blended_prob'], o['over']):.4f}")
    # statistical significance of flat ROI
    pr = g["profit_flat"]
    z = pr.mean() / (pr.std(ddof=1) / np.sqrt(len(pr))) if len(pr) > 1 else np.nan
    print(f"\nz-score of flat ROI vs 0: {z:.2f}  (|z| > 2 ≈ 95% confidence it isn't luck)")
    with pd.ExcelWriter(config.OUT_DIR / "backtest_real_lines.xlsx", engine="openpyxl") as xw:
        summarize(bets).to_frame("ALL").T.to_excel(xw, sheet_name="Overall")
        bets.groupby("season").apply(summarize).to_excel(xw, sheet_name="By season")
        bets.groupby("stat").apply(summarize).to_excel(xw, sheet_name="By prop")
        bets.groupby("side").apply(summarize).to_excel(xw, sheet_name="By side")
        bets.groupby("book").apply(summarize).to_excel(xw, sheet_name="By book")
        bets.groupby("edge_bucket", observed=True).apply(summarize).to_excel(xw, sheet_name="By edge")
        bets.groupby("n_bucket", observed=True).apply(summarize).to_excel(xw, sheet_name="By games played")
        c.to_excel(xw, sheet_name="Calibration")
        cols = ["season", "date", "game", "player", "team", "opp", "stat", "side", "line", "book", "price", "projection",
                "model_prob", "mkt_fair_prob", "blended_prob", "edge", "stake", "n_cur", "inj_note", "min_boost",
                "actual", "result", "profit_flat", "profit_staked"]
        bets[[c for c in cols if c in bets.columns]].to_excel(xw, sheet_name="All bets", index=False)
    print(f"Saved {config.OUT_DIR / 'backtest_real_lines.xlsx'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        season, n = sys.argv[2], int(sys.argv[3])
        tag = sys.argv[4] if len(sys.argv) > 4 else "close"
        every = int(sys.argv[5]) if len(sys.argv) > 5 else 1      # take every k-th sampled day
        total = 0
        for d in sample_days(season, n)[::every]:
            total += fetch_day(season, d, tag)
        print(f"{season}: ~{total} credits spent this run")
    elif cmd == "run":
        run_season(sys.argv[2])
    elif cmd == "report":
        report()
