"""
Usage redistribution + per-position opponent defense.

redistribute(team_rows, out_rows):
    team_rows: baseline projections (minutes, rate_PTS..., pos_group) for available players
    out_rows : same for teammates who are OUT tonight
    1) Minutes: each OUT player's projected minutes are handed to available teammates,
       70% to the same position group, 30% to everyone, proportional to current minutes,
       with a per-player cap (MAX_MINUTES). Overflow cascades to the rest.
    2) Rates: the OUT players' share of the team's projected production in each stat is
       partly absorbed by the remaining players as a per-minute rate bump
       (rate *= 1 + ALPHA[stat] * missing_share * overlap), where overlap is 1.0 for the same
       position group and 0.5 otherwise.
    3) Empirical override: if a player has >= 8 games both with and without a specific OUT
       teammate (prior+current season, same team), the observed ratio of his per-minute rates
       without/with that teammate replaces the generic bump for that pairing (shrunk toward
       the generic value with k = 10 games).

opponent_by_position(cur, prior, rosters):
    From the game logs, what each team allows per game to G / F / C relative to league
    average for that group. Shrunk toward 1 by games sampled.
"""
import numpy as np
import pandas as pd

from config import PRIOR_SHRINK_K

STATS = ["PTS", "REB", "AST", "FG3M"]
# Calibrated on 2025-26 (backtest.py, 900 team-games with a rotation player out): these values
# make teammate projections unbiased; the original 0.6/0.35 with a 100% minutes fill overprojected
# teammates by ~1.8 PTS and was less accurate than no adjustment at all.
ALPHA = {"PTS": 0.3, "AST": 0.3, "FG3M": 0.3, "REB": 0.175}
USE_EMPIRICAL = False   # with/without override: backtest showed no measurable gain (MAE +/- 0.02); off
SAME_POS_MIN_SHARE = 0.7
MAX_MINUTES = 36.0
TEAM_MINUTES = 240.0
ROTATION_SIZE = 10
GARBAGE_MIN = 4.0
FILL_FRACTION = 0.4     # share of the minutes deficit handed to the tracked rotation; the rest goes to
                        # end-of-bench guys who don't have props. Backtest: 1.0 overshot by ~2.5 min/player,
                        # 0.4 is unbiased (+0.1) and beats no-fill on minutes MAE (4.79 vs 4.93).
EMP_MIN_GAMES = 8
EMP_SHRINK_K = 10


# ---------------------------------------------------------------- empirical with/without
def with_without(logs: pd.DataFrame, pid: int, mate_id: int) -> dict | None:
    """Per-minute rate ratio (without mate / with mate) for player pid. None if too thin."""
    if logs is None or logs.empty:
        return None
    me = logs[(logs["PLAYER_ID"] == pid) & (logs["MIN"] > 0)]
    if me.empty:
        return None
    team_games = me[["GAME_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
    mate = logs[(logs["PLAYER_ID"] == mate_id) & (logs["MIN"] > 0)]
    mate_games = set(mate["GAME_ID"]) & set(team_games["GAME_ID"])
    # only games where both were on the same team
    same_team = set(team_games["GAME_ID"]) & set(
        mate[["GAME_ID", "TEAM_ABBREVIATION"]].merge(team_games, on=["GAME_ID", "TEAM_ABBREVIATION"])["GAME_ID"])
    with_g = me[me["GAME_ID"].isin(same_team)]
    # "without" = games my team played that the mate didn't appear in. We only know the team
    # played from my own log, so: my games not in mate's games, restricted to the seasons where
    # he was my teammate at all.
    if mate.empty or not same_team:
        return None
    seasons = set(me[me["GAME_ID"].isin(same_team)]["SEASON"]) if "SEASON" in me else set()
    without_g = me[~me["GAME_ID"].isin(mate_games) & (me["SEASON"].isin(seasons) if seasons else True)]
    if len(with_g) < EMP_MIN_GAMES or len(without_g) < EMP_MIN_GAMES:
        return None
    out = {"n_with": len(with_g), "n_without": len(without_g),
           "min_ratio": without_g["MIN"].mean() / max(with_g["MIN"].mean(), 1)}
    for s in STATS:
        rw = with_g[s].sum() / max(with_g["MIN"].sum(), 1)
        ro = without_g[s].sum() / max(without_g["MIN"].sum(), 1)
        out[f"ratio_{s}"] = (ro / rw) if rw > 0 else 1.0
    return out


# ---------------------------------------------------------------- redistribution
def redistribute(team_rows: pd.DataFrame, out_rows: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Returns team_rows with adjusted minutes and rates, plus an `inj_boost` note column."""
    df = team_rows.copy()
    df["inj_note"] = ""
    df["minutes_base"] = df["minutes"]
    if df.empty:
        return df
    if out_rows.empty:
        out_rows = pd.DataFrame(columns=df.columns)
    # --- team production shares before removal
    tot = {s: (df["minutes"] * df[f"rate_{s}"]).sum() + (out_rows["minutes"] * out_rows[f"rate_{s}"]).sum()
           for s in STATS}
    df["emp_pairs"] = 0
    # ---- 1) minutes: anchor to 240 team minutes --------------------------------------
    # Baseline minutes come from games each guy actually played, so a 15-man roster sums
    # well over 240. Pick a 10-man rotation, give the rest garbage-time minutes, and scale the
    # rotation to fill the 240. Any *deficit* (what the OUT players leave behind) is handed out
    # with a preference for the OUT players' position groups.
    df = df.sort_values("minutes", ascending=False).reset_index(drop=True)
    rot = df.index < ROTATION_SIZE
    df.loc[~rot, "minutes"] = df.loc[~rot, "minutes"].clip(upper=GARBAGE_MIN)   # 11th man and beyond
    target = TEAM_MINUTES - df.loc[~rot, "minutes"].sum()
    cur_sum = df.loc[rot, "minutes"].sum()
    if cur_sum > target:
        # Baseline already covers 240+: the OUT players' minutes are absorbed by the existing
        # rotation and we don't pretend to know whose. (Trimming by rank was tried and wrongly
        # benched real rotation players; the market blend handles over-deep rosters better.)
        pass
    else:                                      # deficit: fill with position preference
        deficit = (target - cur_sum) * FILL_FRACTION
        out_pos_w = out_rows.groupby("pos_group")["minutes"].sum()
        out_pos_w = out_pos_w / out_pos_w.sum() if out_pos_w.sum() > 0 else out_pos_w
        pref = df["pos_group"].map(out_pos_w).fillna(0)          # share of missing minutes at my position
        weight = df["minutes"] * rot * (SAME_POS_MIN_SHARE * pref * 3 + (1 - SAME_POS_MIN_SHARE))
        for _ in range(4):                     # iterate to respect the per-player cap
            room = (MAX_MINUTES - df["minutes"]).clip(lower=0)
            w = weight * (room > 0)
            if w.sum() <= 0 or deficit <= 0.1:
                break
            add = (w / w.sum() * deficit).clip(upper=room)
            df["minutes"] += add
            deficit -= add.sum()
    for _, o in out_rows.iterrows():
        same = df["pos_group"] == o["pos_group"]
        # 2) rates (generic bump) with 3) empirical override per (player, out teammate) pair
        ww = ({r["player_id"]: with_without(logs, r["player_id"], o["player_id"]) for _, r in df.iterrows()}
              if USE_EMPIRICAL else {})
        for s in STATS:
            missing_share = (o["minutes"] * o[f"rate_{s}"]) / tot[s] if tot[s] > 0 else 0
            overlap = np.where(same, 1.0, 0.5)
            generic = pd.Series(1 + ALPHA[s] * missing_share * overlap, index=df.index)
            mult = generic.copy()
            for idx, r in df.iterrows():
                e = ww.get(r["player_id"])
                if e:
                    n = min(e["n_with"], e["n_without"])
                    w = n / (n + EMP_SHRINK_K)
                    mult.loc[idx] = w * float(np.clip(e[f"ratio_{s}"], 0.6, 1.8)) + (1 - w) * generic.loc[idx]
            df[f"rate_{s}"] *= mult
        df["emp_pairs"] = df.get("emp_pairs", 0) + pd.Series({i: int(bool(ww.get(r["player_id"]))) for i, r in df.iterrows()})
        df["inj_note"] += f"{o['player']} out; "
    df["minutes"] = df["minutes"].round(1)
    df["min_boost"] = (df["minutes"] - df["minutes_base"]).round(1)
    return df


# ---------------------------------------------------------------- per-position defense
def opponent_by_position(logs: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    """
    logs: prior + current game logs (roster players only — fine for *relative* comparison).
    pos_map: PLAYER_ID -> 'G'/'F'/'C'
    Returns index (OPP, pos_group) -> REL_<stat> (team allowed / league avg allowed to that group), shrunk.
    """
    d = logs[logs["MIN"] > 0].copy()
    d["pos_group"] = d["PLAYER_ID"].map(pos_map).fillna("F")
    # per opponent-game-group totals, then per-game mean
    g = d.groupby(["OPP", "GAME_ID", "pos_group"])[STATS].sum().reset_index()
    team = g.groupby(["OPP", "pos_group"]).agg(**{f"{s}": (s, "mean") for s in STATS}, n=("GAME_ID", "nunique"))
    league = g.groupby("pos_group")[STATS].mean()
    out = team.copy()
    for s in STATS:
        rel = team[s] / team.index.get_level_values("pos_group").map(league[s]).to_numpy()
        w = team["n"] / (team["n"] + PRIOR_SHRINK_K)
        out[f"REL_{s}"] = 1 + w * (rel - 1)
    return out[[f"REL_{s}" for s in STATS] + ["n"]]
