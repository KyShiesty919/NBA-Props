"""
Projection + edge engine.

For each player on tonight's slate:
  1. Minutes: exponentially weighted mean of last N games (any season).
  2. Per-minute rate for PTS/REB/AST/FG3M: current season shrunk toward prior
     season by games played (w = n / (n + K)).
  3. Adjust for opponent (what they allow vs league avg), opponent pace,
     home court (points only), and back-to-back.
  4. Distribution: negative binomial for counts (dispersion from the player's
     own game-to-game variance), normal for PRA.
  5. P(over line), compared to the de-vigged market consensus.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from config import (PRIOR_SHRINK_K, TEAM_SHRINK_K, MIN_GAMES_TOTAL, MINUTES_LOOKBACK,
                    MINUTES_DECAY, MINUTES_PRIOR_K, MARKET_WEIGHT_EARLY, MARKET_WEIGHT_K,
                    MAX_DISAGREE, OPP_ADJ_STRENGTH, PACE_ADJ_STRENGTH, HOME_PTS_BUMP,
                    B2B_MINUTES_MULT, MARKET_WEIGHT, MIN_EDGE, MIN_BOOKS, BANKROLL,
                    KELLY_FRACTION, MAX_BET_PCT, MAX_PLAYER_PCT, BETTABLE_BOOKS, MIN_N_CUR,
                    SKIP_USAGE_MODEL_BETS, BET_SIDES, BLOWOUT_SPREAD, BLOWOUT_MIN_N_CUR)
from fetch_lines import norm_name, american_to_prob, prob_to_american

STATS = ["PTS", "REB", "AST", "FG3M"]
OPP_COL = {"PTS": "OPP_PTS", "REB": "OPP_REB", "AST": "OPP_AST", "FG3M": "OPP_FG3M"}


# --------------------------------------------------------------------------- teams
def blend_team_stats(t_cur: pd.DataFrame, t_prior: pd.DataFrame) -> pd.DataFrame:
    """Blend this season's pace/defense with last season's by games played."""
    cols = ["PACE"] + list(OPP_COL.values())
    m = t_cur.merge(t_prior[["TEAM_ID"] + cols], on="TEAM_ID", how="left", suffixes=("", "_p"))
    w = m["GP"].fillna(0) / (m["GP"].fillna(0) + TEAM_SHRINK_K)
    for c in cols:
        prior = m[f"{c}_p"].fillna(m[c])
        m[c] = w * m[c].fillna(prior) + (1 - w) * prior
    out = m[["TEAM_ID", "TEAM_NAME"] + (["ABBR"] if "ABBR" in m.columns else []) + cols].copy()
    for c in cols:
        out[f"{c}_REL"] = out[c] / out[c].mean()   # relative to league average
    return out


def team_abbr_map(cur: pd.DataFrame, prior: pd.DataFrame) -> dict:
    logs = pd.concat([cur, prior])
    latest = logs.sort_values("GAME_DATE").groupby("TEAM_ID")["TEAM_ABBREVIATION"].last()
    return latest.to_dict()


# --------------------------------------------------------------------------- players
def _wmean(x: np.ndarray, decay: float) -> float:
    w = decay ** np.arange(len(x))[::-1]
    return float(np.sum(x * w) / np.sum(w))


def project_player(p_cur: pd.DataFrame, p_prior: pd.DataFrame) -> dict | None:
    """Baseline (pre-adjustment) projection for one player."""
    logs = pd.concat([p_prior, p_cur]).sort_values("GAME_DATE")
    logs = logs[logs["MIN"] > 0]
    if len(logs) < MIN_GAMES_TOTAL:
        return None
    n_cur = int((p_cur["MIN"] > 0).sum())
    recent = logs.tail(MINUTES_LOOKBACK)
    minutes_recent = _wmean(recent["MIN"].to_numpy(), MINUTES_DECAY)
    # Early season: last season's typical minutes (games he actually played) beats the last
    # 10 games of last April, which are full of rest days and tanking rotations.
    pp_played = p_prior[p_prior["MIN"] >= 10]
    if len(pp_played) >= 10:
        wm = n_cur / (n_cur + MINUTES_PRIOR_K)
        minutes = wm * minutes_recent + (1 - wm) * float(pp_played["MIN"].mean())
    else:
        minutes = minutes_recent
    if minutes < 8:
        return None

    w = n_cur / (n_cur + PRIOR_SHRINK_K)
    out = {"minutes": minutes, "n_cur": n_cur, "n_total": len(logs), "w_cur": w}
    pc, pp = p_cur[p_cur["MIN"] > 0], p_prior[p_prior["MIN"] > 0]
    for s in STATS:
        r_cur = pc[s].sum() / pc["MIN"].sum() if pc["MIN"].sum() > 0 else np.nan
        r_pri = pp[s].sum() / pp["MIN"].sum() if pp["MIN"].sum() > 0 else np.nan
        if np.isnan(r_pri):
            rate = r_cur
        elif np.isnan(r_cur):
            rate = r_pri
        else:
            rate = w * r_cur + (1 - w) * r_pri
        out[f"rate_{s}"] = rate
        # Dispersion: raw game-to-game variance over the last ~25 games (this keeps
        # minutes volatility in, which is most of why props bust), scaled linearly
        # if tonight's projected minutes differ from that window's average.
        samp = logs.tail(25)
        scale = minutes / max(samp["MIN"].mean(), 1)
        out[f"var_{s}"] = float(np.var(samp[s], ddof=1) * scale) if len(samp) > 2 else np.nan
    samp = logs.tail(25)
    out["var_PRA"] = float(np.var(samp["PRA"], ddof=1) * (minutes / max(samp["MIN"].mean(), 1))) if len(samp) > 2 else np.nan
    # team changed since last season -> flag (rates from the old role may not transfer)
    last_team = logs["TEAM_ABBREVIATION"].iloc[-1]
    out["new_team"] = bool(len(pp) and pp["TEAM_ABBREVIATION"].iloc[-1] != last_team) if len(pc) else False
    out["last_game"] = logs["GAME_DATE"].max()
    return out


def build_projections(cur, prior, teams, slate, game_date, rosters=None, injuries=None) -> pd.DataFrame:
    """
    slate: DataFrame with columns home, away (abbreviations) for tonight.
    Returns one row per available player on the slate with projected mean/variance per stat,
    after removing OUT players, redistributing their minutes/usage, and applying
    team + position matchup adjustments.
    """
    from injuries import pos_group
    from usage import redistribute, opponent_by_position

    teams = teams.copy()
    if "ABBR" not in teams.columns:
        teams["ABBR"] = teams["TEAM_ID"].map(team_abbr_map(cur, prior))
    tstat = teams.set_index("ABBR")
    id_to_abbr = teams.set_index("TEAM_ID")["ABBR"].to_dict()
    logs = pd.concat([prior, cur], ignore_index=True)
    players_cur, players_pri = cur.groupby("PLAYER_ID"), prior.groupby("PLAYER_ID")
    current_team = {pid: id_to_abbr.get(tid) for pid, tid in logs.groupby("PLAYER_ID")["TEAM_ID"].last().items()}
    names = logs.groupby("PLAYER_ID")["PLAYER_NAME"].last().to_dict()
    pos_map = (rosters.set_index("PLAYER_ID")["POS"].map(pos_group).to_dict() if rosters is not None else {})
    posdef = opponent_by_position(logs, pos_map) if pos_map else None
    inj = injuries if injuries is not None else pd.DataFrame(columns=["player_key", "team", "is_out", "is_dtd", "status"])

    def inj_status(pid, team):
        key = norm_name(names[pid])
        m = inj[(inj["player_key"] == key) & ((inj["team"] == team) | (inj["team"] == ""))]
        if m.empty:
            return "", False, False
        r = m.iloc[0]
        return r["status"], bool(r["is_out"]), bool(r["is_dtd"])

    rows = []
    for _, g in slate.iterrows():
        for team, opp, home in ((g["home"], g["away"], True), (g["away"], g["home"], False)):
            if team not in tstat.index or opp not in tstat.index:
                continue
            team_logs = cur[cur["TEAM_ABBREVIATION"] == team]
            b2b = (not team_logs.empty) and ((pd.Timestamp(game_date) - team_logs["GAME_DATE"].max()).days == 1)
            base_rows = []
            for pid in [p for p, t in current_team.items() if t == team]:
                pc = players_cur.get_group(pid) if pid in players_cur.groups else cur.iloc[0:0]
                pp = players_pri.get_group(pid) if pid in players_pri.groups else prior.iloc[0:0]
                base = project_player(pc, pp)
                if base is None:
                    continue
                status, is_out, is_dtd = inj_status(pid, team)
                base_rows.append({**base, "player_id": pid, "player": names[pid], "pos_group": pos_map.get(pid, "F"),
                                  "inj_status": status, "is_out": is_out, "dtd": is_dtd,
                                  "new_team": bool(len(pp) and pp["TEAM_ABBREVIATION"].iloc[-1] != team)})
            if not base_rows:
                continue
            tdf = pd.DataFrame(base_rows)
            avail = redistribute(tdf[~tdf["is_out"]].reset_index(drop=True), tdf[tdf["is_out"]], logs)
            pace_adj = 1 + PACE_ADJ_STRENGTH * (tstat.loc[opp, "PACE_REL"] - 1)
            for _, r in avail.iterrows():
                minutes = r["minutes"] * (B2B_MINUTES_MULT if b2b else 1.0)
                var_scale = minutes / max(r["minutes_base"], 1)
                row = {"player_id": r["player_id"], "player": r["player"], "player_key": norm_name(r["player"]),
                       "team": team, "opp": opp, "home": home, "b2b": b2b, "pos": r["pos_group"],
                       "minutes": round(minutes, 1), "min_boost": r.get("min_boost", 0.0), "n_cur": r["n_cur"],
                       "new_team": r["new_team"], "dtd": r["dtd"], "inj_status": r["inj_status"],
                       "inj_note": r.get("inj_note", ""), "game": f"{g['away']} @ {g['home']}"}
                mu_sum = 0.0
                for s in STATS:
                    team_rel = tstat.loc[opp, OPP_COL[s] + "_REL"]
                    pos_rel = (posdef.loc[(opp, r["pos_group"]), f"REL_{s}"]
                               if posdef is not None and (opp, r["pos_group"]) in posdef.index else team_rel)
                    opp_adj = 1 + OPP_ADJ_STRENGTH * (0.5 * (team_rel - 1) + 0.5 * (pos_rel - 1))
                    mu = minutes * r[f"rate_{s}"] * opp_adj * pace_adj
                    if s == "PTS" and home:
                        mu *= 1 + HOME_PTS_BUMP
                    row[f"mu_{s}"] = round(mu, 2)
                    row[f"var_{s}"] = r[f"var_{s}"] * var_scale
                    row[f"posdef_{s}"] = round(pos_rel, 3)
                    mu_sum += mu if s != "FG3M" else 0
                row["mu_PRA"] = round(mu_sum, 2)
                row["var_PRA"] = r["var_PRA"] * var_scale
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- probabilities
def prob_over(stat: str, mu: float, var: float, line: float) -> tuple[float, float]:
    """Returns (P(over), P(push)). Half-lines never push."""
    if np.isnan(var) or var <= 0:
        var = mu * 1.3
    if stat == "PRA":
        sd = max(np.sqrt(var), 1.0)
        if line == int(line):
            over = 1 - norm.cdf(line + 0.5, mu, sd)
            push = norm.cdf(line + 0.5, mu, sd) - norm.cdf(line - 0.5, mu, sd)
        else:
            over, push = 1 - norm.cdf(line, mu, sd), 0.0
        return float(over), float(push)
    mu = max(mu, 0.05)
    if var <= mu * 1.02:          # near-Poisson
        r = 1e6
    else:
        r = mu ** 2 / (var - mu)
    r = float(np.clip(r, 0.5, 1e6))
    p = r / (r + mu)
    if line == int(line):
        k = int(line)
        return float(nbinom.sf(k, r, p)), float(nbinom.pmf(k, r, p))
    return float(nbinom.sf(int(np.floor(line)), r, p)), 0.0


def evaluate(proj: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Join projections to de-vigged lines; compute edge, EV, Kelly stake per side."""
    if lines.empty or proj.empty:
        return pd.DataFrame()
    df = lines.merge(proj.drop(columns=["player"]), on="player_key", how="inner")
    df = df[df["game_y"] == df["game_x"]].rename(columns={"game_x": "game"}).drop(columns="game_y")
    out = []
    for _, r in df.iterrows():
        s = r["stat"]
        p_over, p_push = prob_over(s, r[f"mu_{s}"], r[f"var_{s}"], r["line"])
        p_under = 1 - p_over - p_push
        # condition on no push so it's comparable to the de-vigged market number
        model_over = p_over / (p_over + p_under) if (p_over + p_under) > 0 else 0.5
        model_side = model_over if r["side"] == "over" else 1 - model_over
        mkt = r.get("mkt_fair_over")
        mkt_side = (mkt if r["side"] == "over" else 1 - mkt) if pd.notna(mkt) else np.nan
        mw = MARKET_WEIGHT + (MARKET_WEIGHT_EARLY - MARKET_WEIGHT) * MARKET_WEIGHT_K / (MARKET_WEIGHT_K + r["n_cur"])
        if r.get("new_team") and r["n_cur"] < 15:
            mw = max(mw, MARKET_WEIGHT_EARLY)      # last year's role on another team: trust the book more
        blended = (mw * mkt_side + (1 - mw) * model_side if pd.notna(mkt_side) else model_side)
        disagree = abs(r[f"mu_{s}"] - r["line"]) / max(r["line"], 1.0)
        max_dis = MAX_DISAGREE * (0.67 if (r.get("new_team") and r["n_cur"] < 15) else 1.0)
        dec = (r["price"] / 100 + 1) if r["price"] > 0 else (100 / -r["price"] + 1)
        ev = blended * (dec - 1) - (1 - blended)          # per $1 staked
        kelly = max(0.0, (blended * (dec - 1) - (1 - blended)) / (dec - 1))
        stake = min(kelly * KELLY_FRACTION, MAX_BET_PCT) * BANKROLL
        edge = blended - mkt_side if pd.notna(mkt_side) else np.nan
        out.append({**r.to_dict(), "p_push": round(p_push, 3), "market_weight": round(mw, 2),
                    "disagree": round(disagree, 3), "suspicious": bool(disagree > max_dis),
                    "model_prob": round(model_side, 4), "mkt_fair_prob": round(mkt_side, 4) if pd.notna(mkt_side) else np.nan,
                    "blended_prob": round(blended, 4), "edge": round(edge, 4) if pd.notna(edge) else np.nan,
                    "ev_per_dollar": round(ev, 4), "fair_price": prob_to_american(blended),
                    "kelly_full": round(kelly, 4), "stake": round(stake, 2),
                    "book_implied": round(american_to_prob(r["price"]), 4)})
    res = pd.DataFrame(out)
    keep = ["game", "player", "team", "opp", "stat", "side", "line", "book", "price",
            "mu_" + "PTS", "minutes", "n_cur", "new_team", "b2b", "n_books", "hold",
            "book_implied", "mkt_fair_prob", "model_prob", "blended_prob", "edge",
            "ev_per_dollar", "fair_price", "kelly_full", "stake", "p_push", "market_weight",
            "disagree", "suspicious", "dtd", "inj_status", "inj_note", "min_boost", "pos", "abs_spread",
            "event_id", "player_id", "player_key", "fetched_utc", "tip_utc"]
    res["projection"] = res.apply(lambda r: r[f"mu_{r['stat']}"], axis=1)
    keep.insert(9, "projection")
    keep.remove("mu_PTS")
    return res[[c for c in keep if c in res.columns]].sort_values("edge", ascending=False)


def pick_bets(evald: pd.DataFrame) -> pd.DataFrame:
    """Best price per (player, stat, side), filter by edge/books, cap per-player exposure."""
    if evald.empty:
        return evald
    ok = evald[(evald["edge"] >= MIN_EDGE) & (evald["n_books"] >= MIN_BOOKS) & (evald["stake"] > 0)
               & ~evald["suspicious"]]
    if BETTABLE_BOOKS:
        ok = ok[ok["book"].isin(BETTABLE_BOOKS)]
    if BET_SIDES:
        ok = ok[ok["side"].isin(BET_SIDES)]
    no_inj = ok["inj_note"].fillna("").astype(str).str.len() == 0 if "inj_note" in ok.columns else True
    spread = ok["abs_spread"] if "abs_spread" in ok.columns else pd.Series(np.nan, index=ok.index)
    core = (ok["n_cur"] >= MIN_N_CUR) & (no_inj if SKIP_USAGE_MODEL_BETS else True)
    blowout = (spread >= BLOWOUT_SPREAD) & (ok["n_cur"] >= BLOWOUT_MIN_N_CUR)
    ok = ok.assign(tier=np.where(blowout, "blowout", np.where(core, "core", "")))
    ok = ok[ok["tier"] != ""]
    # one bet per player/stat/side: the one with the best EV (handles alt lines across books)
    ok = ok.sort_values("ev_per_dollar", ascending=False).drop_duplicates(["player_key", "stat", "side"])
    # never both sides of the same player/stat
    ok = ok.drop_duplicates(["player_key", "stat"])
    # cap exposure per player
    ok = ok.copy()
    # day-to-day players: half stake
    if "dtd" in ok.columns:
        ok.loc[ok["dtd"].astype(bool), "stake"] = (ok.loc[ok["dtd"].astype(bool), "stake"] / 2).round(2)
    cum = ok.groupby("player_key")["stake"].cumsum()
    ok["stake"] = np.where(cum > MAX_PLAYER_PCT * BANKROLL,
                           np.clip(MAX_PLAYER_PCT * BANKROLL - (cum - ok["stake"]), 0, None), ok["stake"])
    ok["stake"] = ok["stake"].round(2)
    return ok[ok["stake"] > 0].sort_values("edge", ascending=False)
