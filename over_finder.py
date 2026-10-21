"""
Over-finder: can a feature model find profitable OVERS where the projection alone couldn't?

Data: every priced over line from the real-line backtest cache (bt_lines_*_all.pkl), one row
per player/stat/day at the consensus line, plus features computed as-of that date from the
game logs. Fit on 2023-24 + 2024-25, test blind on 2025-26.

Features (all known before tip):
  market : mkt_fair_prob, line, n_books, hold, line dispersion across books, best bettable over price
  model  : model_prob, projection/line, minutes projected, n_cur, new_team, disagree
  minutes security: abs_spread (where we have it), b2b, rest_days, home, starter (season avg min >= 28),
           min_trend (last5 min - season min), usage-model touched (inj)
  form   : form5 (last-5 stat avg / season avg), l10_over_rate (share of last 10 games > this line)
  matchup: opp pace rel, opp allowed rel for this stat, player's past avg vs this opp (rel to season avg), n past games
Target: actual > line.

    python over_finder.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import config
from backtest_angles import spreads_table
from backtest_seasons import team_rel_by_date
from fetch_stats import player_game_logs
from model import OPP_COL

pd.set_option("display.width", 230)
OUT = config.OUT_DIR
FIT, TEST = ["2023-24", "2024-25"], ["2025-26"]
PRIOR = {"2023-24": "2022-23", "2024-25": "2023-24", "2025-26": "2024-25"}
STATS = ["PTS", "REB", "AST", "FG3M", "PRA"]


def load_overs():
    ev = pd.concat([pd.read_pickle(p) for p in OUT.glob("bt_lines_*_all.pkl")], ignore_index=True)
    ev = ev[(ev["side"] == "over") & ev["actual"].notna() & ev["mkt_fair_prob"].notna()].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["home"] = [t == g.split("@")[1].strip() for t, g in zip(ev["team"], ev["game"])]
    ev["bettable"] = ev["book"].isin(config.BETTABLE_BOOKS)
    # line dispersion across books for this player/stat/day
    disp = ev.groupby(["date", "player_key", "stat"])["line"].agg(line_min="min", line_max="max", line_med="median").reset_index()
    ev = ev.merge(disp, on=["date", "player_key", "stat"])
    ev["line_disp"] = ev["line_max"] - ev["line_min"]
    # one row per player/stat/day: the consensus (median) line, best bettable over price
    ev = ev[ev["line"] == ev["line_med"]]
    ev = ev.sort_values("price", ascending=False)
    best = ev[ev["bettable"]].drop_duplicates(["date", "player_key", "stat"])
    sp = spreads_table()
    best = best.merge(sp, on="event_id", how="left")
    return best.reset_index(drop=True)


def player_features(df):
    """As-of-date form / rest / matchup features from the game logs."""
    out = []
    for season, d in df.groupby("season"):
        cur = player_game_logs(season)
        pri = player_game_logs(PRIOR[season])
        logs = pd.concat([pri, cur], ignore_index=True).sort_values("GAME_DATE")
        rel = team_rel_by_date(season, PRIOR[season])
        by_p = {p: g.reset_index(drop=True) for p, g in logs.groupby("PLAYER_ID")}
        rows = []
        for r in d.itertuples():
            g = by_p.get(r.player_id)
            if g is None:
                rows.append({}); continue
            h = g[(g["GAME_DATE"] < r.date) & (g["MIN"] > 0)]
            season_start = cur["GAME_DATE"].min()
            hs = h[h["GAME_DATE"] >= season_start]            # this season only
            base = hs if len(hs) >= 5 else h.tail(30)
            l5, l10 = h.tail(5), h.tail(10)
            s = r.stat
            feat = {
                "rest_days": (r.date - h["GAME_DATE"].iloc[-1]).days if len(h) else np.nan,
                "season_min": base["MIN"].mean() if len(base) else np.nan,
                "min_trend": (l5["MIN"].mean() - base["MIN"].mean()) if len(base) and len(l5) else np.nan,
                "form5": (l5[s].mean() / base[s].mean()) if len(base) and len(l5) and base[s].mean() > 0 else np.nan,
                "l10_over_rate": (l10[s] > r.line).mean() if len(l10) else np.nan,
                "l10_min_sd": l10["MIN"].std() if len(l10) > 2 else np.nan,
            }
            vs = h[h["OPP"] == r.opp]
            feat["vs_opp_n"] = len(vs)
            feat["vs_opp_rel"] = (vs[s].mean() / base[s].mean()) if len(vs) >= 2 and len(base) and base[s].mean() > 0 else np.nan
            rr = rel.get((r.opp, r.date.normalize()), {})
            feat["opp_pace_rel"] = rr.get("PACE_REL", np.nan)
            feat["opp_allow_rel"] = rr.get(OPP_COL.get(s, "OPP_PTS") + "_REL", np.nan) if s != "PRA" else rr.get("OPP_PTS_REL", np.nan)
            rows.append(feat)
        out.append(pd.DataFrame(rows, index=d.index))
        print(f"  features: {season} done ({len(d)} rows)", flush=True)
    return pd.concat(out).sort_index()


FEATURES = ["mkt_fair_prob", "line", "n_books", "hold", "line_disp", "price", "model_prob", "proj_rel", "minutes",
            "n_cur", "new_team", "disagree", "abs_spread", "b2b", "rest_days", "home", "starter", "min_trend",
            "inj", "form5", "l10_over_rate", "l10_min_sd", "vs_opp_n", "vs_opp_rel", "opp_pace_rel", "opp_allow_rel",
            "is_PTS", "is_REB", "is_AST", "is_FG3M", "is_PRA"]


def build():
    df = load_overs()
    print(f"{len(df)} over lines (one per player/stat/day, best bettable price)")
    pf = player_features(df)
    df = pd.concat([df, pf], axis=1)
    df["proj_rel"] = df["projection"] / df["line"].clip(lower=0.5)
    df["starter"] = (df["season_min"] >= 28).astype(float)
    df["inj"] = (df["inj_note"].fillna("").astype(str).str.len() > 0).astype(float)
    for s in STATS:
        df[f"is_{s}"] = (df["stat"] == s).astype(float)
    for c in ("new_team", "b2b", "home"):
        df[c] = df[c].astype(float)
    df["y"] = (df["actual"] > df["line"]).astype(int)
    df = df[df["actual"] != df["line"]]          # drop pushes
    df["dec"] = np.where(df["price"] > 0, df["price"] / 100 + 1, 100 / -df["price"] + 1)
    df.to_pickle(OUT / "over_finder_data.pkl")
    return df


def evaluate(df):
    fit, test = df[df["season"].isin(FIT)], df[df["season"].isin(TEST)]
    X_fit, X_test = fit[FEATURES], test[FEATURES]
    y_fit, y_test = fit["y"], test["y"]
    med = X_fit.median()
    X_fit, X_test = X_fit.fillna(med), X_test.fillna(med)

    models = {
        "market only": None,
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "gbm": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=300,
                                              l2_regularization=1.0, min_samples_leaf=200, random_state=0),
    }
    res = {}
    for name, m in models.items():
        if m is None:
            p_fit, p_test = fit["mkt_fair_prob"].to_numpy(), test["mkt_fair_prob"].to_numpy()
        else:
            m.fit(X_fit, y_fit)
            p_fit, p_test = m.predict_proba(X_fit)[:, 1], m.predict_proba(X_test)[:, 1]
        res[name] = (p_fit, p_test)
        bf = ((p_fit - y_fit) ** 2).mean(); bt = ((p_test - y_test) ** 2).mean()
        print(f"{name:12s} Brier fit {bf:.4f}  test {bt:.4f}   (market: fit {((fit['mkt_fair_prob']-y_fit)**2).mean():.4f} test {((test['mkt_fair_prob']-y_test)**2).mean():.4f})")

    print("\n=== OVER BETTING SIM on blind 2025-26: bet over at best bettable price when p_hat - market >= thr ===")
    rows = []
    for name in ("logistic", "gbm"):
        p_test = res[name][1]
        t = test.assign(p_hat=p_test, edge=p_test - test["mkt_fair_prob"])
        for thr in (0.02, 0.03, 0.04, 0.05, 0.07):
            g = t[t["edge"] >= thr]
            if len(g) < 20:
                continue
            profit = np.where(g["y"] == 1, g["dec"] - 1, -1.0)
            rows.append({"model": name, "thr": thr, "bets": len(g), "hit": g["y"].mean(), "roi": profit.mean(),
                         "z": profit.mean() / (profit.std(ddof=1) / np.sqrt(len(g))), "avg_price": g["price"].mean(),
                         "avg_p_hat": g["p_hat"].mean(), "avg_mkt": g["mkt_fair_prob"].mean()})
        # same thing in-sample for reference
        p_fit = res[name][0]
        t = fit.assign(p_hat=p_fit, edge=p_fit - fit["mkt_fair_prob"])
        for thr in (0.03, 0.05):
            g = t[t["edge"] >= thr]
            profit = np.where(g["y"] == 1, g["dec"] - 1, -1.0)
            rows.append({"model": name + " (FIT, in-sample)", "thr": thr, "bets": len(g), "hit": g["y"].mean(),
                         "roi": profit.mean(), "z": profit.mean() / (profit.std(ddof=1) / np.sqrt(len(g))),
                         "avg_price": g["price"].mean(), "avg_p_hat": g["p_hat"].mean(), "avg_mkt": g["mkt_fair_prob"].mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # what drives overs? univariate hit-rate vs market by feature bucket (all seasons)
    print("\n=== WHERE DO OVERS BEAT THE PRICE? actual hit minus market fair prob, by feature bucket (all 3 seasons) ===")
    df = df.copy()
    df["excess"] = df["y"] - df["mkt_fair_prob"]
    def show(col, bins=None, labels=None):
        if bins is not None:
            b = pd.cut(df[col], bins)
        else:
            b = df[col]
        t = df.groupby(b, observed=True).agg(n=("y", "size"), hit=("y", "mean"), market=("mkt_fair_prob", "mean"),
                                             excess=("excess", "mean"))
        t["roi_if_bet_all"] = [np.where(df.loc[b == k, "y"] == 1, df.loc[b == k, "dec"] - 1, -1.0).mean() for k in t.index]
        print(f"\n{col}:"); print(t.round(3).to_string())
    show("abs_spread", [0, 3, 6, 9, 12, 30])
    show("rest_days", [0, 1, 2, 3, 10])
    show("min_trend", [-30, -4, -1.5, 1.5, 4, 30])
    show("form5", [0, 0.8, 0.95, 1.05, 1.2, 5])
    show("l10_over_rate", [-0.01, 0.3, 0.5, 0.7, 1.0])
    show("opp_pace_rel", [0.8, 0.97, 1.0, 1.03, 1.2])
    show("opp_allow_rel", [0.7, 0.96, 1.0, 1.04, 1.3])
    show("vs_opp_rel", [0, 0.85, 1.0, 1.15, 5])
    show("proj_rel", [0, 0.9, 1.0, 1.05, 1.1, 1.2, 5])
    show("line_disp", [-0.1, 0.0, 0.5, 1.0, 5])
    show("starter")
    show("stat")
    show("price", [-10000, -130, -110, 100, 10000])
    # a hand-built "good over" rule, tested blind
    print("\n=== HAND RULE (chosen on FIT, applied blind to TEST): close game, full-minutes starter, hot, pace, model agrees ===")
    for label, cond in [
        ("spread<=6 & starter & proj_rel>=1.05", lambda t: (t["abs_spread"] <= 6) & (t["starter"] == 1) & (t["proj_rel"] >= 1.05)),
        ("spread<=6 & starter & proj_rel>=1.05 & min_trend>=0", lambda t: (t["abs_spread"] <= 6) & (t["starter"] == 1) & (t["proj_rel"] >= 1.05) & (t["min_trend"] >= 0)),
        ("spread<=6 & proj_rel>=1.05 & opp_allow_rel>=1.02", lambda t: (t["abs_spread"] <= 6) & (t["proj_rel"] >= 1.05) & (t["opp_allow_rel"] >= 1.02)),
        ("proj_rel>=1.08 & min_trend>=1 & rest_days>=2", lambda t: (t["proj_rel"] >= 1.08) & (t["min_trend"] >= 1) & (t["rest_days"] >= 2)),
        ("l10_over_rate>=0.7 & proj_rel>=1.03", lambda t: (t["l10_over_rate"] >= 0.7) & (t["proj_rel"] >= 1.03)),
    ]:
        for setname, part in (("FIT", fit), ("TEST", test)):
            g = part[cond(part)]
            if len(g) == 0:
                continue
            profit = np.where(g["y"] == 1, g["dec"] - 1, -1.0)
            print(f"  {label:55s} {setname:4s} n={len(g):5d} hit {g['y'].mean():.3f} roi {profit.mean():+.4f} "
                  f"z {profit.mean() / (profit.std(ddof=1) / np.sqrt(len(g))):+.2f}")
    return res


if __name__ == "__main__":
    import sys
    if (OUT / "over_finder_data.pkl").exists() and "--rebuild" not in sys.argv:
        df = pd.read_pickle(OUT / "over_finder_data.pkl")
    else:
        df = build()
    evaluate(df)
