# NBA Prop Model

Projects PTS / REB / AST / 3PM / PRA for tonight's slate, de-vigs the sportsbook
lines, and only flags a bet when the model + market blend beats the market
consensus by a set edge. Everything gets logged so you can grade yourself
honestly (hit rate, ROI, calibration, closing-line value).

## Setup (one time)

```
pip install pandas numpy scipy requests openpyxl
export ODDS_API_KEY=your_key_here      # the-odds-api.com, same key as DiamondEdge used
```

Edit `config.py` → `SEASON` / `PRIOR_SEASON` if needed, and `BANKROLL`.

## Daily routine

```
python run_daily.py          # morning: pulls data, projects, writes outputs/YYYY-MM-DD_props.xlsx
python run_daily.py --rescore # 4:30pm & 5:30pm: re-pull injuries, re-score vs morning lines, print what changed (free)
python run_daily.py --close  # run hourly from noon to 10pm: grabs each game once, within 75 min of its tip (needed for CLV)
python grade.py              # next morning: grades yesterday, prints + writes performance_report.xlsx
```

Test the whole thing with no network / no key: `python run_daily.py --demo` then `python grade.py --demo`.

Injuries come from ESPN's injury feed automatically. `data/out.csv` (column `player`)
is still there as a manual override for late scratches the feed hasn't caught.

## What's in the spreadsheet

| Sheet | What |
|---|---|
| **Bets** | What clears the bar — unders only, `tier` = blowout (10+ pt spread) or core; one bet per player/stat, best price across books, quarter-Kelly stake capped at 3% of bankroll |
| **All lines** | Every book/line/side the model priced, with model prob, market fair prob, blended prob, edge, EV |
| **Projections** | Raw means per player (minutes, PTS, REB, AST, 3PM, PRA) |
| **Raw lines** | Exactly what the Odds API returned |
| **Injuries** | ESPN injury report at run time (Out / Day-To-Day, return dates) |
| **Settings** | Knobs that produced this file |

## How the model works

1. **Minutes** — exponentially weighted mean of the last 10 games (any season), 3% haircut on a back-to-back.
2. **Per-minute rates** — this season's rate shrunk toward last season's: `w = games / (games + 15)`. Early in the season it's mostly last year; by ~30 games it's mostly this year. Players who changed teams get flagged `new_team`.
3. **Injuries & usage** (`injuries.py`, `usage.py`) — players listed Out are removed. Each team's
   rotation is anchored to 240 minutes: if the available players' baseline minutes fall short, the
   deficit is handed out with a preference for the absent players' position group (capped at 36/player).
   Only 40% of that deficit goes to the tracked rotation (the rest goes to end-of-bench guys with
   no props). The absent players' share of team production is partly absorbed by teammates as a
   per-minute rate bump (30% for PTS/AST/3PM, 17.5% for REB, stronger at the same position).
   These numbers are calibrated, not guessed: `python backtest.py` replays last season's 900+
   games with a rotation player out. Day-To-Day players are projected normally, flagged, and bet
   at half stake.
4. **Matchup** — opponent adjustment is half team-level (what they allow overall) and half
   position-level (what they allow to G / F / C), each vs league average and shrunk by games;
   applied at half strength. Opponent pace at half strength. +1.5% points at home. 3% minutes
   haircut on a back-to-back.
5. **Distribution** — negative binomial for counts, using the player's own game-to-game variance (keeps minutes volatility in, which is most of why props bust). Normal for PRA. Integer lines handle pushes.
6. **Market** — multiplicative de-vig per book, consensus = median fair prob across books at the same line. Needs 2+ books.
7. **Blend** — `final = w × market + (1 − w) × model`, where w starts at 0.70 for a player with no
   games this season and decays to 0.35 by ~15 games (players on a new team stay at 0.70 until 15
   games). Edge = final − market. Bet when edge ≥ 4%, and never when the projection is more than
   30% away from the line (20% for new-team players) — that gap means the book knows something.

## Knobs that matter (config.py)

- `MARKET_WEIGHT` (0.35) — how much to trust the book over the model. **Do not lower this until `grade.py` shows positive average CLV over 200+ bets.** A raw model that ignores the market will find "edges" everywhere and most are noise.
- `MIN_EDGE` (0.04), `MIN_BOOKS` (2), `KELLY_FRACTION` (0.25), `MAX_BET_PCT` (0.03).
- `OPP_ADJ_STRENGTH` / `PACE_ADJ_STRENGTH` (0.5) — opponent effects are real but noisy; full strength overfits.

## Reading grade.py honestly

- **CLV** is the number that matters. If `avg_clv` is positive and `clv_pos_rate` > 50% over a few hundred bets, the edge is real even during a losing stretch. If CLV is ~0, the model isn't beating the book regardless of record.
- **Calibration** — the `gap` column should hover near 0 in every bucket. A big negative gap in the high-confidence bucket means the model is overconfident → raise `MARKET_WEIGHT` or `MIN_EDGE`.
- **By prop** — expect some prop types to be duds. Turn off the ones that don't calibrate by removing them from `MARKETS` rather than hoping they come around.
- Under ~200 graded bets, ROI is noise. The z-score line will tell you.

## What the backtest says (read this before trusting anything)

`backtest_lines.py` replayed the production model against **real sportsbook prop lines** (The Odds
API historical snapshots, ~2h before tip) on 35 game-days per season for 2023-24, 2024-25 and
2025-26: 616k priced lines, 10,768 graded bets. `backtest_tune.py` re-simulates betting rules on
that cache (fit on 2023-25, blind test on 2025-26).

| Configuration | Fit 2023-25 | Blind 2025-26 |
|---|---|---|
| Original (market weight 0.35, edge 4%) | −2.1% ROI, z = −2.0 | −1.8% |
| Market weight 0.75, edge 3% | −0.3% | −1.3% |
| + no usage-model bets, 10+ games played **(current)** | +5.6% (930 bets) | +1.9% (247 bets, z = 0.3) |
| Rebounds + threes only | +4.1% | 0.0% |

Second round (`backtest_angles.py`, spreads + morning + alt-line snapshots, blind-tested on 2025-26):

| Angle | Fit 2023-25 | Blind 2025-26 | Verdict |
|---|---|---|---|
| **Unders in 10+ pt spread games** (mw .75, edge 3%, 5+ games) | +14.9% (199) | **+20.9% (190), z = 2.9** | Live. Starters sit in blowouts. |
| Unders only, core settings | +8.8% (494) | +8.3% (150) | Live. Overs lost in every cut. |
| Morning lines: CLV of model leans | — | +0.6 pts, 56–59% move toward model | Bet in the morning. |
| Back-to-back unders | +5.4% | −3.4% | Didn't hold. |
| Alt points ladders | — | −40% | Longshot bias. Not offered. |
| Double-doubles | — | −26% | Same. |

So production bets **unders only**, in two tiers: `core` (10+ games, no usage-model involvement)
and `blowout` (10+ point spread, 5+ games). Expect ~250 core and ~200 blowout bets a season.

Takeaways:
- The raw model is **less** accurate than the market (Brier 0.248 vs 0.244). Blended with the
  market it is slightly **more** accurate (0.241), so it carries information — just not much.
- The original settings were overconfident by 5–11 points in every probability bucket. Actual hit
  rates landed ~1 point above the market's fair price: a real edge, smaller than the vig.
- The first 5 games of a season ran −28%. Points and PRA lost; rebounds and threes won in-sample
  but not out-of-sample. Usage-model bets lost (−3.2%) vs +0.7% without.
- **Nothing tested is profitable with statistical confidence.** Current settings are best described
  as "probably break-even, maybe slightly positive, ~250–450 bets a season".
- The backtest is against near-closing lines. Morning lines are softer; live CLV tracking in
  `grade.py` is the only thing that can show whether that helps.

## Known gaps (things that would make it better)

- The injury feed is ESPN's, which lags beat reporters by minutes to hours. Late scratches and
  "questionable" guys who get ruled out at 6:30pm will still slip through the morning run.
- The usage model is calibrated on last season's absences (`backtest.py`): unbiased on average,
  and better than no adjustment on minutes. But it is a league-wide rule, and the players it boosts
  most still come out ~0.9 PTS high on average — so treat big `min_boost` overs with some
  suspicion. A per-player with/without override was tested and did nothing (MAE ±0.02), so it's off.
- Minutes: EWMA of the last 10 games played was the best rule tested (MAE 4.6 min); a
  per-team-game average (which would justify trimming deep rosters) was 4 minutes biased low.
  So when a roster's baseline exceeds 240, nobody is trimmed; the market blend handles it.
- Position groups are ESPN's listed position mapped to G / F / C, which is crude for guys like
  Sabonis (listed F) or point forwards.
- Odds API credits: 5 markets × ~10 games × 2 snapshots ≈ 100/day; needs the 20k/month plan.
