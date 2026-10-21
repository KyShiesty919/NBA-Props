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
python run_daily.py --close  # run hourly from noon to 10pm: grabs each game once, within 75 min of its tip (needed for CLV)
python grade.py              # next morning: grades yesterday, prints + writes performance_report.xlsx
```

Test the whole thing with no network / no key: `python run_daily.py --demo` then `python grade.py --demo`.

Injuries come from ESPN's injury feed automatically. `data/out.csv` (column `player`)
is still there as a manual override for late scratches the feed hasn't caught.

## What's in the spreadsheet

| Sheet | What |
|---|---|
| **Bets** | What clears the edge bar — one bet per player/stat, best price across books, stake sized at quarter-Kelly and capped at 3% of bankroll |
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
   The absent players' share of team production is partly absorbed by teammates as a per-minute
   rate bump (60% for PTS/AST/3PM, 35% for REB, stronger at the same position). Where a player has
   8+ games both with and without a specific absent teammate, his observed with/without rate ratio
   replaces the generic bump. Day-To-Day players are projected normally, flagged, and bet at half stake.
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

## Known gaps (things that would make it better)

- The injury feed is ESPN's, which lags beat reporters by minutes to hours. Late scratches and
  "questionable" guys who get ruled out at 6:30pm will still slip through the morning run.
- The usage model is a first pass. The generic bump (60% of missing production absorbed) is a
  reasonable league-wide number, not a per-player one; the empirical with/without override only
  kicks in for pairs with enough games together last season. It has not been validated against
  results yet — watch the `inj_note` bets separately in the grading.
- When a roster's baseline minutes already exceed 240 (most healthy teams), nobody is trimmed;
  the market blend is doing that work instead.
- Position groups are ESPN's listed position mapped to G / F / C, which is crude for guys like
  Sabonis (listed F) or point forwards.
- Odds API credits: 5 markets × ~10 games × 2 snapshots ≈ 100/day; needs the 20k/month plan.
