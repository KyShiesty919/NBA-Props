"""
NBA Prop Model — configuration.
Edit these; nothing else in the package needs touching for day-to-day use.
"""
import os
from pathlib import Path

# load .env if present
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip())

# Pin the process clock to US/Eastern so "today" and "HHMM" match the NBA
# calendar day wherever this runs (the hourly cloud task runs in UTC, where
# 8pm-ET tips are already "tomorrow" and would be skipped by --close).
os.environ["TZ"] = "America/New_York"
try:
    import time as _time
    _time.tzset()
except AttributeError:   # Windows has no tzset; local clock is used as-is
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
LEDGER = DATA_DIR / "picks_ledger.csv"

# ---- Seasons --------------------------------------------------------------
SEASON = "2026-27"          # current season, stats.nba.com format
PRIOR_SEASON = "2025-26"    # used to shrink early-season noise

# ---- Lines source ---------------------------------------------------------
# https://the-odds-api.com  (export ODDS_API_KEY=... or put it in .env)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_REGIONS = "us"
# Odds API market key -> internal stat key
MARKETS = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
    "player_points_rebounds_assists": "PRA",
}
# Books to ignore entirely (DFS pick'em sites etc.)
EXCLUDE_BOOKS = {"prizepicks", "underdog", "betr_picks", "sleeper"}
# Books you can actually bet at. All books still feed the market consensus; only these
# are eligible for the Bets sheet. None = any book. NC-licensed books by default.
BETTABLE_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "bet365", "betrivers"}

# ---- Model knobs ----------------------------------------------------------
PRIOR_SHRINK_K = 15       # games: w_current = n / (n + K) when blending seasons
TEAM_SHRINK_K = 10        # same idea for team pace / defense
MIN_GAMES_TOTAL = 8       # skip players with fewer logged games across both seasons
MINUTES_LOOKBACK = 10     # games used for minutes projection
MINUTES_DECAY = 0.85      # exponential weight per game back
MINUTES_PRIOR_K = 5       # early season: blend recent minutes with last season's full-season avg
                          # w_recent = n_cur / (n_cur + K); last-10-of-last-April is garbage (rest/tank)
OPP_ADJ_STRENGTH = 0.5    # 0 = ignore opponent, 1 = full opponent effect
PACE_ADJ_STRENGTH = 0.5
HOME_PTS_BUMP = 0.015     # +1.5% points at home (small, well-documented)
B2B_MINUTES_MULT = 0.97   # minutes haircut on 2nd night of back-to-back

# ---- Betting knobs ----------------------------------------------------------
# Final prob = MARKET_WEIGHT * market_fair + (1 - MARKET_WEIGHT) * model.
# The market is sharp; a raw model that ignores it will overbet. Start here,
# and move toward 0 ONLY once grading shows the raw model beats the closing line.
# Backtest (backtest_lines.py / backtest_tune.py, 10,768 real-line bets over 2023-26, lines ~2h pre-tip):
#   mw 0.35 / edge 0.04 (original):            -2.1% ROI, z = -2.1  (a loser, with confidence)
#   mw 0.75 / edge 0.03:                        -0.3% fit, -1.3% blind test
#   + no usage-model bets + 10 games played:    +5.6% fit, +1.9% blind test (247 bets, not significant)
# Nothing tested is profitable with statistical confidence. These settings are "probably ~break-even,
# maybe slightly positive". Live CLV tracking is the only thing that can upgrade that verdict.
MARKET_WEIGHT = 0.75      # floor, reached once a player has ~15 games this season
MARKET_WEIGHT_EARLY = 0.85  # used when a player has 0 games this season; decays toward MARKET_WEIGHT
MARKET_WEIGHT_K = 6         # weight = floor + (early - floor) * K / (K + n_cur)
# Second round of backtesting (backtest_angles.py, blind-tested on 2025-26):
#   * OVERS lost in every cut (-3.5% overall). Unders only at these settings: +8.8% fit / +8.3% blind.
#   * UNDERS IN GAMES WITH A 10+ POINT SPREAD: 389 bets, 58.6% hit, +17.8% ROI, z = 3.5, held on both
#     samples and monotonic in spread. Starters sit in blowouts; props don't fully price it.
#   * Morning lines closed toward the model's side 56-59% of the time (+0.6 pts avg CLV): bet mornings.
#   * Alt lines and double-doubles: -25% to -40% (longshot bias). Not offered.
BET_SIDES = {"under", "over"}   # overs only via the experimental over tier below
# OVER tier (over_finder.py): 33k over lines; overs hit ~1 pt BELOW the market's fair price in nearly every
# cut, so overs must clear the vig and that shade. The only rule that held blind: close game, model
# >=5% above the line, opponent allows above average at that stat. +4.4% fit (702) / +4.7% blind (383),
# z ~ 1 each. Weak evidence -> half stakes, labelled experimental. Cut it if it's negative after 150 bets.
OVER_MAX_SPREAD = 6.0
OVER_MIN_PROJ_REL = 1.05
OVER_MIN_OPP_ALLOW_REL = 1.02
OVER_STAKE_MULT = 0.5
BLOWOUT_SPREAD = 10.0     # |spread| at or above this = "blowout" tier
# Games-played gates. The backtest sample starts 2 weeks into each season, so it has NO data on players
# with <5 games; the old "10 games" rule was precaution, not evidence. Where there is data, core unders at
# 5-9 games were +20% (62 bets). So: every tier opens at EARLY_MIN_N_CUR games, bets with fewer than
# FULL_STAKE_N_CUR games are flagged "early" and staked at EARLY_STAKE_MULT.
EARLY_MIN_N_CUR = 1       # one real box score this season before any bet (half stakes until FULL_STAKE_N_CUR)
FULL_STAKE_N_CUR = 5      # below this: half stakes, tier labelled "... (early)"
EARLY_STAKE_MULT = 0.5
BLOWOUT_MIN_N_CUR = EARLY_MIN_N_CUR
MIN_N_CUR = EARLY_MIN_N_CUR   # core tier gate
SKIP_USAGE_MODEL_BETS = True  # don't bet players whose projection was touched by a teammate being out
                              #   (those bets ran -3.2% vs +0.7% for the rest). OUT players are still removed.
MAX_DISAGREE = 0.30       # skip (flag) a bet when |projection - line| / line exceeds this: the
                          # market almost certainly knows something (role change, injury) the model doesn't
MIN_EDGE = 0.03           # blended prob - market fair prob, must exceed this to bet
MIN_BOOKS = 2             # need this many books quoting a prop to trust consensus
BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MAX_BET_PCT = 0.03        # hard cap per bet as fraction of bankroll
MAX_PLAYER_PCT = 0.05     # hard cap total exposure on one player per slate

# ---- stats.nba.com request headers (required or you get 403/timeout) --------
NBA_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

TEAM_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

for d in (DATA_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
