"""
Strategy-specific watchlist builders and portfolio aggregator prompts.

Strategies:
  pre-earnings        — tickers with earnings in the next N business days (existing)
  earnings-reversal   — tickers that just reported and moved >5%
  fomc-fade           — broad-market ETFs after a big FOMC-day move
  max-pain-friday     — tickers far from options max pain heading into Friday expiry
"""

import datetime
import math
from typing import Optional

import pandas as pd
import yfinance as yf


# ═══════════════════════════════════════════════════════════════════════════════
# FOMC calendar
# ═══════════════════════════════════════════════════════════════════════════════

# Statement release dates (Fed announces at 2 PM ET on these days)
FOMC_DATES = {
    # 2025
    datetime.date(2025, 1, 29),
    datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7),
    datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30),
    datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29),
    datetime.date(2025, 12, 17),
    # 2026
    datetime.date(2026, 1, 28),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28),
    datetime.date(2026, 12, 16),
}

# ETFs to trade for macro strategies
MACRO_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# Minimum move (%) to trigger FOMC fade
FOMC_MOVE_THRESHOLD = 0.5

# Minimum move (%) to trigger earnings reversal
EARNINGS_MOVE_THRESHOLD = 5.0

# Minimum distance from max pain (%) to include in watchlist
MAX_PAIN_DISTANCE_THRESHOLD = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Earnings overreaction reversal
# ═══════════════════════════════════════════════════════════════════════════════

def build_earnings_reversal_watchlist(
    tickers: list[str],
    trade_date: Optional[str] = None,
    move_threshold: float = EARNINGS_MOVE_THRESHOLD,
) -> list[tuple[str, str, float, float]]:
    """
    Find tickers that reported earnings in the last 2 trading days and moved
    more than `move_threshold` percent.

    Returns list of (ticker, earnings_date, move_pct, post_earnings_close).
    """
    today = pd.Timestamp(trade_date) if trade_date else pd.Timestamp.today()
    today = today.normalize()
    # Look back 3 calendar days to cover weekends
    lookback_start = today - pd.Timedelta(days=5)

    results = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)

            # Find recent earnings date
            earnings_date = _get_recent_earnings_date(t, lookback_start, today)
            if earnings_date is None:
                continue

            # Get price data around the earnings date
            hist = t.history(start=(earnings_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                             end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue

            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            # Pre-earnings close = last close on or before earnings date - 1 trading day
            pre = hist[hist.index.normalize() < earnings_date.normalize()]
            post = hist[hist.index.normalize() >= earnings_date.normalize()]

            if pre.empty or post.empty:
                continue

            pre_close = float(pre["Close"].iloc[-1])
            post_close = float(post["Close"].iloc[-1])
            move_pct = ((post_close - pre_close) / pre_close) * 100

            if abs(move_pct) >= move_threshold:
                results.append((
                    symbol,
                    earnings_date.strftime("%Y-%m-%d"),
                    round(move_pct, 1),
                    round(post_close, 2),
                ))
        except Exception:
            continue

    # Sort by absolute move size (biggest overreactions first)
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    return results


def _get_recent_earnings_date(ticker_obj, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Find an earnings date that falls within [start, end]."""
    # Method 1: earnings_dates DataFrame (includes past dates)
    try:
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            ed.index = pd.to_datetime(ed.index)
            if ed.index.tz is not None:
                ed.index = ed.index.tz_convert(None)
            mask = (ed.index.normalize() >= start.normalize()) & (ed.index.normalize() <= end.normalize())
            recent = ed[mask]
            if not recent.empty:
                return recent.index.max()
    except Exception:
        pass

    # Method 2: calendar dict
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            for d in dates:
                ts = pd.Timestamp(d).normalize()
                if start.normalize() <= ts <= end.normalize():
                    return ts
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FOMC fade
# ═══════════════════════════════════════════════════════════════════════════════

def build_fomc_fade_watchlist(
    trade_date: Optional[str] = None,
    move_threshold: float = FOMC_MOVE_THRESHOLD,
) -> list[tuple[str, float, str]]:
    """
    If yesterday or today was an FOMC announcement day, check if major ETFs
    moved more than `move_threshold` percent.

    Returns list of (ticker, move_pct, fomc_date_str).
    Returns empty list if not in an FOMC window.
    """
    today = datetime.date.fromisoformat(trade_date) if trade_date else datetime.date.today()

    # Check if today or yesterday was FOMC day
    yesterday = today - datetime.timedelta(days=1)
    # Handle weekends: walk back to Friday
    while yesterday.weekday() >= 5:
        yesterday -= datetime.timedelta(days=1)

    fomc_day = None
    if today in FOMC_DATES:
        fomc_day = today
    elif yesterday in FOMC_DATES:
        fomc_day = yesterday
    else:
        return []

    fomc_str = fomc_day.isoformat()
    results = []

    for symbol in MACRO_ETFS:
        try:
            hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            # Compare close before FOMC to most recent close
            pre = hist[hist.index.normalize() < pd.Timestamp(fomc_day).normalize()]
            post = hist[hist.index.normalize() >= pd.Timestamp(fomc_day).normalize()]

            if pre.empty or post.empty:
                continue

            pre_close = float(pre["Close"].iloc[-1])
            post_close = float(post["Close"].iloc[-1])
            move_pct = ((post_close - pre_close) / pre_close) * 100

            if abs(move_pct) >= move_threshold:
                results.append((symbol, round(move_pct, 2), fomc_str))
        except Exception:
            continue

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Max pain Friday fade
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_max_pain(ticker_obj, expiry_date: str) -> Optional[float]:
    """
    Calculate max pain for a given expiry date.
    Max pain = strike price where total $ value of OTM options (calls + puts) is minimized.
    """
    try:
        chain = ticker_obj.option_chain(expiry_date)
    except Exception:
        return None

    calls = chain.calls
    puts = chain.puts

    if calls.empty or puts.empty:
        return None

    # Get all unique strikes
    strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))

    min_pain = float("inf")
    max_pain_strike = None

    for strike in strikes:
        # Total pain = sum of (how much each option loses if stock settles at this strike)
        call_pain = 0
        for _, row in calls.iterrows():
            if strike > row["strike"]:
                call_pain += (strike - row["strike"]) * row["openInterest"]

        put_pain = 0
        for _, row in puts.iterrows():
            if strike < row["strike"]:
                put_pain += (row["strike"] - strike) * row["openInterest"]

        total_pain = call_pain + put_pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = strike

    return max_pain_strike


def build_max_pain_friday_watchlist(
    tickers: list[str],
    trade_date: Optional[str] = None,
    distance_threshold: float = MAX_PAIN_DISTANCE_THRESHOLD,
) -> list[tuple[str, float, float, float]]:
    """
    On Thursdays (or Wednesdays for Good Friday weeks), find stocks
    trading far from their Friday options max pain.

    Returns list of (ticker, current_price, max_pain_price, distance_pct).
    Returns empty list if today is not the right day.
    """
    today = datetime.date.fromisoformat(trade_date) if trade_date else datetime.date.today()

    # Only run on Thursday (3) or Wednesday (2) if Friday is a holiday
    if today.weekday() not in (2, 3):
        return []

    # Find this week's Friday
    days_to_friday = 4 - today.weekday()
    friday = today + datetime.timedelta(days=days_to_friday)
    friday_str = friday.isoformat()

    results = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)

            # Check if Friday expiry exists
            expirations = t.options
            if friday_str not in expirations:
                continue

            # Get current price
            price = t.fast_info.last_price
            if not price:
                continue
            price = float(price)

            # Calculate max pain
            mp = calculate_max_pain(t, friday_str)
            if mp is None:
                continue

            distance_pct = ((price - mp) / mp) * 100

            if abs(distance_pct) >= distance_threshold:
                results.append((symbol, round(price, 2), round(mp, 2), round(distance_pct, 1)))
        except Exception:
            continue

    # Sort by absolute distance (furthest from max pain first)
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy-specific portfolio aggregator prompts
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_PROMPTS = {
    "pre-earnings": None,  # Use default prompt from portfolio_aggregator.py

    "earnings-reversal": """You are a senior portfolio manager specializing in POST-EARNINGS overreaction reversals.

These stocks JUST reported earnings and moved significantly. Your job is to identify which moves were overreactions that will fade back.

Key principles:
- Stocks that drop >10% on decent (not disastrous) earnings often bounce 3-5% within 2-3 days as panic selling exhausts.
- Stocks that rip >10% on good (not exceptional) earnings often fade as profit-taking kicks in.
- The larger the move, the more likely a partial reversal — but catastrophic misses (huge guidance cuts, fraud) do NOT reverse.
- Insider buying after a drop is very bullish. Insider selling after a rip confirms the fade.

Your task:
1. For each stock, evaluate whether the post-earnings move was JUSTIFIED by the actual results or an OVERREACTION.
2. BUY stocks that dropped too hard (fade the selloff — expect a bounce).
3. SELL/SHORT stocks that ripped too hard (fade the rally — expect profit-taking).
4. HOLD if the move seems fairly priced.
5. Assign conviction (1-10) based on how confident you are in the reversal.
6. Allocate BUY weights summing to 100%.

Output your response in this exact format:

## Daily Portfolio Action Plan — {date}

### Action Summary
| Ticker | Action | Conviction (1-10) | Weight / Priority | Key Rationale |
|--------|--------|--------------------|-------------------|---------------|
(fill in one row per ticker)

### BUY Allocation (% of available capital)
(For each BUY ticker: ticker, weight %, and 1-2 sentence rationale)

### SELL Priority (ranked 1 = highest urgency)
(For each SELL ticker: ticker, priority rank, and 1-2 sentence rationale)

### HOLD Positions
(For each HOLD ticker: ticker and brief rationale)

### Portfolio Strategy Commentary
(2-4 sentences on the reversal thesis and risk factors)

Rules:
- BUY weights must sum to exactly 100%.
- Be specific and decisive.
- If there are no BUY/SELL/HOLD tickers in a category, write "None" for that section.""",

    "fomc-fade": """You are a senior portfolio manager specializing in FOMC announcement fades.

The Federal Reserve just announced a rate decision and these broad-market ETFs moved significantly. Historical data shows that the first big move after Fed announcements is frequently an overreaction — retail and algo traders push prices too far, then institutions fade it over the next 1-3 days.

Key principles:
- If the Fed delivered EXACTLY what was expected and the market still moved big, that's a high-conviction fade.
- If the Fed surprised (unexpected rate change, hawkish/dovish pivot), the move may be justified — lower conviction.
- The "FOMC drift" phenomenon: markets tend to reverse 50-70% of the initial post-announcement move within 48 hours.
- SPY and QQQ tend to revert faster than IWM (small caps are slower to reprice).

Your task:
1. Evaluate whether the Fed action was priced in or a genuine surprise.
2. BUY ETFs that dropped (if you think the selloff will fade).
3. SELL ETFs that ripped (if you think the rally will fade).
4. HOLD if the move seems justified.
5. Assign conviction (1-10) — higher for "as expected" reactions, lower for genuine surprises.
6. Allocate BUY weights summing to 100%.

Output your response in this exact format:

## Daily Portfolio Action Plan — {date}

### Action Summary
| Ticker | Action | Conviction (1-10) | Weight / Priority | Key Rationale |
|--------|--------|--------------------|-------------------|---------------|
(fill in one row per ticker)

### BUY Allocation (% of available capital)
(For each BUY ticker: ticker, weight %, and 1-2 sentence rationale)

### SELL Priority (ranked 1 = highest urgency)
(For each SELL ticker: ticker, priority rank, and 1-2 sentence rationale)

### HOLD Positions
(For each HOLD ticker: ticker and brief rationale)

### Portfolio Strategy Commentary
(2-4 sentences on the Fed action and your fade thesis)

Rules:
- BUY weights must sum to exactly 100%.
- Be specific and decisive.
- If there are no BUY/SELL/HOLD tickers in a category, write "None" for that section.""",

    "max-pain-friday": """You are a senior portfolio manager specializing in OPTIONS MAX PAIN mean reversion.

These stocks are trading significantly away from their options max pain level heading into Friday expiration. Max pain is the strike price where the most options (calls + puts) expire worthless — market makers and options sellers have a financial incentive to push prices toward this level as expiration approaches.

Key principles:
- Stocks tend to gravitate toward max pain in the final 1-2 days before expiration, especially when open interest is high.
- Stocks ABOVE max pain tend to face selling pressure (call sellers defending strikes).
- Stocks BELOW max pain tend to face buying pressure (put sellers defending strikes).
- The effect is strongest for large-cap, liquid names with heavy options volume.
- Max pain is NOT a guarantee — strong fundamental catalysts can overpower the gravitational pull.

Your task:
1. For each stock, evaluate the likelihood of mean reversion toward max pain by Friday close.
2. BUY stocks trading well BELOW max pain (expect upward pressure).
3. SELL stocks trading well ABOVE max pain (expect downward pressure).
4. HOLD if the distance is marginal or a catalyst overrides the options setup.
5. Assign conviction (1-10) based on distance from max pain and options volume.
6. Allocate BUY weights summing to 100%.

Output your response in this exact format:

## Daily Portfolio Action Plan — {date}

### Action Summary
| Ticker | Action | Conviction (1-10) | Weight / Priority | Key Rationale |
|--------|--------|--------------------|-------------------|---------------|
(fill in one row per ticker)

### BUY Allocation (% of available capital)
(For each BUY ticker: ticker, weight %, and 1-2 sentence rationale)

### SELL Priority (ranked 1 = highest urgency)
(For each SELL ticker: ticker, priority rank, and 1-2 sentence rationale)

### HOLD Positions
(For each HOLD ticker: ticker and brief rationale)

### Portfolio Strategy Commentary
(2-4 sentences on the max pain setup and key risk factors)

Rules:
- BUY weights must sum to exactly 100%.
- Be specific and decisive.
- If there are no BUY/SELL/HOLD tickers in a category, write "None" for that section.""",
}


def get_strategy_prompt(strategy: str) -> Optional[str]:
    """Return the strategy-specific portfolio aggregator system prompt, or None for default."""
    return STRATEGY_PROMPTS.get(strategy)
