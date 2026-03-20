"""
Trade execution helpers:
- Fetch open/current prices via yfinance
- Parse a portfolio plan (markdown) into structured trade actions via LLM
- Execute paper trades against an Account
"""

import json
import math
import re
from datetime import datetime, timedelta

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def get_close_price(ticker: str, date_str: str) -> float | None:
    """Return the closing price for ticker on date_str (or the nearest prior trading day)."""
    try:
        # Fetch a small window ending on/after date_str to catch the right session
        start = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)
        end   = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        data = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        # Normalize index to date-only for comparison
        data.index = data.index.normalize()
        target = datetime.strptime(date_str, "%Y-%m-%d")
        # Return the close on date_str if available, otherwise the most recent prior close
        on_date = data[data.index <= target]
        if on_date.empty:
            return None
        return float(on_date["Close"].iloc[-1])
    except Exception:
        return None


def get_close_prices(tickers: list[str], date_str: str) -> dict[str, float]:
    """Fetch closing prices for a list of tickers on a given date."""
    prices = {}
    for ticker in tickers:
        price = get_close_price(ticker, date_str)
        if price is not None:
            prices[ticker] = price
    return prices


# Keep old name as alias so existing callers don't break
get_open_price  = get_close_price
get_open_prices = get_close_prices


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch most-recent prices for mark-to-market (uses fast_info)."""
    prices = {}
    for ticker in tickers:
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            if price:
                prices[ticker] = float(price)
        except Exception:
            pass
    return prices


# ---------------------------------------------------------------------------
# Portfolio plan parsing
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """You are a JSON extractor. Given a portfolio action plan in markdown, extract each ticker's trade instruction.

Return ONLY a valid JSON array — no prose, no code fences. Each element must have exactly these fields:
- "ticker": string (stock symbol, uppercase)
- "action": "BUY", "SELL", or "HOLD"
- "conviction": integer 1–10
- "buy_weight": for BUY actions the allocation fraction as a decimal (e.g. 35% → 0.35); null for SELL and HOLD

Example output:
[
  {"ticker": "NVDA", "action": "BUY", "conviction": 8, "buy_weight": 0.35},
  {"ticker": "AAPL", "action": "SELL", "conviction": 7, "buy_weight": null},
  {"ticker": "MSFT", "action": "HOLD", "conviction": 5, "buy_weight": null}
]"""


def parse_portfolio_plan(llm, portfolio_plan_text: str) -> list[dict]:
    """
    Use an LLM to extract structured trade actions from a portfolio plan.

    Returns list of dicts: {ticker, action, conviction, buy_weight}
    """
    response = llm.invoke([
        SystemMessage(content=_PARSE_SYSTEM),
        HumanMessage(content=portfolio_plan_text),
    ])
    content = response.content.strip()

    # Strip markdown code fences if model added them
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```\s*$", "", content, flags=re.MULTILINE)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: find first [...] block
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def execute_paper_trades(account, actions: list[dict], date_str: str, prices: dict[str, float]) -> list[dict]:
    """
    Execute paper trades for an agent account.

    Processes SELLs first (to free up cash), then BUYs, then records HOLDs.

    Args:
        account:   Account instance
        actions:   Parsed trade actions from parse_portfolio_plan()
        date_str:  Trade date string YYYY-MM-DD
        prices:    {ticker: open_price} dict

    Returns:
        List of result dicts for display/logging.
    """
    results = []

    sells = [a for a in actions if a.get("action") == "SELL"]
    buys  = [a for a in actions if a.get("action") == "BUY"]
    holds = [a for a in actions if a.get("action") == "HOLD"]

    # --- SELLs ---
    for action in sells:
        ticker = action["ticker"]
        price = prices.get(ticker)
        if price is None:
            results.append({"ticker": ticker, "action": "SELL", "status": "SKIPPED", "reason": "No price data"})
            continue
        conviction = (action.get("conviction") or 5) / 10.0
        shares, err = account.execute_sell(ticker, price, conviction, date_str)
        if err:
            results.append({"ticker": ticker, "action": "SELL", "status": "FAILED", "reason": err})
        else:
            results.append({
                "ticker": ticker, "action": "SELL", "status": "OK",
                "shares": shares, "price": price,
                "total": shares * price, "conviction": conviction,
            })

    # --- BUYs ---
    for action in buys:
        ticker = action["ticker"]
        price = prices.get(ticker)
        if price is None:
            results.append({"ticker": ticker, "action": "BUY", "status": "SKIPPED", "reason": "No price data"})
            continue
        buy_weight = action.get("buy_weight") or (action.get("conviction") or 5) / 10.0
        shares, err = account.execute_buy(ticker, price, buy_weight, date_str)
        if err:
            results.append({"ticker": ticker, "action": "BUY", "status": "FAILED", "reason": err})
        else:
            results.append({
                "ticker": ticker, "action": "BUY", "status": "OK",
                "shares": shares, "price": price,
                "total": shares * price, "buy_weight": buy_weight,
            })

    # --- HOLDs ---
    for action in holds:
        results.append({"ticker": action["ticker"], "action": "HOLD", "status": "OK"})

    return results
