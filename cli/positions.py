"""Position context helpers — builds the data the aggregator and auto-sell
backstop need to reason about existing holdings."""

import datetime

from paper_trading.database import get_connection, get_positions


def build_position_context(agent_name: str, all_prices: dict, trade_date: str) -> list[dict]:
    """Return per-position context for the LLM aggregator.

    Each entry: ticker, shares, avg_cost, current_price, unrealized_pnl_pct, days_held.
    days_held is calendar days from the agent's earliest BUY of that ticker.
    """
    positions = get_positions(agent_name)
    if not positions:
        return []

    first_buy = _first_buy_dates(agent_name)
    today = datetime.date.fromisoformat(trade_date)

    context = []
    for ticker, pos in positions.items():
        current_price = all_prices.get(ticker, pos["avg_cost"])
        pnl_pct = (current_price - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] else 0.0

        days_held = None
        first = first_buy.get(ticker)
        if first:
            days_held = (today - datetime.date.fromisoformat(first)).days

        context.append({
            "ticker": ticker,
            "shares": pos["shares"],
            "avg_cost": pos["avg_cost"],
            "current_price": current_price,
            "unrealized_pnl_pct": pnl_pct,
            "days_held": days_held,
        })
    return context


def force_sell_stale_positions(account, all_prices: dict, trade_date: str, console, max_days_held: int = 10) -> list[str]:
    """Sell any position held more than max_days_held calendar days.

    Safety net so cash never gets permanently locked when the LLM forgets to
    rotate out of a finished pre-earnings play. Returns list of tickers sold.
    """
    first_buy = _first_buy_dates(account.agent_name)
    today = datetime.date.fromisoformat(trade_date)
    sold = []

    # Snapshot positions; iterating account.positions while mutating is risky
    for ticker in list(account.positions.keys()):
        first = first_buy.get(ticker)
        if not first:
            continue
        days_held = (today - datetime.date.fromisoformat(first)).days
        if days_held <= max_days_held:
            continue

        price = all_prices.get(ticker)
        if not price:
            console.print(f"  [yellow]Auto-sell skipped {ticker}: no price data[/yellow]")
            continue

        shares, err = account.execute_sell(
            ticker, price, 1.0, trade_date,
            notes=f"auto-sell backstop (held {days_held}d > {max_days_held}d)",
        )
        if err:
            console.print(f"  [yellow]Auto-sell {ticker} failed: {err}[/yellow]")
        else:
            console.print(f"  [dim]Auto-sold {shares} {ticker} (held {days_held}d)[/dim]")
            sold.append(ticker)

    return sold


def _first_buy_dates(agent_name: str) -> dict[str, str]:
    """ticker -> earliest BUY date (YYYY-MM-DD) for this agent."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, MIN(date) FROM trades "
            "WHERE agent_name = ? AND action = 'BUY' GROUP BY ticker",
            (agent_name,),
        ).fetchall()
    return {r[0]: r[1] for r in rows}
