"""Benchmark "agents" that bypass the LLM pipeline.

Used to provide baselines on the leaderboard:
  - spy-buy-hold:        buy SPY on day 1, hold forever
  - watchlist-equal-weight: each day, deploy remaining cash equally across the
                            day's watchlist tickers
"""

from paper_trading.account import Account
from paper_trading.database import get_latest_snapshot
from paper_trading.execution import get_close_price


def run_benchmark(agent_cfg, tickers, all_prices, trade_date, console, dry_run=False):
    name = agent_cfg["name"]
    strategy = agent_cfg.get("benchmark_strategy")

    if dry_run:
        console.print(f"  [dim](dry run — benchmark {strategy} not executed)[/dim]")
        return

    # Benchmarks bypass the 40% per-position cap so they can fully deploy capital.
    account = Account(name, max_position_pct=1.0)

    if strategy == "spy-buy-hold":
        _run_spy_hold(account, all_prices, trade_date, console)
    elif strategy == "watchlist-equal-weight":
        _run_watchlist_basket(account, tickers, all_prices, trade_date, console)
    else:
        console.print(f"  [red]Unknown benchmark strategy: {strategy}[/red]")
        return

    # Always snapshot, even if no trades, so we can mark-to-market.
    snap_prices = dict(all_prices)
    if "SPY" not in snap_prices:
        spy_price = get_close_price("SPY", trade_date)
        if spy_price:
            snap_prices["SPY"] = spy_price
    account.take_snapshot(trade_date, snap_prices)

    snap = get_latest_snapshot(name)
    if snap:
        daily = f"{snap['daily_return_pct']:+.2f}%" if snap.get("daily_return_pct") is not None else "—"
        console.print(
            f"  Portfolio: [bold]${snap['portfolio_value']:,.2f}[/bold]  "
            f"Cash: ${snap['cash']:,.2f}  "
            f"Day: {daily}  "
            f"Total: [bold]{snap['cumulative_return_pct']:+.2f}%[/bold]"
        )


def _run_spy_hold(account, all_prices, trade_date, console):
    """Buy SPY with all cash on the first run; do nothing thereafter."""
    if account.cash < 1.0:
        console.print("  [dim]SPY benchmark already deployed — holding.[/dim]")
        return

    price = all_prices.get("SPY") or get_close_price("SPY", trade_date)
    if not price:
        console.print("  [red]Could not fetch SPY price — skipping.[/red]")
        return

    shares, err = account.execute_buy("SPY", price, 1.0, trade_date, notes="benchmark: SPY buy-and-hold")
    if err:
        console.print(f"  [red]SPY buy failed: {err}[/red]")
    else:
        console.print(f"  [green]Bought {shares} SPY @ ${price:.2f} = ${shares * price:,.2f}[/green]")


def _run_watchlist_basket(account, tickers, all_prices, trade_date, console):
    """Equal-weight today's watchlist using whatever cash is available."""
    if account.cash < 1.0:
        console.print("  [dim]Basket already fully deployed — holding.[/dim]")
        return

    valid_tickers = [t for t in tickers if all_prices.get(t)]
    if not valid_tickers:
        console.print("  [yellow]No priced tickers in watchlist — skipping.[/yellow]")
        return

    weight_per_ticker = 1.0 / len(valid_tickers)
    for ticker in valid_tickers:
        price = all_prices[ticker]
        shares, err = account.execute_buy(
            ticker, price, weight_per_ticker, trade_date,
            notes=f"benchmark: equal-weight basket ({len(valid_tickers)} tickers)",
        )
        if err:
            console.print(f"  [yellow]{ticker}: {err}[/yellow]")
        else:
            console.print(f"  [green]Bought {shares} {ticker} @ ${price:.2f} = ${shares * price:,.2f}[/green]")
