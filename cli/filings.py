"""Filings-driven trade recommendations and scoring.

CLI commands:
  - filings-recommend: analyze the most recent 10-K/10-Q for one ticker
  - filings-batch:     analyze recent filings across a tickers file, write a
                       dated markdown report (used by the scheduled GH workflow)
  - filings-score:     evaluate a recommendations report against actual returns

These commands intentionally do NOT touch the paper trading DB. They only
produce recommendations the user can review. Paper trading integration is a
follow-up once the strategy proves itself.
"""

import datetime
from pathlib import Path
from typing import Optional

import typer
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from paper_trading.config import load_agent_configs, agent_to_graph_config
from tradingagents.dataflows.edgar import get_recent_filings, get_filing_text
from tradingagents.llm_clients.factory import create_llm_client


console = Console()


# Strategy: post-filing momentum. Academic research shows the market under-
# reacts to fundamental disclosures in 10-K/10-Q, producing 5-30 day drift.
POST_FILING_MOMENTUM_PROMPT = """You are a senior fundamental analyst evaluating a recently-filed SEC document.

Academic research shows the market often under-reacts to fundamental disclosures in 10-K/10-Q filings, producing 5-30 day post-filing drift. Your job is to identify whether this filing reveals materially positive, negative, or neutral information vs the company's recent trajectory.

Focus on:
- Revenue trajectory (vs prior period and analyst expectations if mentioned)
- Margin trends (gross, operating)
- Cash flow vs reported earnings (divergence is meaningful — net income up but cash flow down is a quality red flag)
- Forward guidance changes (raised, lowered, withdrawn)
- Material new risks added or prior risks removed (Item 1A / Risk Factors)
- Tone shifts in MD&A (Management's Discussion & Analysis)

Output in this EXACT format (don't add prose around it):

RECOMMENDATION: BUY | SELL | HOLD
CONVICTION: <1-10 integer>
TIME_HORIZON_DAYS: <integer, typically 5-30>
KEY_REASONS:
- <one-line reason>
- <one-line reason>
- <one-line reason>
RISK_FACTORS:
- <thing that could invalidate the thesis>

Rules:
- BUY = filing reveals materially positive info the market hasn't fully priced
- SELL = materially negative info, especially quality/cash-flow concerns or removed positive guidance
- HOLD = filing is in line with expectations / no actionable signal
- If the filing text is truncated or insufficient, say HOLD with low conviction
"""


def _build_llm(agent_name: str, use_deep: bool = True):
    """Construct a single LLM instance from an agent config (no full graph spin-up)."""
    cfgs = load_agent_configs()
    cfg = next((c for c in cfgs if c["name"] == agent_name), None)
    if not cfg:
        raise typer.BadParameter(
            f"Agent '{agent_name}' not found in agent_configs.yaml. "
            f"Available: {', '.join(c['name'] for c in cfgs)}"
        )

    graph_cfg = agent_to_graph_config(cfg)
    if use_deep:
        provider = graph_cfg.get("deep_llm_provider") or graph_cfg["llm_provider"]
        model = graph_cfg["deep_think_llm"]
        base_url = graph_cfg.get("deep_backend_url") or graph_cfg.get("backend_url")
    else:
        provider = graph_cfg["llm_provider"]
        model = graph_cfg["quick_think_llm"]
        base_url = graph_cfg.get("backend_url")

    client = create_llm_client(provider=provider, model=model, base_url=base_url)
    return client.get_llm()


def _analyze_filing(ticker: str, filing: dict, llm, max_chars: int = 150_000) -> str:
    """Run the LLM on one filing. Returns the raw recommendation string."""
    text = get_filing_text(filing, max_chars=max_chars)
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Filing type: {filing['filing_type']}\n"
        f"Filed date: {filing['filed_date']}\n"
        f"Source: {filing['url']}\n\n"
        f"Filing text (truncated to {max_chars:,} chars to fit context):\n\n"
        f"{text}"
    )
    response = llm.invoke([
        SystemMessage(content=POST_FILING_MOMENTUM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


def filings_recommend(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Stock ticker (e.g. AAPL)"),
    filing_type: Optional[str] = typer.Option(None, "--filing-type", help="10-K or 10-Q (default: most recent of either)"),
    agent: str = typer.Option("hybrid-flash-sonnet-earnings", "--agent", "-a", help="Agent config to source the LLM from"),
    days_back: int = typer.Option(120, "--days-back", help="How far back to look for filings"),
):
    """Analyze the most recent 10-K/10-Q for a ticker and print a recommendation."""
    types = [filing_type.upper()] if filing_type else ["10-K", "10-Q"]
    filings = get_recent_filings(ticker, filing_types=types, days_back=days_back)
    if not filings:
        console.print(f"[yellow]No {'/'.join(types)} found for {ticker} in last {days_back} days.[/yellow]")
        raise typer.Exit(1)

    latest = filings[0]
    console.print(f"[cyan]Analyzing {ticker} {latest['filing_type']} filed {latest['filed_date']}[/cyan]")
    console.print(f"[dim]{latest['url']}[/dim]\n")

    llm = _build_llm(agent, use_deep=True)
    result = _analyze_filing(ticker, latest, llm)
    console.print(result)


def filings_batch(
    tickers_file: str = typer.Option("tickers.txt", "--tickers", "-f", help="File with one ticker per line"),
    output: str = typer.Option(..., "--output", "-o", help="Output markdown file path"),
    agent: str = typer.Option("hybrid-flash-sonnet-earnings", "--agent", "-a", help="Agent config for the LLM"),
    days_back: int = typer.Option(7, "--days-back", help="Only analyze filings filed within this many days"),
):
    """Run filings analysis across a tickers file and write a dated markdown report.

    Only emits a section for tickers that actually have a filing within --days-back,
    so most days will report on a small subset of the universe.
    """
    tickers = []
    for line in Path(tickers_file).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            tickers.append(line.upper())

    if not tickers:
        console.print(f"[red]No tickers found in {tickers_file}[/red]")
        raise typer.Exit(1)

    today = datetime.date.today().isoformat()
    console.print(f"[cyan]Filings batch — {today} — {len(tickers)} tickers, last {days_back} days[/cyan]")

    llm = _build_llm(agent, use_deep=True)

    sections: list[str] = []
    analyzed = 0
    skipped = 0
    errored = 0

    for ticker in tickers:
        try:
            filings = get_recent_filings(ticker, filing_types=["10-K", "10-Q"], days_back=days_back)
        except Exception as e:
            console.print(f"  [red]{ticker}: fetch error — {e}[/red]")
            errored += 1
            continue

        if not filings:
            skipped += 1
            continue

        latest = filings[0]
        console.print(f"  [green]Analyzing {ticker} {latest['filing_type']} ({latest['filed_date']})[/green]")
        try:
            recommendation = _analyze_filing(ticker, latest, llm)
        except Exception as e:
            console.print(f"  [red]{ticker}: analysis error — {e}[/red]")
            errored += 1
            continue

        sections.append(
            f"## {ticker} — {latest['filing_type']} (filed {latest['filed_date']})\n\n"
            f"Source: {latest['url']}\n\n"
            f"```\n{recommendation}\n```\n"
        )
        analyzed += 1

    header = (
        f"# Filings Recommendations — {today}\n\n"
        f"Strategy: post-filing momentum. Generated by `tradingagents filings-batch`.\n\n"
        f"**Universe:** {len(tickers)} tickers  |  "
        f"**Analyzed:** {analyzed}  |  "
        f"**No recent filing:** {skipped}  |  "
        f"**Errored:** {errored}\n\n"
        f"---\n\n"
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(sections))
    console.print(f"\n[bold green]Wrote {out_path}[/bold green]  ({analyzed} recommendation(s))")


# ---------------------------------------------------------------------------
# Scorer — evaluate recommendations against actual returns
# ---------------------------------------------------------------------------

import re

import yfinance as yf
from rich import box
from rich.rule import Rule
from rich.table import Table


def _parse_recommendations(report_path: str) -> list[dict]:
    """Parse a recommendations markdown file into structured dicts.

    Returns list of {ticker, filing_type, filed_date, recommendation, conviction, time_horizon_days}.
    """
    text = Path(report_path).read_text()
    recs = []
    current: dict = {}

    for line in text.splitlines():
        # Header: ## PG — 10-Q (filed 2026-04-24)
        m = re.match(r"^## (\S+)\s+—\s+(\S+)\s+\(filed (\d{4}-\d{2}-\d{2})\)", line)
        if m:
            if current.get("ticker"):
                recs.append(current)
            current = {
                "ticker": m.group(1),
                "filing_type": m.group(2),
                "filed_date": m.group(3),
            }
            continue

        if line.startswith("RECOMMENDATION:"):
            current["recommendation"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONVICTION:"):
            try:
                current["conviction"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                current["conviction"] = 5
        elif line.startswith("TIME_HORIZON_DAYS:"):
            try:
                current["time_horizon_days"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                current["time_horizon_days"] = 15

    if current.get("ticker"):
        recs.append(current)

    return recs


def _get_return(ticker: str, start_date: str, days: int) -> tuple[Optional[float], Optional[str]]:
    """Fetch actual return from start_date over the next `days` trading days.

    Returns (return_pct, eval_date_str) or (None, None) if data is unavailable
    or the eval period hasn't elapsed yet.
    """
    import pandas as pd

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=days + 10)  # overshoot to cover weekends

    try:
        hist = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
    except Exception:
        return None, None

    if hist.empty or len(hist) < 2:
        return None, None

    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)

    # Entry price: first close on or after start_date
    entry_price = float(hist["Close"].iloc[0])

    # Exit price: close on the Nth trading day (0-indexed → iloc[days])
    if len(hist) <= days:
        return None, None  # eval period hasn't elapsed yet

    exit_price = float(hist["Close"].iloc[days])
    eval_date = hist.index[days].strftime("%Y-%m-%d")

    return_pct = ((exit_price - entry_price) / entry_price) * 100
    return round(return_pct, 2), eval_date


def filings_score(
    report: str = typer.Option(..., "--report", "-r", help="Path to a recommendations markdown file"),
    days: Optional[int] = typer.Option(None, "--days", "-d", help="Override holding period (default: each rec's TIME_HORIZON_DAYS)"),
):
    """Score a recommendations report against actual stock returns.

    Parses the report, fetches actual returns for each recommendation over
    the specified holding period, and prints hit rates and alpha vs SPY.
    """
    recs = _parse_recommendations(report)
    if not recs:
        console.print("[red]No recommendations found in report.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Scoring {len(recs)} recommendations from {report}[/cyan]\n")

    results = []
    for rec in recs:
        ticker = rec["ticker"]
        hold_days = days or rec.get("time_horizon_days", 15)
        filed = rec.get("filed_date", "")
        signal = rec.get("recommendation", "HOLD")
        conviction = rec.get("conviction", 5)

        ticker_ret, eval_date = _get_return(ticker, filed, hold_days)
        spy_ret, _ = _get_return("SPY", filed, hold_days)

        results.append({
            **rec,
            "hold_days": hold_days,
            "return_pct": ticker_ret,
            "spy_return_pct": spy_ret,
            "alpha_pct": round(ticker_ret - spy_ret, 2) if (ticker_ret is not None and spy_ret is not None) else None,
            "eval_date": eval_date,
            "correct": _is_correct(signal, ticker_ret) if ticker_ret is not None else None,
        })

    # Split by status
    scored = [r for r in results if r["return_pct"] is not None]
    pending = [r for r in results if r["return_pct"] is None]

    # Results table
    if scored:
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 1))
        table.add_column("Ticker", style="cyan")
        table.add_column("Signal")
        table.add_column("Conv", justify="right")
        table.add_column("Filed")
        table.add_column("Days", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("SPY", justify="right")
        table.add_column("Alpha", justify="right")
        table.add_column("Hit?", justify="center")

        for r in scored:
            sig = r["recommendation"]
            sig_color = "green" if sig == "BUY" else "red" if sig == "SELL" else "yellow"
            ret = r["return_pct"]
            ret_color = "green" if ret > 0 else "red"
            alpha = r["alpha_pct"]
            alpha_color = "green" if alpha and alpha > 0 else "red" if alpha and alpha < 0 else "dim"
            hit = r["correct"]
            hit_str = "[green]Y[/green]" if hit else "[red]N[/red]"

            table.add_row(
                r["ticker"],
                f"[{sig_color}]{sig}[/{sig_color}]",
                str(r.get("conviction", "?")),
                r["filed_date"],
                str(r["hold_days"]),
                f"[{ret_color}]{ret:+.2f}%[/{ret_color}]",
                f"{r['spy_return_pct']:+.2f}%" if r["spy_return_pct"] is not None else "—",
                f"[{alpha_color}]{alpha:+.2f}%[/{alpha_color}]" if alpha is not None else "—",
                hit_str,
            )
        console.print(table)

    if pending:
        console.print(f"\n[dim]{len(pending)} recommendation(s) still pending (eval period not elapsed):[/dim]")
        console.print(f"[dim]  {', '.join(r['ticker'] for r in pending)}[/dim]")

    # Summary stats
    if scored:
        console.print()
        console.print(Rule("Summary", style="bold green"))

        for signal in ("BUY", "SELL", "HOLD"):
            group = [r for r in scored if r["recommendation"] == signal]
            if not group:
                continue
            hits = sum(1 for r in group if r["correct"])
            avg_ret = sum(r["return_pct"] for r in group) / len(group)
            avg_alpha = sum(r["alpha_pct"] for r in group if r["alpha_pct"] is not None)
            alpha_count = sum(1 for r in group if r["alpha_pct"] is not None)
            avg_alpha = avg_alpha / alpha_count if alpha_count else 0

            color = "green" if signal == "BUY" else "red" if signal == "SELL" else "yellow"
            console.print(
                f"  [{color}]{signal}[/{color}]: "
                f"{len(group)} recs  |  "
                f"Hit rate: {hits}/{len(group)} ({hits/len(group)*100:.0f}%)  |  "
                f"Avg return: {avg_ret:+.2f}%  |  "
                f"Avg alpha vs SPY: {avg_alpha:+.2f}%"
            )

        all_alpha = [r["alpha_pct"] for r in scored if r["alpha_pct"] is not None]
        if all_alpha:
            overall_alpha = sum(all_alpha) / len(all_alpha)
            overall_hit = sum(1 for r in scored if r["correct"]) / len(scored) * 100
            console.print(
                f"\n  [bold]Overall[/bold]: "
                f"{len(scored)} scored  |  "
                f"Hit rate: {overall_hit:.0f}%  |  "
                f"Avg alpha: {overall_alpha:+.2f}%"
            )


def _is_correct(signal: str, return_pct: float) -> bool:
    """Did the recommendation direction match the actual return?"""
    if signal == "BUY":
        return return_pct > 0
    elif signal == "SELL":
        return return_pct < 0
    else:  # HOLD
        return abs(return_pct) < 3.0  # HOLD is "correct" if the stock didn't move much
