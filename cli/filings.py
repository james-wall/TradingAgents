"""Filings-driven trade recommendations.

CLI commands:
  - filings-recommend: analyze the most recent 10-K/10-Q for one ticker
  - filings-batch:     analyze recent filings across a tickers file, write a
                       dated markdown report (used by the scheduled GH workflow)

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
