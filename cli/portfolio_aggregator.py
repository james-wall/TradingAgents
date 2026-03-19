"""Portfolio aggregator — combines individual ticker analyses into a weighted daily recommendation."""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a senior portfolio manager synthesizing individual stock analyses into a cohesive daily trading action plan.

Your task:
1. Review each stock's BUY/SELL/HOLD signal and the reasoning behind it.
2. Assign a conviction score (1–10) to each recommendation based on the strength and consistency of the evidence.
3. For BUY recommendations: allocate weights that sum to 100% (representing how to split available new capital across all BUY positions).
4. For SELL recommendations: rank by urgency and assign a liquidation priority (1 = sell first).
5. For HOLD recommendations: briefly confirm the rationale.
6. Write a short portfolio strategy commentary covering overall market tone and the day's key themes.

Output your response in this exact format:

## Daily Portfolio Action Plan — {date}

### Action Summary
| Ticker | Action | Conviction (1–10) | Weight / Priority | Key Rationale |
|--------|--------|--------------------|-------------------|---------------|
(fill in one row per ticker)

### BUY Allocation (% of available capital)
(For each BUY ticker: ticker, weight %, and 1-2 sentence rationale for the sizing decision)

### SELL Priority (ranked 1 = highest urgency)
(For each SELL ticker: ticker, priority rank, and 1-2 sentence rationale)

### HOLD Positions
(For each HOLD ticker: ticker and brief rationale)

### Portfolio Strategy Commentary
(2-4 sentences on the overall market picture and the day's strategic themes)

Rules:
- BUY weights must sum to exactly 100%.
- Be specific and decisive — avoid vague language like "may" or "might consider".
- If there are no BUY/SELL/HOLD tickers in a category, write "None" for that section."""


def aggregate_portfolio_recommendations(
    llm,
    ticker_results: List[Dict[str, Any]],
    analysis_date: str,
) -> str:
    """
    Synthesize individual ticker analyses into a weighted portfolio recommendation.

    Args:
        llm: LangChain LLM instance to use for aggregation.
        ticker_results: List of dicts, each with keys:
            - ticker (str)
            - signal (str): "BUY", "SELL", or "HOLD"
            - final_trade_decision (str): full decision text from the risk judge
        analysis_date: Date string (YYYY-MM-DD) for the analysis.

    Returns:
        Markdown-formatted portfolio action plan string.
    """
    summaries = []
    for i, result in enumerate(ticker_results, 1):
        summaries.append(
            f"### {i}. {result['ticker']} — Signal: **{result['signal']}**\n\n"
            f"{result['final_trade_decision']}"
        )

    combined = "\n\n---\n\n".join(summaries)

    user_prompt = (
        f"Synthesize the following {len(ticker_results)} stock analyses into a cohesive "
        f"daily portfolio action plan for {analysis_date}.\n\n"
        f"{combined}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(date=analysis_date)),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return response.content
