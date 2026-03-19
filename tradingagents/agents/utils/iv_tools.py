from langchain_core.tools import tool
from typing import Annotated
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd


@tool
def get_iv_data(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """
    Retrieve implied volatility (IV) data for a stock using its options chain.
    Returns ATM IV from the nearest expiration, comparison to 20-day historical
    (realized) volatility, and IV term structure across the next 4 expirations.
    High IV vs historical vol suggests options are pricing in an upcoming event
    (earnings, FDA decision, etc.). Useful for assessing market expectations and
    option pricing relative to actual recent volatility.
    """
    from tradingagents.dataflows.y_finance import get_iv_data_YFin
    return get_iv_data_YFin(symbol)
