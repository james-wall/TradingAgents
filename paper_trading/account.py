"""Account class — manages cash, positions, and trade execution for one paper trading agent."""

import math
from . import database as db


MAX_POSITION_PCT = 0.40  # max 40% of portfolio in a single position


class Account:
    def __init__(self, agent_name: str, max_position_pct: float = MAX_POSITION_PCT):
        self.agent_name = agent_name
        self.max_position_pct = max_position_pct
        agent = db.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in database. Run init-paper-trading first.")
        self.starting_balance = float(agent["starting_balance"])

    @property
    def cash(self) -> float:
        return db.get_cash(self.agent_name)

    @property
    def positions(self) -> dict:
        return db.get_positions(self.agent_name)

    def portfolio_value(self, prices: dict) -> float:
        """Mark-to-market total portfolio value (cash + positions)."""
        pos = self.positions
        pos_val = sum(
            p["shares"] * prices.get(ticker, p["avg_cost"])
            for ticker, p in pos.items()
        )
        return self.cash + pos_val

    def positions_value(self, prices: dict) -> float:
        """Mark-to-market value of positions only."""
        pos = self.positions
        return sum(
            p["shares"] * prices.get(ticker, p["avg_cost"])
            for ticker, p in pos.items()
        )

    def execute_buy(self, ticker: str, price: float, buy_weight: float, date_str: str, notes: str = None):
        """
        Buy shares of ticker.

        buy_weight: fraction of current cash to target deploying (0.0–1.0).
                    Further capped so no single position exceeds max_position_pct
                    of total estimated portfolio value.

        Returns (shares_bought, error_message_or_None).
        """
        if price <= 0:
            return 0, "Invalid price"

        cash = self.cash
        positions = self.positions

        # Estimate total portfolio value using avg_cost for positions without fresh prices
        port_value = cash + sum(
            p["shares"] * (price if t == ticker else p["avg_cost"])
            for t, p in positions.items()
        )

        # Current market value in this position
        current_pos_value = positions[ticker]["shares"] * price if ticker in positions else 0.0

        # Max we're allowed in this position
        max_allowed_in_pos = port_value * self.max_position_pct
        headroom = max(0.0, max_allowed_in_pos - current_pos_value)

        # How much we want to spend
        desired_spend = cash * buy_weight

        spend = min(desired_spend, headroom, cash)
        if spend < price:
            return 0, f"Position cap or insufficient funds (headroom ${headroom:.2f}, cash ${cash:.2f})"

        shares = math.floor(spend / price)
        if shares == 0:
            return 0, "Cannot afford even one share"

        total_cost = shares * price

        with db.get_connection() as conn:
            existing = positions.get(ticker)
            if existing:
                new_shares = existing["shares"] + shares
                new_avg = (existing["shares"] * existing["avg_cost"] + total_cost) / new_shares
                db.upsert_position(self.agent_name, ticker, new_shares, new_avg, conn)
            else:
                db.upsert_position(self.agent_name, ticker, shares, price, conn)
            db.set_cash(self.agent_name, cash - total_cost, conn)
            db.record_trade(self.agent_name, date_str, ticker, "BUY", shares, price, buy_weight, notes, conn=conn)

        return shares, None

    def execute_sell(self, ticker: str, price: float, conviction: float, date_str: str, notes: str = None):
        """
        Sell a fraction of a position proportional to conviction (0.0–1.0).
        conviction=1.0 → liquidate fully, conviction=0.5 → sell half.

        Returns (shares_sold, error_message_or_None).
        """
        if price <= 0:
            return 0, "Invalid price"

        positions = self.positions
        if ticker not in positions:
            return 0, f"No position in {ticker}"

        pos = positions[ticker]
        shares_to_sell = math.floor(pos["shares"] * conviction)
        shares_to_sell = max(1, min(shares_to_sell, math.floor(pos["shares"])))

        proceeds = shares_to_sell * price
        remaining = pos["shares"] - shares_to_sell

        with db.get_connection() as conn:
            db.upsert_position(self.agent_name, ticker, remaining, pos["avg_cost"], conn)
            db.set_cash(self.agent_name, self.cash + proceeds, conn)
            db.record_trade(self.agent_name, date_str, ticker, "SELL", shares_to_sell, price, conviction, notes, conn=conn)

        return shares_to_sell, None

    def take_snapshot(self, date_str: str, prices: dict):
        """Record a daily portfolio snapshot to the database."""
        pos_val = self.positions_value(prices)
        db.record_snapshot(self.agent_name, date_str, self.cash, pos_val, self.starting_balance)
