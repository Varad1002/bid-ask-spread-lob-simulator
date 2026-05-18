"""
simulator.py -- Stochastic simulation engine with dynamic mid-price.

Key changes from the previous version:
  - Mid-price follows a random walk (not fixed).
  - Order prices use exponential offsets around the moving mid-price.
  - Multiple limit orders, market orders, and cancellations per step.
  - Market orders are large enough to consume depth and widen the spread.
  - Stale-biased cancellation removes orders far from current mid-price.
  - Direct seeding bypasses matching engine for fast initialization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict

from order import Order, Trade, OrderType, Side
from order_book import OrderBook
from matching_engine import MatchingEngine


@dataclass
class SimulationConfig:
    """Configuration for one simulation run."""

    # Simulation length
    num_steps: int = 20_000

    # Initial book seeding
    initial_orders: int = 100_000
    reference_price: float = 100.0
    tick_size: float = 0.01

    # -- Price distribution --
    # Sigma for initial seeding (wide = thin depth per level)
    initial_sigma: float = 2.0
    # Sigma for ongoing limit orders (tighter, near current mid)
    order_sigma: float = 0.30
    # Per-step volatility of the reference price random walk
    mid_price_volatility: float = 0.02

    # -- Order flow per step --
    buy_ratio: float = 0.5
    limit_orders_per_step: int = 3
    market_order_prob: float = 0.5

    # Limit order quantity range
    min_quantity: int = 1
    max_quantity: int = 10
    # Market order quantity range (larger to consume depth)
    market_qty_min: int = 10
    market_qty_max: int = 50

    # -- Cancellation --
    cancels_per_step: int = 5
    # Orders further than this from mid are cancelled with high probability
    stale_distance: float = 1.0

    # Reproducibility
    seed: Optional[int] = 42
    label: str = "default"


class Simulator:
    """
    Stochastic LOB simulator with dynamic mid-price and realistic spread.

    The reference price follows a random walk. Orders are placed with
    exponential offsets around the reference price. Market orders consume
    top-of-book depth. Stale orders far from the current reference are
    preferentially cancelled.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.order_book = OrderBook()
        self.engine = MatchingEngine(self.order_book)
        self._next_order_id = 1
        self._ref_price = config.reference_price
        self.records: List[Dict] = []

    def _new_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    # -- Order generation --

    def _make_limit_order(self, step: int, sigma: float = None) -> Order:
        """
        Generate a limit order with exponential offset from reference price.
        Exponential distribution gives natural depth profile: many orders
        near mid, fewer deeper in the book.
        """
        cfg = self.config
        if sigma is None:
            sigma = cfg.order_sigma

        is_buy = self.rng.random() < cfg.buy_ratio
        side = Side.BUY if is_buy else Side.SELL

        # Exponential offset: most orders near mid, tail extends deep
        offset = abs(self.rng.exponential(sigma))

        if side == Side.BUY:
            price = self._ref_price - offset
        else:
            price = self._ref_price + offset

        # Round to tick
        price = round(round(price / cfg.tick_size) * cfg.tick_size, 2)

        qty = int(self.rng.integers(cfg.min_quantity, cfg.max_quantity + 1))

        return Order(
            id=self._new_id(),
            order_type=OrderType.LIMIT,
            side=side,
            price=price,
            quantity=qty,
            timestamp=step,
        )

    def _make_market_order(self, step: int) -> Order:
        """Generate a market order with larger quantity to consume depth."""
        cfg = self.config
        is_buy = self.rng.random() < cfg.buy_ratio
        side = Side.BUY if is_buy else Side.SELL
        price = float("inf") if side == Side.BUY else 0.0
        qty = int(self.rng.integers(cfg.market_qty_min, cfg.market_qty_max + 1))

        return Order(
            id=self._new_id(),
            order_type=OrderType.MARKET,
            side=side,
            price=price,
            quantity=qty,
            timestamp=step,
        )

    # -- Seeding --

    def _seed_book(self):
        """
        Seed the book with initial orders using WIDE distribution.
        Direct insertion (no matching) since buys are always below mid
        and sells always above mid by construction.
        """
        for _ in range(self.config.initial_orders):
            order = self._make_limit_order(step=0, sigma=self.config.initial_sigma)
            self.order_book.add_limit_order(order)

    # -- Cancellation --

    def _cancel_stale(self):
        """
        Cancel orders using random ID probing, biased toward stale orders
        (those far from current reference price).
        """
        max_id = self._next_order_id
        if max_id <= 1:
            return

        for _ in range(self.config.cancels_per_step):
            # Random probe for an active order (O(1) average)
            order = None
            for _attempt in range(20):
                candidate = int(self.rng.integers(1, max_id))
                order = self.order_book.get_order(candidate)
                if order is not None:
                    break
            if order is None:
                continue

            # Stale-biased: orders far from ref_price are more likely cancelled
            distance = abs(order.price - self._ref_price)
            cancel_prob = min(1.0, 0.1 + distance / self.config.stale_distance)

            if self.rng.random() < cancel_prob:
                self.order_book.cancel_order(order.id)

    # -- State recording --

    def _record_state(self, step: int, trades: List[Trade]):
        self.records.append({
            "step": step,
            "best_bid": self.order_book.best_bid,
            "best_ask": self.order_book.best_ask,
            "spread": self.order_book.spread,
            "mid_price": self.order_book.mid_price,
            "ref_price": self._ref_price,
            "num_buy_orders": self.order_book.num_buy_orders,
            "num_sell_orders": self.order_book.num_sell_orders,
            "total_orders": self.order_book.total_orders,
            "num_trades": len(trades),
            "volume_traded": sum(t.quantity for t in trades),
        })

    # -- Main simulation loop --

    def run(self) -> pd.DataFrame:
        cfg = self.config
        print(f"[{cfg.label}] Seeding {cfg.initial_orders} orders, "
              f"then simulating {cfg.num_steps} steps...")

        self._seed_book()

        for step in range(1, cfg.num_steps + 1):
            trades_this_step: List[Trade] = []

            # 1. Evolve reference price (random walk)
            self._ref_price += self.rng.normal(0, cfg.mid_price_volatility)

            # 2. Generate limit orders around current reference price
            for _ in range(cfg.limit_orders_per_step):
                order = self._make_limit_order(step)
                step_trades = self.engine.process_order(order)
                trades_this_step.extend(step_trades)

            # 3. Maybe generate a market order (consumes top-of-book)
            if self.rng.random() < cfg.market_order_prob:
                order = self._make_market_order(step)
                step_trades = self.engine.process_order(order)
                trades_this_step.extend(step_trades)

            # 4. Cancel stale orders
            self._cancel_stale()

            # 5. Trade impact: nudge ref_price toward last trade price
            if trades_this_step:
                last_price = trades_this_step[-1].price
                self._ref_price = 0.95 * self._ref_price + 0.05 * last_price

            # 6. Record state
            self._record_state(step, trades_this_step)

        df = pd.DataFrame(self.records)
        print(f"[{cfg.label}] Done. {len(df)} steps recorded.")
        return df
