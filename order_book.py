"""
order_book.py -- Optimized Limit Order Book with O(1) counters and random access.
"""

from sortedcontainers import SortedList
from typing import Optional, List, Dict
from order import Order, Side, OrderType


class PriceLevel:
    """A queue of orders at the same price level (FIFO)."""

    def __init__(self, price: float):
        self.price = price
        self.orders: List[Order] = []

    @property
    def total_quantity(self) -> int:
        return sum(o.remaining_quantity for o in self.orders)

    @property
    def is_empty(self) -> bool:
        return len(self.orders) == 0

    def add_order(self, order: Order):
        self.orders.append(order)

    def remove_order(self, order_id: int) -> Optional[Order]:
        for i, order in enumerate(self.orders):
            if order.id == order_id:
                return self.orders.pop(i)
        return None

    def peek(self) -> Optional[Order]:
        if self.orders:
            return self.orders[0]
        return None


class OrderBook:
    """
    Limit Order Book with bid/ask sides and O(1) order count tracking.

    - Bid side: sorted descending by price (best bid = highest)
    - Ask side: sorted ascending by price (best ask = lowest)
    """

    def __init__(self):
        self.bid_levels: Dict[float, PriceLevel] = {}
        self.ask_levels: Dict[float, PriceLevel] = {}
        self._bid_prices = SortedList()  # stores -price for descending
        self._ask_prices = SortedList()  # stores price for ascending
        self._orders: Dict[int, Order] = {}

        # O(1) counters -- updated incrementally
        self._buy_count = 0
        self._sell_count = 0

    # -- Properties (O(1)) --

    @property
    def best_bid(self) -> Optional[float]:
        if self._bid_prices:
            return -self._bid_prices[0]
        return None

    @property
    def best_ask(self) -> Optional[float]:
        if self._ask_prices:
            return self._ask_prices[0]
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return round(ba - bb, 6)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    @property
    def num_buy_orders(self) -> int:
        return self._buy_count

    @property
    def num_sell_orders(self) -> int:
        return self._sell_count

    @property
    def total_orders(self) -> int:
        return self._buy_count + self._sell_count

    # -- Order access --

    def has_order(self, order_id: int) -> bool:
        return order_id in self._orders

    def get_order(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)

    # -- Core Operations --

    def add_limit_order(self, order: Order):
        """Add a limit order to the appropriate side."""
        if order.side == Side.BUY:
            if order.price not in self.bid_levels:
                self.bid_levels[order.price] = PriceLevel(order.price)
                self._bid_prices.add(-order.price)
            self.bid_levels[order.price].add_order(order)
            self._buy_count += 1
        else:
            if order.price not in self.ask_levels:
                self.ask_levels[order.price] = PriceLevel(order.price)
                self._ask_prices.add(order.price)
            self.ask_levels[order.price].add_order(order)
            self._sell_count += 1

        self._orders[order.id] = order

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Remove an order from the book by its ID."""
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]

        if order.side == Side.BUY:
            levels, prices, price_key = self.bid_levels, self._bid_prices, -order.price
        else:
            levels, prices, price_key = self.ask_levels, self._ask_prices, order.price

        if order.price in levels:
            removed = levels[order.price].remove_order(order_id)
            if removed and levels[order.price].is_empty:
                del levels[order.price]
                prices.remove(price_key)

        if order.side == Side.BUY:
            self._buy_count -= 1
        else:
            self._sell_count -= 1

        del self._orders[order_id]
        return order

    def get_best_bid_level(self) -> Optional[PriceLevel]:
        bb = self.best_bid
        return self.bid_levels.get(bb) if bb is not None else None

    def get_best_ask_level(self) -> Optional[PriceLevel]:
        ba = self.best_ask
        return self.ask_levels.get(ba) if ba is not None else None

    def _clean_empty_level(self, side: Side, price: float):
        if side == Side.BUY:
            if price in self.bid_levels and self.bid_levels[price].is_empty:
                del self.bid_levels[price]
                self._bid_prices.remove(-price)
        else:
            if price in self.ask_levels and self.ask_levels[price].is_empty:
                del self.ask_levels[price]
                self._ask_prices.remove(price)

    def remove_filled_order(self, order: Order):
        """Remove a fully-filled order from tracking and decrement count."""
        if order.id in self._orders:
            if order.side == Side.BUY:
                self._buy_count -= 1
            else:
                self._sell_count -= 1
            del self._orders[order.id]
