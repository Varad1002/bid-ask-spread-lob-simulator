"""
order.py -- Order data structure for the Limit Order Book simulator.
"""

from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a single order in the limit order book."""

    id: int
    order_type: OrderType
    side: Side
    price: float
    quantity: int
    timestamp: int

    remaining_quantity: int = field(init=False)

    def __post_init__(self):
        self.remaining_quantity = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.remaining_quantity <= 0

    def fill(self, qty: int) -> int:
        """Fill this order by `qty` units. Returns the actually filled amount."""
        filled = min(qty, self.remaining_quantity)
        self.remaining_quantity -= filled
        return filled

    def __repr__(self) -> str:
        return (
            f"Order(id={self.id}, type={self.order_type.value}, "
            f"side={self.side.value}, price={self.price:.2f}, "
            f"qty={self.remaining_quantity}/{self.quantity}, t={self.timestamp})"
        )


@dataclass
class Trade:
    """Represents an executed trade between a buy and sell order."""

    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: int

    def __repr__(self) -> str:
        return (
            f"Trade(buy={self.buy_order_id}, sell={self.sell_order_id}, "
            f"price={self.price:.2f}, qty={self.quantity}, t={self.timestamp})"
        )
