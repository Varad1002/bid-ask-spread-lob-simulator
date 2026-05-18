"""
matching_engine.py -- Matching engine with price-time priority.
"""

from typing import List
from order import Order, Trade, Side, OrderType
from order_book import OrderBook


class MatchingEngine:
    """
    Matching engine that processes incoming orders against the order book.

    Rules:
      - Price priority: best prices matched first.
      - Time priority: FIFO within the same price level.
      - Limit orders: try to match, then rest in the book.
      - Market orders: execute immediately against best available.
    """

    def __init__(self, order_book: OrderBook):
        self.order_book = order_book
        self.trades: List[Trade] = []

    def process_order(self, order: Order) -> List[Trade]:
        if order.order_type == OrderType.MARKET:
            return self._process_market_order(order)
        else:
            return self._process_limit_order(order)

    def _process_market_order(self, order: Order) -> List[Trade]:
        trades = []
        if order.side == Side.BUY:
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)
        self.trades.extend(trades)
        return trades

    def _process_limit_order(self, order: Order) -> List[Trade]:
        trades = []
        if order.side == Side.BUY:
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)

        # Rest any unfilled quantity in the book
        if not order.is_filled:
            self.order_book.add_limit_order(order)

        self.trades.extend(trades)
        return trades

    def _match_buy(self, incoming: Order) -> List[Trade]:
        trades = []
        ob = self.order_book

        while not incoming.is_filled:
            ask_level = ob.get_best_ask_level()
            if ask_level is None:
                break

            if incoming.order_type == OrderType.LIMIT and incoming.price < ask_level.price:
                break

            while not incoming.is_filled and not ask_level.is_empty:
                resting = ask_level.peek()
                if resting is None:
                    break

                trade_qty = min(incoming.remaining_quantity, resting.remaining_quantity)
                trade_price = resting.price

                incoming.fill(trade_qty)
                resting.fill(trade_qty)

                trade = Trade(
                    buy_order_id=incoming.id,
                    sell_order_id=resting.id,
                    price=trade_price,
                    quantity=trade_qty,
                    timestamp=incoming.timestamp,
                )
                trades.append(trade)

                if resting.is_filled:
                    ask_level.orders.pop(0)
                    ob.remove_filled_order(resting)

            ob._clean_empty_level(Side.SELL, ask_level.price)

        return trades

    def _match_sell(self, incoming: Order) -> List[Trade]:
        trades = []
        ob = self.order_book

        while not incoming.is_filled:
            bid_level = ob.get_best_bid_level()
            if bid_level is None:
                break

            if incoming.order_type == OrderType.LIMIT and incoming.price > bid_level.price:
                break

            while not incoming.is_filled and not bid_level.is_empty:
                resting = bid_level.peek()
                if resting is None:
                    break

                trade_qty = min(incoming.remaining_quantity, resting.remaining_quantity)
                trade_price = resting.price

                incoming.fill(trade_qty)
                resting.fill(trade_qty)

                trade = Trade(
                    buy_order_id=resting.id,
                    sell_order_id=incoming.id,
                    price=trade_price,
                    quantity=trade_qty,
                    timestamp=incoming.timestamp,
                )
                trades.append(trade)

                if resting.is_filled:
                    bid_level.orders.pop(0)
                    ob.remove_filled_order(resting)

            ob._clean_empty_level(Side.BUY, bid_level.price)

        return trades
