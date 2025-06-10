from models import Position
from config import TradingParams
from ib_insync import Stock, MarketOrder
import logging

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, ib_client):
        self.ib_client = ib_client
        self.open_positions = {}

    def check_exit_conditions(self, position):
        if not position.current_price:
            return None
        pnl_pct = ((position.current_price - position.entry_price) / position.entry_price
                   if position.side == 'buy' else
                   (position.entry_price - position.current_price) / position.entry_price)

        if pnl_pct >= TradingParams.PROFIT_TARGET:
            return 'profit_target'
        if pnl_pct <= -TradingParams.STOP_LOSS:
            return 'stop_loss'
        return None

    async def place_order(self, signal):
        try:
            contract = Stock(signal.ticker, 'SMART' if signal.market == 'US' else 'SEHK', 'USD')
            order = MarketOrder(signal.signal_type.upper(), 100)
            self.ib_client.placeOrder(contract, order)
            self.open_positions[f"{signal.ticker}_{signal.market}"] = Position(
                ticker=signal.ticker, market=signal.market, side=signal.signal_type,
                quantity=100, entry_price=signal.entry_price, entry_time=signal.timestamp)
            return True
        except Exception as e:
            logger.error(f"Order failed for {signal.ticker}: {e}")
            return False
