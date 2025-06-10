from models import TradingSignal
from config import TradingParams
import datetime as dt

class SignalEvaluator:
    def __init__(self, sentiment_analyzer, us_news_provider, hk_news_provider):
        self.sentiment_analyzer = sentiment_analyzer
        self.us_news_provider = us_news_provider
        self.hk_news_provider = hk_news_provider

    def evaluate_entry_signal(self, ticker, market, price_movement, current_price):
        if abs(price_movement) < TradingParams.PRICE_TRIGGER:
            return None

        signal_type = 'buy' if price_movement > 0 else 'sell'
        sentiment = 'pending'
        return TradingSignal(
            ticker=ticker,
            market=market,
            signal_type=signal_type,
            confidence=0.0,
            price_movement=price_movement,
            sentiment=sentiment,
            entry_price=current_price,
            timestamp=dt.datetime.now()
        )
