
from dataclasses import dataclass
import datetime as dt
from typing import List, Optional

@dataclass
class TradingSignal:
    ticker: str
    market: str
    signal_type: str
    confidence: float
    price_movement: float
    sentiment: str
    entry_price: float
    timestamp: dt.datetime

@dataclass
class Position:
    ticker: str
    market: str
    side: str
    quantity: int
    entry_price: float
    entry_time: dt.datetime
    current_price: Optional[float] = None

@dataclass
class MarketConfig:
    timezone: str
    open_time: str
    close_time: str
    trading_days: List[int]
    tickers: List[str]
    exchange: str
    currency: str


@dataclass
class NewsArticle:
    title: str
    published: dt.datetime
    source: str
    language: str = 'en'
    weight: float = 1.0
    description: Optional[str] = None