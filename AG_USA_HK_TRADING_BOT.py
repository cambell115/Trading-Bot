# US-HKEX 24/7 Trading Bot with Curated Stock Lists - Modularized Architecture

import yfinance as yf
import requests
import time
import datetime as dt
import pytz
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from ib_insync import *
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

try:
    import feedparser
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "feedparser"])
    import feedparser

try:
    from googletrans import Translator
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "googletrans==4.0.0rc1"])
    from googletrans import Translator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS & CONFIGURATION
# =============================================================================

@dataclass
class TradingSignal:
    ticker: str
    market: str
    signal_type: str  # 'buy' or 'sell'
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

# Trading Parameters
class TradingParams:
    PRICE_TRIGGER = 0.006
    PROFIT_TARGET = 0.004
    STOP_LOSS = 0.003
    POSITION_SIZE_PCT = 0.10
    MAX_POSITIONS_PER_MARKET = 6
    SCAN_INTERVAL = 5
    CLOSE_POSITIONS_BEFORE_CLOSE = 15

# API Keys
NEWS_API_KEY = "1f9206a5e9b6438485a914e4b35a2138"
FINNHUB_API_KEY = "d1360q9r01qv1k0onergd1360q9r01qv1k0ones0"

# Global caches
news_cache = {}
translation_cache = {}
CACHE_DURATION = 300  # 5 minutes

# =============================================================================
# STOCK LISTS & MARKET CONFIGURATION
# =============================================================================

class StockListManager:
    @staticmethod
    def get_curated_hkex_stocks() -> List[str]:
        CURATED_HKEX_STOCKS = {
            'TECH_AI_INTERNET': [
                '0700', '9988', '9618', '3690', '1810', '2382', '2013', '2858', 
                '1478', '3888', '1347', '1415', '1181', '7878', '669'
            ],
            'HEALTHCARE_BIOTECH': [
                '1093', '1177', '2269', '1877', '9999', '2318', '2196', '6098'
            ],
            'CONSUMER_RETAIL_FOOD': [
                '2319', '322', '291', '960', '9633', '2038', '110', '6808', '992', '2020'
            ],
            'PROPERTY_INFRASTRUCTURE': [
                '0001', '0002', '0016', '0836', '1038', '1113', '0003', '0066', '1997', '101'
            ],
            'FINANCIALS_VOLATILE': [
                '1398', '2318', '3988', '6881', '388', '6658', '6060'
            ]
        }
        
        all_stocks = []
        for stocks in CURATED_HKEX_STOCKS.values():
            all_stocks.extend(stocks)
        return list(dict.fromkeys(all_stocks))  # Remove duplicates
    
    @staticmethod
    def get_curated_us_stocks() -> List[str]:
        CURATED_US_STOCKS = {
            'TECH_AI_CHIPS': [
                'NVDA', 'AMD', 'SMCI', 'ARM', 'AVGO', 'INTC', 'MU', 'QCOM', 'TSM',
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'CRM', 'ORCL', 'IBM', 'SNOW',
                'PLTR', 'DDOG', 'ZS', 'NET', 'MDB', 'OKTA', 'PANW', 'CRWD', 'ANET',
                'ADBE', 'SHOP', 'U', 'TWLO', 'HUBS', 'DOCN', 'ASML', 'TTD'
            ],
            'HEALTHCARE_BIOTECH': [
                'JNJ', 'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'GILD', 'AMGN', 'REGN', 'BIIB',
                'VRTX', 'AZN', 'CVS', 'CI', 'UNH', 'HUM', 'TMO', 'SYK', 'ISRG', 'ZBH'
            ],
            'RETAIL_CONSUMER': [
                'WMT', 'COST', 'TGT', 'AMZN', 'HD', 'LOW', 'MCD', 'SBUX', 'TPR', 'ULTA',
                'NKE', 'LULU', 'DG', 'DLTR', 'BBY', 'KR', 'TJX', 'ROST', 'ETSY', 'EBAY'
            ],
            'FINTECH_MEGACAP': [
                'PYPL', 'SQ', 'COIN', 'HOOD', 'V', 'MA', 'AXP', 'BKNG', 'UBER', 'LYFT'
            ],
            'NEWSMAKER_OUTLIERS': [
                'TSLA', 'RIVN', 'LCID', 'XPEV', 'NIO', 'NKLA', 'GME', 'AMC', 'PARA',
                'DIS', 'NFLX', 'DWAC', 'SNAP', 'BZFD', 'ROKU'
            ]
        }
        
        all_stocks = []
        for stocks in CURATED_US_STOCKS.values():
            all_stocks.extend(stocks)
        return list(dict.fromkeys(all_stocks))

class MarketConfigManager:
    @staticmethod
    def get_market_configs() -> Dict[str, MarketConfig]:
        return {
            'US': MarketConfig(
                timezone='US/Eastern',
                open_time='09:30',
                close_time='16:00',
                trading_days=[0, 1, 2, 3, 4],
                tickers=StockListManager.get_curated_us_stocks(),
                exchange='SMART',
                currency='USD'
            ),
            'HKEX': MarketConfig(
                timezone='Asia/Hong_Kong',
                open_time='09:30',
                close_time='16:00',
                trading_days=[0, 1, 2, 3, 4],
                tickers=StockListManager.get_curated_hkex_stocks(),
                exchange='SEHK',
                currency='HKD'
            )
        }

# =============================================================================
# SENTIMENT ANALYSIS ENGINE
# =============================================================================

class SentimentAnalyzer:
    def __init__(self):
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.translator = None
    
    def initialize(self) -> bool:
        try:
            logger.info("Loading FinBERT...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.translator = Translator()
            logger.info("FinBERT loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return False
    
    def analyze_sentiment_finbert(self, text: str) -> str:
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)
            
            labels = ['negative', 'neutral', 'positive']
            scores = predictions[0].numpy()
            max_idx = np.argmax(scores)
            confidence = scores[max_idx]
            
            return labels[max_idx] if confidence > 0.6 else 'neutral'
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 'neutral'
    
    async def translate_chinese_to_english(self, chinese_text: str) -> str:
        if not chinese_text or len(chinese_text.strip()) < 5:
            return chinese_text
        
        cache_key = chinese_text[:100]
        if cache_key in translation_cache:
            return translation_cache[cache_key]
        
        try:
            result = self.translator.translate(chinese_text, src='auto', dest='en')
            translated_text = result.text
            translation_cache[cache_key] = translated_text
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return chinese_text

# =============================================================================
# NEWS DATA PROVIDERS
# =============================================================================

class NewsDataProvider:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
    
    @staticmethod
    def calculate_recency_weight(published_time: dt.datetime, current_time: dt.datetime) -> float:
        minutes_ago = (current_time - published_time).total_seconds() / 60
        if minutes_ago <= 5:
            return 3.0
        elif minutes_ago <= 10:
            return 2.0
        elif minutes_ago <= 15:
            return 1.0
        else:
            return 0.5
    
    @staticmethod
    def get_source_reliability_weight(source_name: str) -> float:
        weights = {
            'Reuters Business': 2.0, 'Bloomberg': 2.0, 'SCMP Finance': 1.8,
            'Finnhub': 1.8, 'Reuters Asia': 1.7, 'CNBC': 1.5,
            'MarketWatch': 1.3, 'Yahoo Finance': 1.2, 'Seeking Alpha': 1.1,
            'Hong Kong Economic Times': 1.4, 'Sina Finance HK': 1.3,
            'NewsAPI': 1.0
        }
        return weights.get(source_name, 1.0)
    
    @staticmethod
    def remove_duplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_words = set(article.title.lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words) > 0 and len(seen_words) > 0:
                    overlap = len(title_words.intersection(seen_words))
                    similarity = overlap / min(len(title_words), len(seen_words))
                    if similarity > 0.7:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(article.title.lower())
        
        return unique_articles

class USNewsProvider(NewsDataProvider):
    async def get_finnhub_news(self, ticker: str, minutes_back: int = 15) -> List[NewsArticle]:
        try:
            now = dt.datetime.now()
            from_time = int((now - dt.timedelta(minutes=minutes_back)).timestamp())
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': dt.datetime.fromtimestamp(from_time).strftime('%Y-%m-%d'),
                'to': now.strftime('%Y-%m-%d'),
                'token': FINNHUB_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json()
                recent_articles = []
                
                for article in articles:
                    article_time = dt.datetime.fromtimestamp(article.get('datetime', 0))
                    if (now - article_time).total_seconds() <= minutes_back * 60:
                        recent_articles.append(NewsArticle(
                            title=article.get('headline', ''),
                            published=article_time,
                            source='Finnhub'
                        ))
                
                return recent_articles
            return []
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return []
    
    async def get_rss_news(self, ticker: str, company_name: str, minutes_back: int = 15) -> List[NewsArticle]:
        rss_feeds = {
            'Yahoo Finance': f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}',
            'MarketWatch': f'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'Seeking Alpha': f'https://seekingalpha.com/api/sa/combined/{ticker}.xml',
            'Reuters Business': 'https://feeds.reuters.com/reuters/businessNews',
            'CNBC': 'https://feeds.feedburner.com/reuters/businessNews'
        }
        
        all_articles = []
        now = dt.datetime.now()
        
        for source_name, feed_url in rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:
                    title = entry.get('title', '')
                    
                    if ticker.lower() in title.lower() or company_name.lower() in title.lower():
                        published_time = now
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_time = dt.datetime(*entry.published_parsed[:6])
                        
                        if (now - published_time).total_seconds() <= minutes_back * 60:
                            all_articles.append(NewsArticle(
                                title=title,
                                published=published_time,
                                source=source_name
                            ))
                            
            except Exception as e:
                logger.error(f"RSS feed error for {source_name}: {e}")
                continue
        
        return all_articles
    
    async def get_enhanced_newsapi(self, ticker: str, company_name: str, minutes_back: int = 15) -> List[NewsArticle]:
        try:
            now = dt.datetime.now()
            from_time = now - dt.timedelta(minutes=minutes_back)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} OR "{company_name}"',
                'from': from_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'to': now.strftime('%Y-%m-%dT%H:%M:%S'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('articles', []):
                    title = article.get('title', '')
                    published_str = article.get('publishedAt', '')
                    
                    if ticker.lower() in title.lower() or company_name.lower() in title.lower():
                        try:
                            published_time = dt.datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                            published_time = published_time.replace(tzinfo=None)
                            
                            if (now - published_time).total_seconds() <= minutes_back * 60:
                                articles.append(NewsArticle(
                                    title=title,
                                    published=published_time,
                                    source='NewsAPI'
                                ))
                        except:
                            continue
                            
                return articles
            return []
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

class HKNewsProvider(NewsDataProvider):
    def get_hk_company_names(self, ticker: str) -> Dict[str, str]:
        company_names = {
            '0700': {'en': 'Tencent Holdings', 'zh_hant': '騰訊控股', 'zh_hans': '腾讯控股'},
            '9988': {'en': 'Alibaba Group', 'zh_hant': '阿里巴巴集團', 'zh_hans': '阿里巴巴集团'},
            '9618': {'en': 'JD.com', 'zh_hant': '京東集團', 'zh_hans': '京东集团'},
            '3690': {'en': 'Meituan', 'zh_hant': '美團', 'zh_hans': '美团'},
            '1810': {'en': 'Xiaomi Corp', 'zh_hant': '小米集團', 'zh_hans': '小米集团'},
            # Add more as needed...
        }
        
        return company_names.get(ticker, {
            'en': ticker, 'zh_hant': ticker, 'zh_hans': ticker
        })
    
    async def get_english_rss_news(self, ticker: str, company_names: Dict[str, str], minutes_back: int = 15) -> List[NewsArticle]:
        english_rss_feeds = {
            'SCMP Finance': 'https://www.scmp.com/rss/322/feed',
            'Yahoo Finance HK': 'https://hk.finance.yahoo.com/rss/',
            'Investing.com HK': 'https://www.investing.com/rss/news_301.rss',
            'AAStocks News': 'https://www.aastocks.com/en/stocks/news/rss.aspx',
            'Reuters Asia': 'https://feeds.reuters.com/reuters/AsiaBusinessNews'
        }
        
        all_articles = []
        now = dt.datetime.now()
        
        search_terms = [
            ticker, ticker.lstrip('0'), f"HK:{ticker}",
            company_names['en'].lower()
        ]
        
        for source_name, feed_url in english_rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:15]:
                    title = entry.get('title', '').lower()
                    description = entry.get('description', '').lower()
                    content = f"{title} {description}"
                    
                    if any(term.lower() in content for term in search_terms):
                        published_time = now
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published_time = dt.datetime(*entry.published_parsed[:6])
                            except:
                                pass
                        
                        if (now - published_time).total_seconds() <= minutes_back * 60:
                            all_articles.append(NewsArticle(
                                title=entry.get('title', ''),
                                published=published_time,
                                source=source_name,
                                language='en'
                            ))
                            
            except Exception as e:
                logger.error(f"Error fetching {source_name}: {e}")
                continue
        
        return all_articles

# =============================================================================
# MARKET DATA & TRADING ENGINE
# =============================================================================

class MarketDataProvider:
    def __init__(self, ib_client):
        self.ib_client = ib_client
    
    def create_contract(self, ticker: str, market: str, markets_config: Dict[str, MarketConfig]) -> Stock:
        market_config = markets_config[market]
        if market == 'US':
            return Stock(ticker, market_config.exchange, market_config.currency)
        elif market == 'HKEX':
            return Stock(ticker, 'SEHK', market_config.currency)
    
    async def get_price_data(self, ticker: str, market: str, markets_config: Dict[str, MarketConfig]) -> Optional[Tuple[float, float, float]]:
        try:
            contract = self.create_contract(ticker, market, markets_config)
            self.ib_client.reqMktData(contract, '', False, False)
            await asyncio.sleep(3)
            
            ticker_data = self.ib_client.ticker(contract)
            
            if (ticker_data and 
                hasattr(ticker_data, 'last') and ticker_data.last and ticker_data.last > 0 and
                hasattr(ticker_data, 'open') and ticker_data.open and ticker_data.open > 0):
                
                current_price = float(ticker_data.last)
                open_price = float(ticker_data.open)
                movement = (current_price - open_price) / open_price
                
                self.ib_client.cancelMktData(contract)
                return movement, open_price, current_price
            else:
                self.ib_client.cancelMktData(contract)
                return None
                
        except Exception as e:
            logger.error(f"Market data error for {ticker}: {e}")
            return None
    
    async def scan_single_stock_price(self, ticker: str, market: str, markets_config: Dict[str, MarketConfig]) -> Optional[Dict]:
        """Scan a single stock for price movement - optimized for parallel execution"""
        try:
            price_data = await self.get_price_data(ticker, market, markets_config)
            if price_data is None:
                return None
            
            movement, open_price, current_price = price_data
            
            # Only return if movement exceeds trigger
            if abs(movement) >= TradingParams.PRICE_TRIGGER:
                return {
                    'ticker': ticker,
                    'market': market,
                    'movement': movement,
                    'open_price': open_price,
                    'current_price': current_price,
                    'signal_type': 'buy' if movement >= TradingParams.PRICE_TRIGGER else 'sell'
                }
            return None
            
        except Exception as e:
            logger.error(f"Error scanning {ticker} price: {e}")
            return None
    
    async def scan_all_prices_parallel(self, tickers: List[str], market: str, markets_config: Dict[str, MarketConfig], batch_size: int = 10) -> List[Dict]:
        """Scan all stocks in parallel batches for price movements"""
        logger.info(f"Scanning {len(tickers)} {market} stocks for price movements (parallel)...")
        
        price_movers = []
        
        # Process in batches to avoid overwhelming IB API
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
            
            # Create tasks for parallel execution
            tasks = [
                self.scan_single_stock_price(ticker, market, markets_config)
                for ticker in batch
            ]
            
            # Execute batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect valid movers
            for result in results:
                if isinstance(result, dict) and result is not None:
                    price_movers.append(result)
                    logger.info(f"MOVER FOUND: {result['ticker']} {result['movement']:.2%} -> {result['signal_type'].upper()}")
            
            # Small delay between batches to be nice to IB API
            if i + batch_size < len(tickers):
                await asyncio.sleep(1)
        
        logger.info(f"Price scan complete: {len(price_movers)} movers found from {len(tickers)} stocks")
        return price_movers

class SignalEvaluator:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, us_news_provider: USNewsProvider, hk_news_provider: HKNewsProvider):
        self.sentiment_analyzer = sentiment_analyzer
        self.us_news_provider = us_news_provider
        self.hk_news_provider = hk_news_provider
    
    def evaluate_entry_signal(self, ticker: str, market: str, price_movement: float, current_price: float) -> Optional[TradingSignal]:
        """Evaluate if current conditions warrant a trading signal"""
        try:
            if abs(price_movement) < TradingParams.PRICE_TRIGGER:
                return None
            
            # Determine required sentiment direction
            if price_movement >= TradingParams.PRICE_TRIGGER:
                required_sentiment = 'positive'
                signal_type = 'buy'
            elif price_movement <= -TradingParams.PRICE_TRIGGER:
                required_sentiment = 'negative'
                signal_type = 'sell'
            else:
                return None
            
            # Get sentiment (this would be called from async context)
            # For now, return a placeholder - this needs to be refactored for async
            return TradingSignal(
                ticker=ticker,
                market=market,
                signal_type=signal_type,
                confidence=0.0,  # Will be set by sentiment analysis
                price_movement=price_movement,
                sentiment='pending',  # Will be set by sentiment analysis
                entry_price=current_price,
                timestamp=dt.datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating entry signal for {ticker}: {e}")
            return None

class PositionManager:
    def __init__(self, ib_client):
        self.ib_client = ib_client
        self.open_positions: Dict[str, Position] = {}
    
    def check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed based on P&L or risk management"""
        try:
            if not position.current_price:
                return None
            
            entry_price = position.entry_price
            current_price = position.current_price
            
            if position.side.lower() == 'buy':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # sell/short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check profit target
            if pnl_pct >= TradingParams.PROFIT_TARGET:
                return 'profit_target'
            
            # Check stop loss
            if pnl_pct <= -TradingParams.STOP_LOSS:
                return 'stop_loss'
            
            # Check time-based exit (market close)
            # This would need market hours logic
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {position.ticker}: {e}")
            return None
    
    async def place_order(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal"""
        try:
            # Create contract (simplified - would need market config)
            if signal.market == 'US':
                contract = Stock(signal.ticker, 'SMART', 'USD')
            else:
                contract = Stock(signal.ticker, 'SEHK', 'HKD')
            
            quantity = 100  # Simplified position sizing
            order = MarketOrder(signal.signal_type.upper(), quantity)
            
            trade = self.ib_client.placeOrder(contract, order)
            
            self.log_trade(signal, quantity, 'ORDER_PLACED')
            
            # Add to positions
            position_key = f"{signal.ticker}_{signal.market}"
            self.open_positions[position_key] = Position(
                ticker=signal.ticker,
                market=signal.market,
                side=signal.signal_type,
                quantity=quantity,
                entry_price=signal.entry_price,
                entry_time=signal.timestamp
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Order failed for {signal.ticker}: {e}")
            return False
    
    def log_trade(self, signal: TradingSignal, quantity: int, action: str):
        """Comprehensive trade logging"""
        logger.info(f"TRADE LOG - {action}")
        logger.info(f"  Ticker: {signal.ticker} ({signal.market})")
        logger.info(f"  Signal: {signal.signal_type.upper()}")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Price: ${signal.entry_price:.2f}")
        logger.info(f"  Movement: {signal.price_movement:.2%}")
        logger.info(f"  Sentiment: {signal.sentiment}")
        logger.info(f"  Confidence: {signal.confidence:.1f}%")
        logger.info(f"  Timestamp: {signal.timestamp}")

# =============================================================================
# MARKET STATUS & ORCHESTRATION
# =============================================================================

class MarketStatusManager:
    @staticmethod
    def get_market_status(markets_config: Dict[str, MarketConfig]) -> Dict[str, Dict]:
        status = {}
        for market_name, config in markets_config.items():
            tz = pytz.timezone(config.timezone)
            now = dt.datetime.now(tz)
            open_time = dt.datetime.strptime(config.open_time, '%H:%M').time()
            close_time = dt.datetime.strptime(config.close_time, '%H:%M').time()
            current_time = now.time()
            
            is_open = (now.weekday() in config.trading_days and 
                      open_time <= current_time <= close_time)
            
            status[market_name] = {
                'is_open': is_open,
                'should_trade': is_open,
                'local_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'ticker_count': len(config.tickers)
            }
        
        return status

class TradingBot:
    def __init__(self):
        self.ib_client = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.us_news_provider = USNewsProvider(self.sentiment_analyzer)
        self.hk_news_provider = HKNewsProvider(self.sentiment_analyzer)
        self.market_data_provider = None
        self.signal_evaluator = None
        self.position_manager = None
        self.markets_config = MarketConfigManager.get_market_configs()
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing trading bot...")
        
        # Initialize sentiment analyzer
        if not self.sentiment_analyzer.initialize():
            return False
        
        # Connect to Interactive Brokers
        if not await self.connect_to_ib():
            return False
        
        # Initialize other components
        self.market_data_provider = MarketDataProvider(self.ib_client)
        self.signal_evaluator = SignalEvaluator(
            self.sentiment_analyzer, 
            self.us_news_provider, 
            self.hk_news_provider
        )
        self.position_manager = PositionManager(self.ib_client)
        
        logger.info("Trading bot initialized successfully")
        return True
    
    async def connect_to_ib(self) -> bool:
        """Connect to Interactive Brokers"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.ib_client = IB()
                try:
                    await self.ib_client.connectAsync('127.0.0.1', 7497, clientId=1)
                    logger.info("Connected to IB TWS")
                    break
                except:
                    await self.ib_client.connectAsync('127.0.0.1', 4002, clientId=1)
                    logger.info("Connected to IB Gateway")
                    break
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    return False
        
        try:
            account = self.ib_client.managedAccounts()[0]
            logger.info(f"IB Account: {account}")
            return True
        except Exception as e:
            logger.error(f"Account access failed: {e}")
            return False
    
    async def scan_market(self, market_name: str, should_trade: bool) -> int:
        """Scan a specific market for trading opportunities"""
        config = self.markets_config[market_name]
        tickers = config.tickers
        
        logger.info(f"Scanning {market_name} market ({len(tickers)} curated stocks)")
        
        signals_found = 0
        current_positions = len([p for p in self.position_manager.open_positions.values() 
                               if p.market == market_name])
        
        if current_positions >= TradingParams.MAX_POSITIONS_PER_MARKET:
            logger.info(f"Max positions reached for {market_name}")
            return 0
        
        for ticker in tickers:
            try:
                # Get market data
                price_data = await self.market_data_provider.get_price_data(
                    ticker, market_name, self.markets_config
                )
                if price_data is None:
                    continue
                
                movement, open_price, current_price = price_data
                
                # Evaluate entry signal
                signal = self.signal_evaluator.evaluate_entry_signal(
                    ticker, market_name, movement, current_price
                )
                
                if signal:
                    # Get sentiment analysis
                    sentiment = await self.get_sentiment_for_market(
                        ticker, market_name
                    )
                    
                    signal.sentiment = sentiment
                    
                    # Check if sentiment aligns with price movement
                    should_execute = False
                    if signal.signal_type == 'buy' and sentiment == 'positive':
                        should_execute = True
                    elif signal.signal_type == 'sell' and sentiment == 'negative':
                        should_execute = True
                    
                    if should_execute and should_trade:
                        if await self.position_manager.place_order(signal):
                            signals_found += 1
                            current_positions += 1
                            if current_positions >= TradingParams.MAX_POSITIONS_PER_MARKET:
                                break
                    else:
                        logger.info(f"HOLD {ticker} | Move: {movement:.2%} | Sentiment: {sentiment}")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        logger.info(f"{market_name} scan complete - {signals_found} signals found")
        return signals_found
    
    async def get_sentiment_for_market(self, ticker: str, market: str) -> str:
        """Get sentiment analysis for specific market"""
        try:
            if market == 'US':
                return await self.get_us_sentiment(ticker)
            elif market == 'HKEX':
                return await self.get_hk_sentiment(ticker)
            else:
                return 'neutral'
        except Exception as e:
            logger.error(f"Sentiment analysis error for {ticker}: {e}")
            return 'neutral'
    
    async def get_us_sentiment(self, ticker: str) -> str:
        """Enhanced US sentiment with multiple sources"""
        cache_key = f"{ticker}_{int(dt.datetime.now().timestamp() / 300)}"
        
        if cache_key in news_cache:
            return news_cache[cache_key]
        
        # Get company name for better search
        us_companies = {
            'NVDA': 'NVIDIA', 'AMD': 'Advanced Micro Devices', 'AAPL': 'Apple',
            'MSFT': 'Microsoft', 'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Meta',
            'TSLA': 'Tesla', 'JNJ': 'Johnson Johnson', 'PFE': 'Pfizer'
            # Add more as needed
        }
        company_name = us_companies.get(ticker, ticker)
        
        all_articles = []
        current_time = dt.datetime.now()
        
        # Try multiple time windows
        for minutes_back in [15, 30, 60]:
            try:
                # Fetch from all US sources
                tasks = [
                    self.us_news_provider.get_finnhub_news(ticker, minutes_back),
                    self.us_news_provider.get_rss_news(ticker, company_name, minutes_back),
                    self.us_news_provider.get_enhanced_newsapi(ticker, company_name, minutes_back)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        all_articles.extend(result)
                
                all_articles = NewsDataProvider.remove_duplicate_articles(all_articles)
                
                if len(all_articles) >= 3:
                    break
                elif minutes_back < 60:
                    logger.info(f"Expanding US time window to {minutes_back * 2} minutes for {ticker}")
                    all_articles = []
                    
            except Exception as e:
                logger.error(f"Error in US sentiment analysis: {e}")
                continue
        
        if len(all_articles) < 3:
            logger.info(f"Insufficient US news coverage for {ticker} ({len(all_articles)} articles)")
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        # Analyze sentiment with weighting
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in all_articles:
            sentiment = self.sentiment_analyzer.analyze_sentiment_finbert(article.title)
            recency_weight = NewsDataProvider.calculate_recency_weight(article.published, current_time)
            reliability_weight = NewsDataProvider.get_source_reliability_weight(article.source)
            total_weight = recency_weight * reliability_weight
            
            sentiment_scores[sentiment] += total_weight
        
        # Calculate final sentiment
        total_weight = sum(sentiment_scores.values())
        if total_weight == 0:
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        winner = max(sentiment_scores, key=sentiment_scores.get)
        confidence = (sentiment_scores[winner] / total_weight) * 100
        
        final_sentiment = winner if confidence >= 70 or winner == 'neutral' else 'neutral'
        
        logger.info(f"US Sentiment for {ticker}: {final_sentiment} (confidence: {confidence:.1f}%)")
        news_cache[cache_key] = final_sentiment
        return final_sentiment
    
    async def get_hk_sentiment(self, ticker: str) -> str:
        """Enhanced HK sentiment with Chinese sources"""
        cache_key = f"hk_{ticker}_{int(dt.datetime.now().timestamp() / 300)}"
        
        if cache_key in news_cache:
            return news_cache[cache_key]
        
        company_names = self.hk_news_provider.get_hk_company_names(ticker)
        all_articles = []
        current_time = dt.datetime.now()
        
        # Try multiple time windows
        for minutes_back in [15, 30, 60]:
            try:
                # Fetch from HK sources
                english_articles = await self.hk_news_provider.get_english_rss_news(
                    ticker, company_names, minutes_back
                )
                all_articles.extend(english_articles)
                
                # Remove duplicates
                all_articles = NewsDataProvider.remove_duplicate_articles(all_articles)
                
                if len(all_articles) >= 2:  # Lower threshold for HK
                    break
                elif minutes_back < 60:
                    logger.info(f"Expanding HK time window to {minutes_back * 2} minutes for {ticker}")
                    all_articles = []
                    
            except Exception as e:
                logger.error(f"Error in HK sentiment analysis: {e}")
                continue
        
        if len(all_articles) < 2:
            logger.info(f"Insufficient HK news coverage for {ticker} ({len(all_articles)} articles)")
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        # Analyze sentiment
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in all_articles:
            # Translate if needed (simplified for now)
            title = article.title
            if article.language == 'zh':
                title = await self.sentiment_analyzer.translate_chinese_to_english(title)
            
            sentiment = self.sentiment_analyzer.analyze_sentiment_finbert(title)
            recency_weight = NewsDataProvider.calculate_recency_weight(article.published, current_time)
            reliability_weight = NewsDataProvider.get_source_reliability_weight(article.source)
            total_weight = recency_weight * reliability_weight
            
            sentiment_scores[sentiment] += total_weight
        
        # Calculate final sentiment
        total_weight = sum(sentiment_scores.values())
        if total_weight == 0:
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        winner = max(sentiment_scores, key=sentiment_scores.get)
        confidence = (sentiment_scores[winner] / total_weight) * 100
        
        final_sentiment = winner if confidence >= 60 or winner == 'neutral' else 'neutral'
        
        logger.info(f"HK Sentiment for {ticker}: {final_sentiment} (confidence: {confidence:.1f}%)")
        news_cache[cache_key] = final_sentiment
        return final_sentiment
    
    def display_status(self) -> Dict[str, Dict]:
        """Display comprehensive system status"""
        status = MarketStatusManager.get_market_status(self.markets_config)
        
        logger.info("\n" + "="*50)
        logger.info("GLOBAL MARKET STATUS")
        logger.info("="*50)
        
        active_markets = []
        for market, info in status.items():
            status_icon = "OPEN" if info['is_open'] else "CLOSED"
            logger.info(f"{market}: {info['local_time']} | {status_icon} | {info['ticker_count']} stocks")
            if info['is_open']:
                active_markets.append(market)
        
        if self.position_manager and len(self.position_manager.open_positions) > 0:
            logger.info(f"Open Positions: {len(self.position_manager.open_positions)}")
            for pos_key, pos in self.position_manager.open_positions.items():
                logger.info(f"  {pos.ticker} ({pos.market}): {pos.side.upper()} {pos.quantity}")
        
        return status
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting US-HKEX 24/7 Trading Bot with Modular Architecture")
        logger.info("Enhanced multi-source sentiment analysis with 15-minute windows")
        logger.info("="*70)
        
        if not await self.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                
                market_status = self.display_status()
                
                trading_markets = [market for market, status in market_status.items() 
                                 if status['should_trade']]
                
                if trading_markets:
                    logger.info(f"\nACTIVE TRADING - Scan #{scan_count}")
                    total_signals = 0
                    
                    for market in trading_markets:
                        status = market_status[market]
                        signals = await self.scan_market(market, status['should_trade'])
                        total_signals += signals
                    
                    logger.info(f"Total signals: {total_signals}")
                    await asyncio.sleep(TradingParams.SCAN_INTERVAL)
                else:
                    logger.info("All markets closed - sleeping")
                    await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            if self.ib_client:
                self.ib_client.disconnect()
                logger.info("Disconnected from IB")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main entry point"""
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "nest-asyncio"])
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error: {e}")