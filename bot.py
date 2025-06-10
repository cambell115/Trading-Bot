from config import TradingParams, NEWS_API_KEY, FINNHUB_API_KEY
from market_config import MarketConfigManager
from sentiment import SentimentAnalyzer
from news_providers import USNewsProvider, NewsDataProvider
from market_data import MarketDataProvider
from signal_evaluator import SignalEvaluator
from position_manager import PositionManager
from utils import remove_duplicate_articles
import datetime as dt
import logging
import pytz
from ib_insync import IB
import asyncio
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

news_cache = {}

class TradingBot:
    def __init__(self):
        self.ib_client = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.us_news_provider = USNewsProvider(self.sentiment_analyzer)
        self.hk_news_provider = NewsDataProvider(self.sentiment_analyzer)
        self.market_data_provider = None
        self.signal_evaluator = None
        self.position_manager = None
        self.markets_config = MarketConfigManager.get_market_configs()

    async def initialize(self):
        logger.info("Initializing trading bot...")
        if not self.sentiment_analyzer.initialize():
            return False
        if not await self.connect_to_ib():
            return False
        self.market_data_provider = MarketDataProvider(self.ib_client)
        self.signal_evaluator = SignalEvaluator(self.sentiment_analyzer, self.us_news_provider, self.hk_news_provider)
        self.position_manager = PositionManager(self.ib_client)
        logger.info("Trading bot initialized successfully")
        return True

    async def connect_to_ib(self):
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

    async def get_us_sentiment(self, ticker: str) -> str:
        """Enhanced US sentiment with comprehensive coverage and detailed logging"""
        cache_key = f"{ticker}_{int(dt.datetime.now().timestamp() / 300)}"
        
        if cache_key in news_cache:
            cached_result = news_cache[cache_key]
            logger.info(f"{ticker} sentiment (cached): {cached_result}")
            return cached_result
        
        # Get company name
        company_name = self.us_news_provider.us_companies.get(ticker, ticker)
        
        all_articles = []
        current_time = dt.datetime.now()
        articles_by_source = {}
        
        # Try multiple time windows with detailed logging
        for minutes_back in [15, 30, 60]:
            try:
                logger.info(f"Fetching {ticker} ({company_name}) news - {minutes_back}min window...")
                
                # Fetch from all sources
                tasks = [
                    self.us_news_provider.get_finnhub_news(ticker, minutes_back),
                    self.us_news_provider.get_rss_news(ticker, company_name, minutes_back),
                    self.us_news_provider.get_enhanced_newsapi(ticker, company_name, minutes_back)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results and track by source
                for i, result in enumerate(results):
                    if isinstance(result, list):
                        source_names = ['Finnhub', 'RSS Feeds', 'NewsAPI']
                        articles_by_source[source_names[i]] = len(result)
                        all_articles.extend(result)
                
                # Log source breakdown
                total_before_dedup = len(all_articles)
                all_articles = remove_duplicate_articles(all_articles)
                
                logger.info(f"{ticker} | {minutes_back}min | Sources: {articles_by_source} | "
                           f"Total: {total_before_dedup} -> {len(all_articles)} after dedup")
                
                # If we have sufficient articles, proceed
                if len(all_articles) >= 3:
                    break
                elif minutes_back < 60:
                    logger.info(f"Insufficient coverage ({len(all_articles)} articles), expanding time window...")
                    all_articles = []
                    articles_by_source = {}
            
            except Exception as e:
                logger.error(f"Error in US sentiment analysis: {e}")
                continue
        
        # Fallback to Google News if still insufficient
        if len(all_articles) < 3:
            logger.info(f"Triggering Google News fallback for {ticker}...")
            try:
                google_articles = await self.us_news_provider.get_google_news_fallback(
                    ticker, company_name, 60
                )
                all_articles.extend(google_articles)
                all_articles = remove_duplicate_articles(all_articles)
                articles_by_source['Google News Fallback'] = len(google_articles)
                
                logger.info(f"{ticker} | Google fallback added {len(google_articles)} articles | "
                           f"Total: {len(all_articles)}")
            except Exception as e:
                logger.error(f"Google News fallback failed: {e}")
        
        # Final check
        if len(all_articles) < 2:
            logger.warning(f"{ticker} | INSUFFICIENT NEWS | Only {len(all_articles)} articles found | "
                          f"Sources: {articles_by_source} | Returning neutral")
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        # Analyze sentiment with enhanced method
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        article_details = []
        
        for article in all_articles:
            # Use enhanced sentiment analysis (title + description)
            sentiment = self.sentiment_analyzer.analyze_article_sentiment(article)
            
            # Calculate weights
            recency_weight = NewsDataProvider.calculate_recency_weight(article.published, current_time)
            reliability_weight = NewsDataProvider.get_source_reliability_weight(article.source)
            total_weight = recency_weight * reliability_weight * getattr(article, 'weight', 1.0)
            
            sentiment_scores[sentiment] += total_weight
            
            # Track for detailed logging
            article_details.append({
                'source': article.source,
                'sentiment': sentiment,
                'weight': total_weight,
                'title_preview': article.title[:60] + "..." if len(article.title) > 60 else article.title
            })
        
        # Calculate final sentiment
        total_weight = sum(sentiment_scores.values())
        if total_weight == 0:
            logger.warning(f"{ticker} | Zero total weight in sentiment calculation")
            news_cache[cache_key] = 'neutral'
            return 'neutral'
        
        winner = max(sentiment_scores, key=sentiment_scores.get)
        confidence = (sentiment_scores[winner] / total_weight) * 100
        
        # Enhanced confidence threshold
        final_sentiment = winner if confidence >= 65 or winner == 'neutral' else 'neutral'
        
        # Comprehensive logging
        logger.info(f"{ticker} SENTIMENT ANALYSIS COMPLETE:")
        logger.info(f"  Articles analyzed: {len(all_articles)}")
        logger.info(f"  Sources: {articles_by_source}")
        logger.info(f"  Sentiment breakdown: P={sentiment_scores['positive']:.1f} "
                   f"N={sentiment_scores['negative']:.1f} Neu={sentiment_scores['neutral']:.1f}")
        logger.info(f"  Final: {final_sentiment.upper()} (confidence: {confidence:.1f}%)")
        
        # Log a few sample articles for debugging
        for detail in article_details[:3]:
            logger.info(f"    {detail['source']}: {detail['sentiment']} | {detail['title_preview']}")
        
        news_cache[cache_key] = final_sentiment
        return final_sentiment

    def get_market_status(self) -> Dict[str, Dict]:
        """Get current market status"""
        status = {}
        for market_name, config in self.markets_config.items():
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

    def display_status(self) -> Dict[str, Dict]:
        """Display comprehensive system status"""
        status = self.get_market_status()
        
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

    async def scan_market(self, market_name: str, should_trade: bool) -> int:
        """Scan a specific market for trading opportunities"""
        config = self.markets_config[market_name]
        tickers = config.tickers[:5]  # Limit to first 5 for testing
        
        logger.info(f"Scanning {market_name} market ({len(tickers)} stocks)")
        
        signals_found = 0
        current_positions = len([p for p in self.position_manager.open_positions.values() 
                               if p.market == market_name])
        
        if current_positions >= TradingParams.MAX_POSITIONS_PER_MARKET:
            logger.info(f"Max positions reached for {market_name}")
            return 0
        
        for ticker in tickers:
            try:
                # Get market data
                price_data = await self.market_data_provider.get_price_data(ticker, market_name)
                if price_data is None:
                    continue
                
                movement, open_price, current_price = price_data
                
                # Log price movement
                logger.info(f"{ticker} moved {movement:.2%} | Current: ${current_price:.2f}")
                
                # Check if movement exceeds trigger
                if abs(movement) >= TradingParams.PRICE_TRIGGER:
                    logger.info(f"PRICE TRIGGER: {ticker} moved {movement:.2%}")
                    
                    # Evaluate entry signal
                    signal = self.signal_evaluator.evaluate_entry_signal(
                        ticker, market_name, movement, current_price
                    )
                    
                    if signal:
                        # Get sentiment analysis for US market
                        if market_name == 'US':
                            sentiment = await self.get_us_sentiment(ticker)
                            signal.sentiment = sentiment
                            
                            # Check if sentiment aligns with price movement
                            should_execute = False
                            if signal.signal_type == 'buy' and sentiment == 'positive':
                                should_execute = True
                            elif signal.signal_type == 'sell' and sentiment == 'negative':
                                should_execute = True
                            
                            if should_execute and should_trade:
                                logger.info(f"SIGNAL CONFIRMED: {ticker} {signal.signal_type.upper()} | Sentiment: {sentiment}")
                                if await self.position_manager.place_order(signal):
                                    signals_found += 1
                                    current_positions += 1
                                    if current_positions >= TradingParams.MAX_POSITIONS_PER_MARKET:
                                        break
                            else:
                                logger.info(f"HOLD {ticker} | Move: {movement:.2%} | Sentiment: {sentiment} | No alignment")
                        else:
                            # For non-US markets, simplified logic for now
                            signal.sentiment = 'neutral'
                            logger.info(f"NON-US SIGNAL: {ticker} {signal.signal_type.upper()}")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        logger.info(f"{market_name} scan complete - {signals_found} signals found")
        return signals_found

    async def run(self):
        """Main trading loop"""
        logger.info("Starting Enhanced US-HKEX Trading Bot")
        logger.info("="*50)
        
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
                    await asyncio.sleep(30)  # Shorter sleep for testing

        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            if self.ib_client:
                self.ib_client.disconnect()
                logger.info("Disconnected from IB")