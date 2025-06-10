from models import NewsArticle
from utils import remove_duplicate_articles
from config import NEWS_API_KEY, FINNHUB_API_KEY
from typing import List, Dict
import datetime as dt
import requests
import asyncio
import logging

try:
    import feedparser
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "feedparser"])
    import feedparser

logger = logging.getLogger(__name__)

class NewsDataProvider:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    @staticmethod
    def calculate_recency_weight(published_time: dt.datetime, current_time: dt.datetime) -> float:
        minutes_ago = (current_time - published_time).total_seconds() / 60
        if minutes_ago <= 5: return 3.0
        elif minutes_ago <= 10: return 2.0
        elif minutes_ago <= 15: return 1.0
        else: return 0.5

    @staticmethod
    def get_source_reliability_weight(source_name: str) -> float:
        weights = {
            'Reuters Business': 2.0, 'Bloomberg': 2.0, 'SCMP Finance': 1.8,
            'Finnhub': 1.8, 'Reuters Asia': 1.7, 'CNBC': 1.5,
            'MarketWatch': 1.3, 'Yahoo Finance': 1.2, 'Seeking Alpha': 1.1,
            'Google News': 1.2, 'NewsAPI': 1.0
        }
        return weights.get(source_name, 1.0)

class USNewsProvider(NewsDataProvider):
    def __init__(self, sentiment_analyzer):
        super().__init__(sentiment_analyzer)
        self.us_companies = self._get_comprehensive_company_mapping()
    
    def _get_comprehensive_company_mapping(self) -> Dict[str, str]:
        """Comprehensive company name mapping for all curated US stocks"""
        return {
            # TECH_AI_CHIPS
            'NVDA': 'NVIDIA Corporation', 'AMD': 'Advanced Micro Devices', 
            'SMCI': 'Super Micro Computer', 'ARM': 'Arm Holdings', 
            'AVGO': 'Broadcom', 'INTC': 'Intel Corporation', 
            'MU': 'Micron Technology', 'QCOM': 'Qualcomm', 
            'TSM': 'Taiwan Semiconductor', 'AAPL': 'Apple',
            'MSFT': 'Microsoft', 'GOOGL': 'Google Alphabet', 
            'AMZN': 'Amazon', 'META': 'Meta Facebook',
            'CRM': 'Salesforce', 'ORCL': 'Oracle Corporation',
            'IBM': 'International Business Machines', 'SNOW': 'Snowflake',
            'PLTR': 'Palantir Technologies', 'DDOG': 'Datadog',
            'ZS': 'Zscaler', 'NET': 'Cloudflare', 'MDB': 'MongoDB',
            'OKTA': 'Okta', 'PANW': 'Palo Alto Networks',
            'CRWD': 'CrowdStrike', 'ANET': 'Arista Networks',
            'ADBE': 'Adobe', 'SHOP': 'Shopify', 'U': 'Unity Software',
            'TWLO': 'Twilio', 'HUBS': 'HubSpot', 'DOCN': 'DigitalOcean',
            'ASML': 'ASML Holding', 'TTD': 'Trade Desk',
            
            # HEALTHCARE_BIOTECH
            'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer', 'MRK': 'Merck',
            'LLY': 'Eli Lilly', 'ABBV': 'AbbVie', 'BMY': 'Bristol Myers Squibb',
            'GILD': 'Gilead Sciences', 'AMGN': 'Amgen', 'REGN': 'Regeneron',
            'BIIB': 'Biogen', 'VRTX': 'Vertex Pharmaceuticals', 'AZN': 'AstraZeneca',
            'CVS': 'CVS Health', 'CI': 'Cigna', 'UNH': 'UnitedHealth Group',
            'HUM': 'Humana', 'TMO': 'Thermo Fisher Scientific', 'SYK': 'Stryker',
            'ISRG': 'Intuitive Surgical', 'ZBH': 'Zimmer Biomet',
            
            # RETAIL_CONSUMER
            'WMT': 'Walmart', 'COST': 'Costco', 'TGT': 'Target',
            'HD': 'Home Depot', 'LOW': 'Lowe\'s', 'MCD': 'McDonald\'s',
            'SBUX': 'Starbucks', 'TPR': 'Tapestry', 'ULTA': 'Ulta Beauty',
            'NKE': 'Nike', 'LULU': 'Lululemon', 'DG': 'Dollar General',
            'DLTR': 'Dollar Tree', 'BBY': 'Best Buy', 'KR': 'Kroger',
            'TJX': 'TJX Companies', 'ROST': 'Ross Stores', 'ETSY': 'Etsy',
            'EBAY': 'eBay',
            
            # FINTECH_MEGACAP
            'PYPL': 'PayPal', 'SQ': 'Block Square', 'COIN': 'Coinbase',
            'HOOD': 'Robinhood', 'V': 'Visa', 'MA': 'Mastercard',
            'AXP': 'American Express', 'BKNG': 'Booking Holdings',
            'UBER': 'Uber Technologies', 'LYFT': 'Lyft',
            
            # NEWSMAKER_OUTLIERS
            'TSLA': 'Tesla', 'RIVN': 'Rivian', 'LCID': 'Lucid Motors',
            'XPEV': 'XPeng', 'NIO': 'NIO', 'NKLA': 'Nikola',
            'GME': 'GameStop', 'AMC': 'AMC Entertainment', 'PARA': 'Paramount',
            'DIS': 'Walt Disney', 'NFLX': 'Netflix', 'DWAC': 'Digital World',
            'SNAP': 'Snapchat', 'BZFD': 'BuzzFeed', 'ROKU': 'Roku'
        }

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
                        news_article = NewsArticle(
                            title=article.get('headline', ''),
                            published=article_time,
                            source='Finnhub'
                        )
                        news_article.description = article.get('summary', '')
                        recent_articles.append(news_article)
                
                return recent_articles
            return []
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return []

    async def get_enhanced_newsapi(self, ticker: str, company_name: str, minutes_back: int = 15) -> List[NewsArticle]:
        try:
            now = dt.datetime.now()
            from_time = now - dt.timedelta(minutes=minutes_back)
            
            search_queries = [
                f'{ticker} OR "{company_name}"',
                f'"{company_name}" stock',
                f'{ticker} earnings OR revenue OR guidance'
            ]
            
            all_articles = []
            
            for query in search_queries:
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': query,
                        'from': from_time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'to': now.strftime('%Y-%m-%dT%H:%M:%S'),
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 15,
                        'apiKey': NEWS_API_KEY
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code != 200:
                        continue
                        
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        title = article.get('title', '')
                        description = article.get('description', '') or ''
                        published_str = article.get('publishedAt', '')
                        
                        content_to_check = f"{title} {description}".lower()
                        if (ticker.lower() in content_to_check or 
                            company_name.lower() in content_to_check):
                            
                            try:
                                published_time = dt.datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                                published_time = published_time.replace(tzinfo=None)
                                
                                if (now - published_time).total_seconds() <= minutes_back * 60:
                                    news_article = NewsArticle(
                                        title=title,
                                        published=published_time,
                                        source='NewsAPI'
                                    )
                                    news_article.description = description
                                    all_articles.append(news_article)
                            except:
                                continue
                    
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"NewsAPI query error: {e}")
                    continue
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Enhanced NewsAPI error: {e}")
            return []

    async def get_rss_news(self, ticker: str, company_name: str, minutes_back: int = 15) -> List[NewsArticle]:
        rss_feeds = {
            'Yahoo Finance': f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}',
            'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'Reuters Business': 'https://feeds.reuters.com/reuters/businessNews',
            'CNBC': 'https://feeds.feedburner.com/cnbc/business',
            'Seeking Alpha': 'https://seekingalpha.com/market_currents.xml'
        }
        
        all_articles = []
        now = dt.datetime.now()
        
        for source_name, feed_url in rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:12]:
                    title = entry.get('title', '')
                    description = entry.get('description', '') or entry.get('summary', '') or ''
                    
                    content_to_check = f"{title} {description}".lower()
                    if (ticker.lower() in content_to_check or 
                        company_name.lower() in content_to_check):
                        
                        published_time = now
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published_time = dt.datetime(*entry.published_parsed[:6])
                            except:
                                pass
                        
                        if (now - published_time).total_seconds() <= minutes_back * 60:
                            news_article = NewsArticle(
                                title=title,
                                published=published_time,
                                source=source_name
                            )
                            news_article.description = description
                            all_articles.append(news_article)
                            
            except Exception as e:
                logger.error(f"RSS feed error for {source_name}: {e}")
                continue
        
        return all_articles

    async def get_google_news_fallback(self, ticker: str, company_name: str, minutes_back: int = 60) -> List[NewsArticle]:
        try:
            search_queries = [
                f"{ticker} stock",
                f"{company_name}",
                f"{ticker} earnings",
                f"{company_name} news"
            ]
            
            all_articles = []
            now = dt.datetime.now()
            
            for query in search_queries:
                try:
                    encoded_query = query.replace(' ', '+')
                    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                    
                    feed = feedparser.parse(url)
                    
                    for entry in feed.entries[:8]:
                        title = entry.get('title', '')
                        
                        if (ticker.lower() in title.lower() or 
                            any(word.lower() in title.lower() for word in company_name.split()[:2])):
                            
                            published_time = now
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                try:
                                    published_time = dt.datetime(*entry.published_parsed[:6])
                                except:
                                    pass
                            
                            if (now - published_time).total_seconds() <= minutes_back * 60:
                                news_article = NewsArticle(
                                    title=title,
                                    published=published_time,
                                    source='Google News',
                                    weight=1.2
                                )
                                all_articles.append(news_article)
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Google News query error for '{query}': {e}")
                    continue
            
            return remove_duplicate_articles(all_articles)
            
        except Exception as e:
            logger.error(f"Google News fallback error: {e}")
            return []