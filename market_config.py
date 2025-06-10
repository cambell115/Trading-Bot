
from models import MarketConfig
from typing import Dict, List

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
        return list(dict.fromkeys(all_stocks))

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
                trading_days=[0,1,2,3,4],
                tickers=StockListManager.get_curated_us_stocks(),
                exchange='SMART',
                currency='USD'
            ),
            'HKEX': MarketConfig(
                timezone='Asia/Hong_Kong',
                open_time='09:30',
                close_time='16:00',
                trading_days=[0,1,2,3,4],
                tickers=StockListManager.get_curated_hkex_stocks(),
                exchange='SEHK',
                currency='HKD'
            )
        }