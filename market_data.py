from ib_insync import Stock
from typing import Optional, Tuple
from config import TradingParams
import asyncio
import logging

logger = logging.getLogger(__name__)

class MarketDataProvider:
    def __init__(self, ib_client):
        self.ib_client = ib_client
    
    def create_contract(self, ticker: str, market: str) -> Stock:
        if market == 'US':
            return Stock(ticker, 'SMART', 'USD')
        elif market == 'HKEX':
            return Stock(ticker, 'SEHK', 'HKD')
    
    async def get_price_data(self, ticker: str, market: str) -> Optional[Tuple[float, float, float]]:
        try:
            contract = self.create_contract(ticker, market)
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