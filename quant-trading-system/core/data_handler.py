"""Data handler module for fetching and storing market data.

Handles data retrieval from various sources and caching for the trading system.
"""

import os
import pandas as pd
import yaml
from typing import Optional, List, Dict
from datetime import datetime
from utils.logger import get_logger
from utils.decorators import timing_decorator, exception_handler


logger = get_logger(__name__)


class DataHandler:
    """Handles market data retrieval and storage for backtesting and live trading."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize DataHandler with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_source = self.config.get('data', {}).get('source', 'tqsdk')
        self.storage_type = self.config.get('data', {}).get('storage', {}).get('type', 'csv')
        self.storage_path = self.config.get('data', {}).get('storage', {}).get('path', './data/market_data')
        
        os.makedirs(self.storage_path, exist_ok=True)
        logger.info(f"DataHandler initialized with source: {self.data_source}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    @timing_decorator
    @exception_handler
    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = '1m'
    ) -> pd.DataFrame:
        """Fetch market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RB', 'IF')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('1m', '5m', '1h', 'd')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Check cache first
        cached_data = self._load_from_cache(symbol, start_date, end_date, frequency)
        if cached_data is not None:
            logger.info(f"Loaded {len(cached_data)} rows from cache")
            return cached_data
        
        # Fetch from data source
        if self.data_source == 'tqsdk':
            data = self._fetch_from_tqsdk(symbol, start_date, end_date, frequency)
        else:
            logger.warning(f"Data source {self.data_source} not implemented, returning sample data")
            data = self._generate_sample_data(symbol, start_date, end_date, frequency)
        
        # Save to cache
        self._save_to_cache(data, symbol, start_date, end_date, frequency)
        
        return data
    
    def _fetch_from_tqsdk(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> pd.DataFrame:
        """Fetch data from TqSdk.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            DataFrame with OHLCV data
        """
        # TODO: Implement actual TqSdk integration
        logger.info("TqSdk integration pending, using sample data")
        return self._generate_sample_data(symbol, start_date, end_date, frequency)
    
    def _generate_sample_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> pd.DataFrame:
        """Generate sample data for testing.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            DataFrame with sample OHLCV data
        """
        import numpy as np
        
        # Convert frequency to pandas offset
        freq_map = {'1m': '1min', '5m': '5min', '1h': '1H', 'd': '1D'}
        pd_freq = freq_map.get(frequency, '1min')
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=pd_freq)
        
        # Generate random OHLCV data
        base_price = 3700 if symbol == 'RB' else 4000
        np.random.seed(42)
        
        close_prices = base_price + np.cumsum(np.random.randn(len(date_range)) * 10)
        
        data = pd.DataFrame({
            'datetime': date_range,
            'open': close_prices + np.random.randn(len(date_range)) * 5,
            'high': close_prices + abs(np.random.randn(len(date_range))) * 10,
            'low': close_prices - abs(np.random.randn(len(date_range))) * 10,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, len(date_range))
        })
        
        data.set_index('datetime', inplace=True)
        return data
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            Cached DataFrame or None if not found
        """
        cache_file = self._get_cache_filename(symbol, start_date, end_date, frequency)
        
        if os.path.exists(cache_file):
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(
        self,
        data: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> None:
        """Save data to cache.
        
        Args:
            data: DataFrame to save
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
        """
        cache_file = self._get_cache_filename(symbol, start_date, end_date, frequency)
        
        try:
            data.to_csv(cache_file)
            logger.info(f"Saved {len(data)} rows to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _get_cache_filename(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> str:
        """Generate cache filename.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            Cache file path
        """
        filename = f"{symbol}_{start_date}_{end_date}_{frequency}.csv"
        return os.path.join(self.storage_path, filename)
    
    def get_symbols_list(self) -> List[str]:
        """Get list of trading symbols from configuration.
        
        Returns:
            List of symbol strings
        """
        return self.config.get('trading', {}).get('symbols', ['RB', 'IF'])
