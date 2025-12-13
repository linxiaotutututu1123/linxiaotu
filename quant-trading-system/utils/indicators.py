"""Technical indicators calculation module.

Implements common technical indicators for quantitative trading following PEP 8.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


class TechnicalIndicators:
    """Technical indicators calculator for trading strategies."""
    
    @staticmethod
    def rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI values as pandas Series
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        prices: Union[pd.Series, np.ndarray],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        prices: Union[pd.Series, np.ndarray],
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period (default 20)
            num_std: Number of standard deviations (default 2.0)
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def sma(prices: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA).
        
        Args:
            prices: Price series
            period: Moving average period
            
        Returns:
            SMA values as pandas Series
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA).
        
        Args:
            prices: Price series
            period: Moving average period
            
        Returns:
            EMA values as pandas Series
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def atr(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)
            
        Returns:
            ATR values as pandas Series
        """
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_volatility(
        prices: Union[pd.Series, np.ndarray],
        period: int = 20
    ) -> pd.Series:
        """Calculate price volatility (standard deviation of returns).
        
        Args:
            prices: Price series
            period: Period for volatility calculation
            
        Returns:
            Volatility values as pandas Series
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        returns = prices.pct_change()
        volatility = returns.rolling(window=period).std()
        
        return volatility
