"""Multi-factor model for trading signal generation.

Combines multiple factors including technical, fund flow, sentiment, and correlation factors.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Optional
from utils.logger import get_logger
from utils.indicators import TechnicalIndicators


logger = get_logger(__name__)


class MultiFactorModel:
    """Multi-factor model for generating trading signals."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize MultiFactorModel with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.factor_config = self.config.get('multi_factor', {})
        
        # Factor weights
        self.weights = self.factor_config.get('weights', {
            'technical': 0.30,
            'fund_flow': 0.25,
            'sentiment': 0.25,
            'correlation': 0.15,
            'anomaly': 0.05
        })
        
        # Technical parameters
        self.tech_params = self.factor_config.get('technical', {})
        self.rsi_period = self.tech_params.get('rsi_period', 14)
        self.macd_fast = self.tech_params.get('macd_fast', 12)
        self.macd_slow = self.tech_params.get('macd_slow', 26)
        self.macd_signal = self.tech_params.get('macd_signal', 9)
        self.bb_period = self.tech_params.get('bollinger_period', 20)
        self.bb_std = self.tech_params.get('bollinger_std', 2)
        
        # Signal thresholds
        self.thresholds = self.factor_config.get('signal_thresholds', {})
        self.strong_buy = self.thresholds.get('strong_buy', 0.75)
        self.buy = self.thresholds.get('buy', 0.60)
        self.strong_sell = self.thresholds.get('strong_sell', 0.25)
        self.sell = self.thresholds.get('sell', 0.40)
        
        self.indicators = TechnicalIndicators()
        
        logger.info("MultiFactorModel initialized")
    
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
    
    def calculate_technical_factor(self, data: pd.DataFrame) -> float:
        """Calculate technical factor score.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Technical factor score (0 to 1)
        """
        if len(data) < max(self.rsi_period, self.macd_slow, self.bb_period):
            return 0.5
        
        prices = data['close']
        
        # RSI
        rsi = self.indicators.rsi(prices, self.rsi_period)
        rsi_score = (100 - rsi.iloc[-1]) / 100 if not pd.isna(rsi.iloc[-1]) else 0.5
        
        # MACD
        macd_line, signal_line, _ = self.indicators.macd(
            prices, self.macd_fast, self.macd_slow, self.macd_signal
        )
        macd_score = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.0
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(
            prices, self.bb_period, self.bb_std
        )
        current_price = prices.iloc[-1]
        
        if not pd.isna(upper.iloc[-1]) and not pd.isna(lower.iloc[-1]):
            bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            bb_score = 1 - bb_position  # Lower in band = higher score (oversold)
        else:
            bb_score = 0.5
        
        # Combine technical indicators
        tech_score = (rsi_score * 0.4 + macd_score * 0.3 + bb_score * 0.3)
        
        return np.clip(tech_score, 0, 1)
    
    def calculate_fund_flow_factor(self, data: pd.DataFrame) -> float:
        """Calculate fund flow factor score.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Fund flow factor score (0 to 1)
        """
        if len(data) < 20:
            return 0.5
        
        # Volume trend
        recent_volume = data['volume'].tail(5).mean()
        past_volume = data['volume'].tail(20).head(15).mean()
        
        if past_volume > 0:
            volume_ratio = recent_volume / past_volume
            volume_score = np.clip((volume_ratio - 0.5) / 1.5, 0, 1)
        else:
            volume_score = 0.5
        
        # Price-volume correlation
        price_change = data['close'].pct_change().tail(10)
        volume_change = data['volume'].pct_change().tail(10)
        
        if len(price_change) > 0 and len(volume_change) > 0:
            correlation = price_change.corr(volume_change)
            corr_score = (correlation + 1) / 2  # Convert from [-1,1] to [0,1]
        else:
            corr_score = 0.5
        
        fund_score = (volume_score * 0.6 + corr_score * 0.4)
        
        return np.clip(fund_score, 0, 1)
    
    def calculate_sentiment_factor(self, data: pd.DataFrame) -> float:
        """Calculate market sentiment factor score.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Sentiment factor score (0 to 1)
        """
        if len(data) < 20:
            return 0.5
        
        # Momentum
        returns = data['close'].pct_change().tail(10)
        momentum = returns.mean()
        momentum_score = np.clip((momentum * 100 + 2) / 4, 0, 1)
        
        # Volatility (higher volatility = more caution)
        volatility = returns.std()
        volatility_score = np.clip(1 - (volatility * 50), 0, 1)
        
        sentiment_score = (momentum_score * 0.7 + volatility_score * 0.3)
        
        return np.clip(sentiment_score, 0, 1)
    
    def calculate_correlation_factor(
        self,
        data: pd.DataFrame,
        all_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> float:
        """Calculate cross-commodity correlation factor.
        
        Args:
            data: DataFrame with OHLCV data for target symbol
            all_data: Dictionary of all symbols' data
            
        Returns:
            Correlation factor score (0 to 1)
        """
        # Simplified correlation factor (would require multiple symbols)
        if all_data is None or len(all_data) <= 1:
            return 0.5
        
        # TODO: Implement actual cross-commodity correlation analysis
        return 0.5
    
    def calculate_anomaly_factor(self, data: pd.DataFrame) -> float:
        """Calculate anomaly detection factor.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Anomaly factor score (0 to 1, 0 = anomaly detected)
        """
        if len(data) < 50:
            return 0.5
        
        # Check for unusual price movements
        returns = data['close'].pct_change().tail(50)
        recent_return = returns.iloc[-1]
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return > 0:
            z_score = abs((recent_return - mean_return) / std_return)
            # High z-score indicates anomaly
            anomaly_score = np.clip(1 - (z_score / 3), 0, 1)
        else:
            anomaly_score = 0.5
        
        return anomaly_score
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        all_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> float:
        """Generate trading signal based on all factors.
        
        Args:
            data: DataFrame with OHLCV data for symbol
            all_data: Dictionary of all symbols' data
            
        Returns:
            Signal score (0 to 1, >0.6 = buy, <0.4 = sell)
        """
        # Calculate individual factor scores
        tech_score = self.calculate_technical_factor(data)
        fund_score = self.calculate_fund_flow_factor(data)
        sentiment_score = self.calculate_sentiment_factor(data)
        correlation_score = self.calculate_correlation_factor(data, all_data)
        anomaly_score = self.calculate_anomaly_factor(data)
        
        # Weighted combination
        signal = (
            tech_score * self.weights['technical'] +
            fund_score * self.weights['fund_flow'] +
            sentiment_score * self.weights['sentiment'] +
            correlation_score * self.weights['correlation'] +
            anomaly_score * self.weights['anomaly']
        )
        
        logger.debug(f"Factor scores - Tech: {tech_score:.2f}, Fund: {fund_score:.2f}, "
                    f"Sentiment: {sentiment_score:.2f}, Signal: {signal:.2f}")
        
        return signal
    
    def get_trade_action(self, signal: float) -> int:
        """Convert signal score to trade action.
        
        Args:
            signal: Signal score (0 to 1)
            
        Returns:
            Trade action (1 = buy, -1 = sell, 0 = hold)
        """
        if signal >= self.buy:
            return 1  # Buy
        elif signal <= self.sell:
            return -1  # Sell
        else:
            return 0  # Hold
