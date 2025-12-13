"""Factor-based trading strategy.

Implements trading strategy based on multi-factor model signals.
"""

import pandas as pd
from typing import Dict
from .base_strategy import BaseStrategy
from models.multi_factor import MultiFactorModel
from utils.logger import get_logger


logger = get_logger(__name__)


class FactorStrategy(BaseStrategy):
    """Trading strategy based on multi-factor model."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize FactorStrategy.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("FactorStrategy")
        self.model = MultiFactorModel(config_path)
        self.previous_signals: Dict[str, float] = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generate trading signals based on multi-factor model.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping symbol to signal (1=buy, -1=sell, 0=hold)
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 50:  # Need sufficient data
                signals[symbol] = 0
                continue
            
            # Generate signal from multi-factor model
            signal_score = self.model.generate_signal(df, data)
            
            # Convert signal score to action
            action = self.model.get_trade_action(signal_score)
            
            # Check current position
            current_position = self.get_position(symbol)
            
            # Adjust signal based on current position
            if action == 1 and current_position >= 0:
                # Buy signal and not short
                signals[symbol] = 1
            elif action == -1 and current_position <= 0:
                # Sell signal and not long
                signals[symbol] = -1
            elif action == 1 and current_position < 0:
                # Buy to cover short
                signals[symbol] = 1
            elif action == -1 and current_position > 0:
                # Sell to close long
                signals[symbol] = -1
            else:
                signals[symbol] = 0
            
            # Store signal score for tracking
            self.previous_signals[symbol] = signal_score
            
            if signals[symbol] != 0:
                logger.debug(f"Factor signal for {symbol}: {signals[symbol]} (score: {signal_score:.2f})")
        
        return signals
