"""Combined strategy integrating multiple approaches.

Combines multi-factor and grid trading strategies for robust performance.
"""

import pandas as pd
from typing import Dict
from .base_strategy import BaseStrategy
from .factor_strategy import FactorStrategy
from .grid_strategy import GridStrategy
from utils.logger import get_logger


logger = get_logger(__name__)


class CombinedStrategy(BaseStrategy):
    """Combined strategy using both factor and grid trading approaches."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize CombinedStrategy.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("CombinedStrategy")
        self.factor_strategy = FactorStrategy(config_path)
        self.grid_strategy = GridStrategy(config_path)
        
        # Weights for combining signals
        self.factor_weight = 0.6
        self.grid_weight = 0.4
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generate combined trading signals.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping symbol to signal (1=buy, -1=sell, 0=hold)
        """
        # Get signals from both strategies
        factor_signals = self.factor_strategy.generate_signals(data)
        grid_signals = self.grid_strategy.generate_signals(data)
        
        combined_signals = {}
        
        for symbol in data.keys():
            factor_sig = factor_signals.get(symbol, 0)
            grid_sig = grid_signals.get(symbol, 0)
            
            # Combine signals with weights
            combined_score = (
                factor_sig * self.factor_weight +
                grid_sig * self.grid_weight
            )
            
            # Convert to discrete signal
            if combined_score > 0.5:
                combined_signals[symbol] = 1
            elif combined_score < -0.5:
                combined_signals[symbol] = -1
            else:
                combined_signals[symbol] = 0
            
            # Log when signals agree strongly
            if factor_sig != 0 and factor_sig == grid_sig:
                logger.info(f"Strong consensus signal for {symbol}: {factor_sig}")
            
            if combined_signals[symbol] != 0:
                logger.debug(f"Combined signal for {symbol}: {combined_signals[symbol]} "
                           f"(factor: {factor_sig}, grid: {grid_sig})")
        
        return combined_signals
    
    def update_position(self, symbol: str, size: int) -> None:
        """Update positions across all strategies.
        
        Args:
            symbol: Trading symbol
            size: Change in position size
        """
        super().update_position(symbol, size)
        self.factor_strategy.update_position(symbol, size)
        self.grid_strategy.update_position(symbol, size)
    
    def reset_positions(self) -> None:
        """Reset positions for all strategies."""
        super().reset_positions()
        self.factor_strategy.reset_positions()
        self.grid_strategy.reset_positions()
    
    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information.
        
        Returns:
            Dictionary with strategy information
        """
        info = super().get_strategy_info()
        info['factor_strategy'] = self.factor_strategy.get_strategy_info()
        info['grid_strategy'] = self.grid_strategy.get_strategy_info()
        info['weights'] = {
            'factor': self.factor_weight,
            'grid': self.grid_weight
        }
        return info
