"""Grid trading strategy.

Implements grid trading strategy for range-bound market conditions.
"""

import pandas as pd
from typing import Dict, Optional
from .base_strategy import BaseStrategy
from models.grid_trading import GridTradingModel
from utils.logger import get_logger


logger = get_logger(__name__)


class GridStrategy(BaseStrategy):
    """Trading strategy based on grid trading model."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize GridStrategy.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__("GridStrategy")
        self.model = GridTradingModel(config_path)
        self.previous_prices: Dict[str, float] = {}
        self.grids_initialized: Dict[str, bool] = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generate trading signals based on grid levels.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping symbol to signal (1=buy, -1=sell, 0=hold)
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 20:  # Need sufficient data for price analysis
                signals[symbol] = 0
                continue
            
            current_price = df['close'].iloc[-1]
            
            # Initialize grid if not already done
            if not self.grids_initialized.get(symbol, False):
                # Use recent average as base price
                base_price = df['close'].tail(20).mean()
                self.model.setup_grid(symbol, base_price=base_price)
                self.grids_initialized[symbol] = True
                logger.info(f"Grid initialized for {symbol} at base price {base_price:.2f}")
            
            # Get previous price for crossing detection
            previous_price = self.previous_prices.get(symbol)
            
            # Generate signal
            signal = self.model.generate_signal(symbol, current_price, previous_price)
            
            # Calculate position size if signal is non-zero
            if signal != 0:
                size = self.model.calculate_position_size(symbol, signal)
                signals[symbol] = signal
                logger.debug(f"Grid signal for {symbol}: {signal} (price: {current_price:.2f})")
            else:
                signals[symbol] = 0
            
            # Update previous price
            self.previous_prices[symbol] = current_price
        
        return signals
    
    def update_position(self, symbol: str, size: int) -> None:
        """Update position for both strategy and model.
        
        Args:
            symbol: Trading symbol
            size: Change in position size
        """
        super().update_position(symbol, size)
        self.model.update_position(symbol, size)
    
    def reset_grid(self, symbol: str) -> None:
        """Reset grid for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        self.model.reset_grid(symbol)
        self.grids_initialized[symbol] = False
        if symbol in self.previous_prices:
            del self.previous_prices[symbol]
        logger.info(f"Grid reset for {symbol}")
