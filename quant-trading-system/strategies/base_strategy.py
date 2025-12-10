"""Base strategy class for all trading strategies.

Provides common interface and functionality for trading strategies.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional
from utils.logger import get_logger


logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str):
        """Initialize base strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.positions: Dict[str, int] = {}
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generate trading signals for all symbols.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping symbol to signal (1=buy, -1=sell, 0=hold)
        """
        pass
    
    def update_position(self, symbol: str, size: int) -> None:
        """Update position for a symbol.
        
        Args:
            symbol: Trading symbol
            size: Change in position size
        """
        self.positions[symbol] = self.positions.get(symbol, 0) + size
        logger.debug(f"{self.name}: Position updated for {symbol}: {self.positions[symbol]}")
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.positions.get(symbol, 0)
    
    def reset_positions(self) -> None:
        """Reset all positions to zero."""
        self.positions = {}
        logger.info(f"{self.name}: All positions reset")
    
    def on_bar(self, symbol: str, bar: pd.Series) -> Optional[int]:
        """Handle new bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            bar: Bar data (OHLCV)
            
        Returns:
            Optional signal (1=buy, -1=sell, 0=hold, None=no action)
        """
        # Default implementation - can be overridden by subclasses
        return None
    
    def on_trade(self, symbol: str, price: float, size: int) -> None:
        """Handle trade execution callback.
        
        Args:
            symbol: Trading symbol
            price: Execution price
            size: Execution size
        """
        logger.debug(f"{self.name}: Trade executed - {symbol} @ {price}, size: {size}")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'positions': self.positions.copy(),
            'total_position': sum(abs(p) for p in self.positions.values())
        }
