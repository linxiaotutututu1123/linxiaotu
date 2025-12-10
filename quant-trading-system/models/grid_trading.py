"""Grid trading model for capturing profits in range-bound markets.

Implements grid trading strategy that places buy/sell orders at predetermined price levels.
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger


logger = get_logger(__name__)


class GridTradingModel:
    """Grid trading model for systematic trading in price ranges."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize GridTradingModel with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.grid_config = self.config.get('grid_trading', {})
        
        # Grid parameters
        self.enabled = self.grid_config.get('enabled', True)
        self.grid_width = self.grid_config.get('grid', {}).get('grid_width', 10)
        self.grid_num = self.grid_config.get('grid', {}).get('grid_num', 20)
        
        # Symbol-specific overrides
        self.symbol_overrides = self.grid_config.get('symbol_override', {})
        
        # Position management
        self.pos_config = self.grid_config.get('position_management', {})
        self.max_position = self.pos_config.get('max_position', 50)
        self.entry_step = self.pos_config.get('entry_step', 1)
        self.exit_step = self.pos_config.get('exit_step', 1)
        
        # Grid state tracking
        self.grids: Dict[str, List[float]] = {}
        self.positions: Dict[str, int] = {}
        
        logger.info("GridTradingModel initialized")
    
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
    
    def setup_grid(
        self,
        symbol: str,
        base_price: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> List[float]:
        """Set up grid levels for a symbol.
        
        Args:
            symbol: Trading symbol
            base_price: Base price for grid center (optional)
            current_price: Current market price (used if base_price not provided)
            
        Returns:
            List of grid price levels
        """
        # Get symbol-specific parameters or use defaults
        if symbol in self.symbol_overrides:
            override = self.symbol_overrides[symbol]
            grid_width = override.get('grid_width', self.grid_width)
            grid_num = override.get('grid_num', self.grid_num)
            center_price = override.get('base_price', base_price or current_price)
        else:
            grid_width = self.grid_width
            grid_num = self.grid_num
            center_price = base_price or current_price
        
        if center_price is None:
            raise ValueError(f"Must provide base_price or current_price for {symbol}")
        
        # Generate grid levels
        half_range = (grid_num // 2) * grid_width
        grid_levels = []
        
        for i in range(grid_num + 1):
            level = center_price - half_range + (i * grid_width)
            grid_levels.append(level)
        
        self.grids[symbol] = sorted(grid_levels)
        
        logger.info(f"Grid setup for {symbol}: {len(grid_levels)} levels, "
                   f"range [{min(grid_levels):.2f}, {max(grid_levels):.2f}]")
        
        return grid_levels
    
    def find_grid_position(self, symbol: str, price: float) -> Optional[int]:
        """Find which grid level a price belongs to.
        
        Args:
            symbol: Trading symbol
            price: Current price
            
        Returns:
            Grid level index or None if outside grid range
        """
        if symbol not in self.grids:
            return None
        
        grid_levels = self.grids[symbol]
        
        # Find nearest grid level
        for i, level in enumerate(grid_levels):
            if price < level:
                return max(0, i - 1)
        
        return len(grid_levels) - 1
    
    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        previous_price: Optional[float] = None
    ) -> int:
        """Generate trading signal based on grid levels.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            previous_price: Previous price (to detect crossings)
            
        Returns:
            Signal (1 = buy, -1 = sell, 0 = hold)
        """
        if not self.enabled or symbol not in self.grids:
            return 0
        
        grid_levels = self.grids[symbol]
        current_position = self.positions.get(symbol, 0)
        
        # Check if price is outside grid range (risk management)
        if current_price < grid_levels[0] or current_price > grid_levels[-1]:
            logger.warning(f"Price {current_price} outside grid range for {symbol}")
            # Exit positions if price breaks out
            if current_position > 0:
                return -1  # Sell
            elif current_position < 0:
                return 1  # Buy to cover
            return 0
        
        # Find current grid level
        current_level_idx = self.find_grid_position(symbol, current_price)
        
        if current_level_idx is None:
            return 0
        
        # If we have previous price, check for grid crossing
        if previous_price is not None:
            prev_level_idx = self.find_grid_position(symbol, previous_price)
            
            if prev_level_idx is not None and prev_level_idx != current_level_idx:
                # Price crossed a grid level
                if current_level_idx < prev_level_idx:
                    # Price went down - buy signal
                    if abs(current_position) < self.max_position:
                        logger.debug(f"Grid buy signal for {symbol} at {current_price}")
                        return 1
                elif current_level_idx > prev_level_idx:
                    # Price went up - sell signal
                    if current_position > 0:
                        logger.debug(f"Grid sell signal for {symbol} at {current_price}")
                        return -1
        
        return 0
    
    def calculate_position_size(self, symbol: str, signal: int) -> int:
        """Calculate position size for grid trading.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            
        Returns:
            Position size in contracts
        """
        if signal == 1:
            return self.entry_step
        elif signal == -1:
            return -self.exit_step
        else:
            return 0
    
    def update_position(self, symbol: str, size: int) -> None:
        """Update position tracking.
        
        Args:
            symbol: Trading symbol
            size: Change in position size
        """
        self.positions[symbol] = self.positions.get(symbol, 0) + size
        logger.debug(f"Grid position updated for {symbol}: {self.positions[symbol]}")
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.positions.get(symbol, 0)
    
    def reset_grid(self, symbol: str) -> None:
        """Reset grid for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.grids:
            del self.grids[symbol]
        if symbol in self.positions:
            self.positions[symbol] = 0
        
        logger.info(f"Grid reset for {symbol}")
    
    def get_grid_info(self, symbol: str) -> Dict:
        """Get grid information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with grid information
        """
        return {
            'enabled': self.enabled,
            'levels': self.grids.get(symbol, []),
            'position': self.positions.get(symbol, 0),
            'max_position': self.max_position,
            'grid_width': self.grid_width,
            'grid_num': self.grid_num
        }
