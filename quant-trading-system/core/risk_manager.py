"""Risk management module for position and capital control.

Implements risk controls including position limits, drawdown management, and volatility adjustments.
"""

import yaml
from typing import Dict, Optional
from utils.logger import get_logger


logger = get_logger(__name__)


class RiskManager:
    """Manages trading risks including position sizing and loss limits."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize RiskManager with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.risk_config = self.config.get('risk_management', {})
        
        # Position limits
        self.max_total_position = self.risk_config.get('position', {}).get('max_total_position', 100)
        self.max_single_symbol = self.risk_config.get('position', {}).get('max_single_symbol', 50)
        self.max_capital_usage = self.risk_config.get('position', {}).get('max_capital_usage', 0.80)
        
        # Stop loss settings
        self.max_drawdown = self.risk_config.get('stop_loss', {}).get('max_drawdown', 0.15)
        self.single_trade_loss = self.risk_config.get('stop_loss', {}).get('single_trade_loss', 0.05)
        self.daily_loss_limit = self.risk_config.get('stop_loss', {}).get('daily_loss_limit', 0.05)
        
        # Volatility adjustment
        self.high_vol_threshold = self.risk_config.get('volatility_adjustment', {}).get('high_volatility_threshold', 0.08)
        self.medium_vol_threshold = self.risk_config.get('volatility_adjustment', {}).get('medium_volatility_threshold', 0.05)
        self.high_vol_ratio = self.risk_config.get('volatility_adjustment', {}).get('high_vol_position_ratio', 0.5)
        self.medium_vol_ratio = self.risk_config.get('volatility_adjustment', {}).get('medium_vol_position_ratio', 0.75)
        
        # Tracking
        self.current_positions: Dict[str, int] = {}
        self.daily_pnl = 0.0
        self.total_capital = 0.0
        self.peak_capital = 0.0
        
        logger.info("RiskManager initialized")
    
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
    
    def check_position_limit(self, symbol: str, additional_size: int) -> bool:
        """Check if adding position would exceed limits.
        
        Args:
            symbol: Trading symbol
            additional_size: Number of contracts to add
            
        Returns:
            True if position is allowed, False otherwise
        """
        current_symbol_position = abs(self.current_positions.get(symbol, 0))
        total_position = sum(abs(pos) for pos in self.current_positions.values())
        
        # Check single symbol limit
        if current_symbol_position + additional_size > self.max_single_symbol:
            logger.warning(f"Position limit exceeded for {symbol}")
            return False
        
        # Check total position limit
        if total_position + additional_size > self.max_total_position:
            logger.warning("Total position limit exceeded")
            return False
        
        return True
    
    def check_drawdown(self, current_capital: float) -> bool:
        """Check if current drawdown exceeds limit.
        
        Args:
            current_capital: Current account capital
            
        Returns:
            True if within limit, False if exceeded
        """
        if self.peak_capital == 0:
            self.peak_capital = current_capital
            self.total_capital = current_capital
            return True
        
        # Update peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Calculate drawdown
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if drawdown > self.max_drawdown:
            logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            return False
        
        return True
    
    def check_daily_loss(self, daily_pnl: float, initial_capital: float) -> bool:
        """Check if daily loss exceeds limit.
        
        Args:
            daily_pnl: Daily profit/loss
            initial_capital: Initial capital for the day
            
        Returns:
            True if within limit, False if exceeded
        """
        daily_loss_rate = abs(daily_pnl) / initial_capital
        
        if daily_pnl < 0 and daily_loss_rate > self.daily_loss_limit:
            logger.warning(f"Daily loss limit exceeded: {daily_loss_rate:.2%}")
            return False
        
        return True
    
    def calculate_position_size(
        self,
        symbol: str,
        volatility: float,
        capital: float
    ) -> int:
        """Calculate position size based on volatility.
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility level
            capital: Available capital
            
        Returns:
            Recommended position size in contracts
        """
        # Base position size
        base_size = int(capital * self.max_capital_usage / (self.max_total_position * 10000))
        
        # Adjust for volatility
        if volatility > self.high_vol_threshold:
            adjusted_size = int(base_size * self.high_vol_ratio)
            logger.debug(f"High volatility detected for {symbol}, reducing position")
        elif volatility > self.medium_vol_threshold:
            adjusted_size = int(base_size * self.medium_vol_ratio)
            logger.debug(f"Medium volatility detected for {symbol}")
        else:
            adjusted_size = base_size
        
        # Ensure within limits
        adjusted_size = min(adjusted_size, self.max_single_symbol)
        
        return max(1, adjusted_size)
    
    def update_position(self, symbol: str, size: int) -> None:
        """Update position tracking.
        
        Args:
            symbol: Trading symbol
            size: Position size (positive for long, negative for short)
        """
        self.current_positions[symbol] = self.current_positions.get(symbol, 0) + size
        logger.debug(f"Position updated for {symbol}: {self.current_positions[symbol]}")
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.current_positions.get(symbol, 0)
    
    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics."""
        self.daily_pnl = 0.0
        logger.debug("Daily tracking reset")
    
    def can_trade(
        self,
        symbol: str,
        size: int,
        current_capital: float,
        volatility: Optional[float] = None
    ) -> bool:
        """Check if a trade is allowed based on all risk controls.
        
        Args:
            symbol: Trading symbol
            size: Intended position size
            current_capital: Current account capital
            volatility: Optional volatility level
            
        Returns:
            True if trade is allowed, False otherwise
        """
        # Check position limits
        if not self.check_position_limit(symbol, abs(size)):
            return False
        
        # Check drawdown
        if not self.check_drawdown(current_capital):
            return False
        
        # Check daily loss
        if not self.check_daily_loss(self.daily_pnl, self.total_capital):
            return False
        
        return True
