"""Backtest engine for strategy testing.

Provides backtesting functionality using historical data to evaluate trading strategies.
"""

import pandas as pd
import yaml
from typing import Dict, List, Optional
from datetime import datetime
from utils.logger import get_logger
from utils.decorators import timing_decorator


logger = get_logger(__name__)


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize BacktestEngine with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.backtest_config = self.config.get('backtest', {})
        
        self.initial_capital = self.backtest_config.get('initial_capital', 1000000)
        self.commission = self.backtest_config.get('commission', 0.0001)
        self.slippage = self.backtest_config.get('slippage', 0.5)
        
        # Tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.positions: Dict[str, int] = {}
        self.cash = self.initial_capital
        
        logger.info(f"BacktestEngine initialized with capital: {self.initial_capital}")
    
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
    def run_backtest(
        self,
        strategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy instance
            data: Dictionary of symbol -> DataFrame
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Reset state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.positions = {}
        self.cash = self.initial_capital
        
        # Get common date range
        all_dates = self._get_common_dates(data)
        
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        
        # Run strategy for each date
        for idx, date in enumerate(all_dates):
            # Get current data slice
            current_data = {
                symbol: df.loc[:date] for symbol, df in data.items()
                if date in df.index
            }
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if signal != 0 and symbol in current_data:
                    self._execute_trade(symbol, signal, current_data[symbol].loc[date], date)
            
            # Update equity curve
            equity = self._calculate_equity(current_data, date)
            self.equity_curve.append(equity)
            
            if idx % 1000 == 0:
                logger.debug(f"Processed {idx}/{len(all_dates)} dates")
        
        # Calculate performance metrics
        results = self._calculate_metrics()
        
        logger.info("Backtest completed")
        return results
    
    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        """Get common dates across all data series.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            List of common timestamps
        """
        if not data:
            return []
        
        # Use first symbol's dates as base
        first_symbol = list(data.keys())[0]
        return sorted(data[first_symbol].index.tolist())
    
    def _execute_trade(
        self,
        symbol: str,
        signal: int,
        bar: pd.Series,
        date: pd.Timestamp
    ) -> None:
        """Execute a trade based on signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            bar: Current bar data
            date: Current date
        """
        price = bar['close']
        
        # Apply slippage
        if signal > 0:
            price += self.slippage
        elif signal < 0:
            price -= self.slippage
        
        # Calculate position size (simplified)
        position_size = signal
        
        # Calculate cost
        trade_value = abs(position_size) * price
        commission_cost = trade_value * self.commission
        
        # Check if we have enough cash
        if signal > 0 and (trade_value + commission_cost) > self.cash:
            logger.debug(f"Insufficient cash for trade: {symbol}")
            return
        
        # Update positions and cash
        self.positions[symbol] = self.positions.get(symbol, 0) + position_size
        
        if signal > 0:
            self.cash -= (trade_value + commission_cost)
        else:
            self.cash += (trade_value - commission_cost)
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'size': position_size,
            'commission': commission_cost,
            'cash': self.cash
        }
        self.trades.append(trade)
        
        logger.debug(f"Trade executed: {symbol} @ {price}, signal: {signal}")
    
    def _calculate_equity(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """Calculate current equity value.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            date: Current date
            
        Returns:
            Total equity value
        """
        equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in data and date in data[symbol].index:
                price = data[symbol].loc[date, 'close']
                equity += position * price
        
        return equity
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        
        # Calculate annual return (assuming 252 trading days)
        trading_days = len(equity_series)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
        
        # Win rate
        winning_trades = sum(1 for t in self.trades if 'pnl' in t and t.get('pnl', 0) > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_series.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        # Log summary
        logger.info(f"Backtest Results:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Max Drawdown: {abs(max_drawdown):.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        
        return results
