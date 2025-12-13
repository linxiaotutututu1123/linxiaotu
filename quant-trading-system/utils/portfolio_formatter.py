"""Portfolio statistics formatter utility.

Provides formatted output for portfolio and backtest statistics.
"""

from typing import Dict, List, Any


class PortfolioFormatter:
    """Formatter for portfolio statistics and backtest results."""
    
    def format_backtest_results(self, results: Dict[str, Any]) -> str:
        """Format backtest results in a readable table format.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST RESULTS")
        lines.append("=" * 60)
        
        # Capital information
        initial = results.get('initial_capital', 0)
        final = results.get('final_capital', 0)
        lines.append(f"Initial Capital:    {self._format_number(initial)}")
        lines.append(f"Final Capital:      {self._format_number(final)}")
        lines.append(f"Profit/Loss:        {self._format_number(final - initial)}")
        lines.append("-" * 60)
        
        # Performance metrics
        total_return = results.get('total_return', 0)
        annual_return = results.get('annual_return', 0)
        max_drawdown = results.get('max_drawdown', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        
        lines.append(f"Total Return:       {self._format_percentage(total_return)}")
        lines.append(f"Annual Return:      {self._format_percentage(annual_return)}")
        lines.append(f"Max Drawdown:       {self._format_percentage(max_drawdown)}")
        lines.append(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
        lines.append("-" * 60)
        
        # Trade statistics
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0)
        
        lines.append(f"Total Trades:       {total_trades}")
        lines.append(f"Win Rate:           {self._format_percentage(win_rate)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def format_trades_summary(self, trades: List[Dict[str, Any]]) -> str:
        """Format trades summary.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TRADES SUMMARY")
        lines.append("=" * 60)
        
        if not trades:
            lines.append("No trades executed")
            lines.append("=" * 60)
            return "\n".join(lines)
        
        lines.append(f"Total Trades:       {len(trades)}")
        
        # Count buy and sell signals
        buy_trades = sum(1 for t in trades if t.get('signal', 0) > 0)
        sell_trades = sum(1 for t in trades if t.get('signal', 0) < 0)
        
        lines.append(f"Buy Signals:        {buy_trades}")
        lines.append(f"Sell Signals:       {sell_trades}")
        lines.append("-" * 60)
        
        # Symbol distribution
        symbols = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            symbols[symbol] = symbols.get(symbol, 0) + 1
        
        lines.append("Trades by Symbol:")
        for symbol, count in sorted(symbols.items()):
            lines.append(f"  {symbol:<15} {count:>5} trades")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def format_equity_curve(self, equity_curve: List[float]) -> str:
        """Format equity curve summary.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EQUITY CURVE SUMMARY")
        lines.append("=" * 60)
        
        if not equity_curve:
            lines.append("No equity data available")
            lines.append("=" * 60)
            return "\n".join(lines)
        
        lines.append(f"Starting Equity:    {self._format_number(equity_curve[0])}")
        lines.append(f"Ending Equity:      {self._format_number(equity_curve[-1])}")
        lines.append(f"Peak Equity:        {self._format_number(max(equity_curve))}")
        lines.append(f"Lowest Equity:      {self._format_number(min(equity_curve))}")
        lines.append(f"Data Points:        {len(equity_curve)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_number(self, value: float) -> str:
        """Format a number with thousands separator.
        
        Args:
            value: Number to format
            
        Returns:
            Formatted string
        """
        return f"{value:,.2f}"
    
    def _format_percentage(self, value: float) -> str:
        """Format a decimal as percentage.
        
        Args:
            value: Decimal value (e.g., 0.30 for 30%)
            
        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.2f}%"
