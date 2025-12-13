"""Example usage of PortfolioFormatter.

This script demonstrates how to use the PortfolioFormatter utility
to display backtest results in a formatted way.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.portfolio_formatter import PortfolioFormatter


def main():
    """Demonstrate PortfolioFormatter usage."""
    formatter = PortfolioFormatter()
    
    # Example backtest results
    backtest_results = {
        'initial_capital': 1000000,
        'final_capital': 1350000,
        'total_return': 0.35,
        'annual_return': 0.32,
        'max_drawdown': 0.12,
        'sharpe_ratio': 2.25,
        'total_trades': 180,
        'win_rate': 0.65
    }
    
    print(formatter.format_backtest_results(backtest_results))
    print()
    
    # Example trades
    trades = [
        {'date': '2023-01-01', 'symbol': 'RB', 'signal': 1, 'price': 3700},
        {'date': '2023-01-05', 'symbol': 'RB', 'signal': -1, 'price': 3750},
        {'date': '2023-01-10', 'symbol': 'IF', 'signal': 1, 'price': 4200},
        {'date': '2023-01-15', 'symbol': 'IF', 'signal': -1, 'price': 4250},
        {'date': '2023-01-20', 'symbol': 'RB', 'signal': 1, 'price': 3800},
    ]
    
    print(formatter.format_trades_summary(trades))
    print()
    
    # Example equity curve
    equity_curve = [1000000, 1020000, 1050000, 1030000, 1080000, 1100000, 1150000]
    
    print(formatter.format_equity_curve(equity_curve))


if __name__ == '__main__':
    main()
