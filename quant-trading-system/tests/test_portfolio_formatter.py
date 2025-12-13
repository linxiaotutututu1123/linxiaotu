"""Tests for portfolio formatter utility."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.portfolio_formatter import PortfolioFormatter


def test_portfolio_formatter_initialization():
    """Test PortfolioFormatter initialization."""
    formatter = PortfolioFormatter()
    assert formatter is not None


def test_format_backtest_results():
    """Test formatting backtest results."""
    formatter = PortfolioFormatter()
    
    # Sample backtest results
    results = {
        'initial_capital': 1000000,
        'final_capital': 1300000,
        'total_return': 0.30,
        'annual_return': 0.28,
        'max_drawdown': 0.12,
        'sharpe_ratio': 2.15,
        'total_trades': 150,
        'win_rate': 0.62
    }
    
    output = formatter.format_backtest_results(results)
    
    # Check that output is a string
    assert isinstance(output, str)
    # Check that key metrics are included
    assert 'Total Return' in output or '30.00%' in output
    assert 'Sharpe' in output or '2.15' in output
    assert '150' in output  # Total trades


def test_format_backtest_results_with_missing_fields():
    """Test formatting with missing fields."""
    formatter = PortfolioFormatter()
    
    # Incomplete results
    results = {
        'initial_capital': 1000000,
        'final_capital': 1100000
    }
    
    output = formatter.format_backtest_results(results)
    
    # Should not raise error and return string
    assert isinstance(output, str)
    assert '1000000' in output or '1,000,000' in output


def test_format_trades_summary():
    """Test formatting trades summary."""
    formatter = PortfolioFormatter()
    
    # Sample trades
    trades = [
        {'date': '2023-01-01', 'symbol': 'RB', 'signal': 1, 'price': 3700, 'size': 10},
        {'date': '2023-01-02', 'symbol': 'RB', 'signal': -1, 'price': 3750, 'size': -10},
        {'date': '2023-01-03', 'symbol': 'IF', 'signal': 1, 'price': 4200, 'size': 5},
    ]
    
    output = formatter.format_trades_summary(trades)
    
    # Check output
    assert isinstance(output, str)
    assert '3' in output  # Total number of trades


def test_format_empty_trades():
    """Test formatting empty trades list."""
    formatter = PortfolioFormatter()
    
    output = formatter.format_trades_summary([])
    
    assert isinstance(output, str)
    assert 'No trades' in output or '0' in output


def test_format_equity_curve():
    """Test formatting equity curve."""
    formatter = PortfolioFormatter()
    
    equity_curve = [1000000, 1050000, 1020000, 1080000, 1100000]
    
    output = formatter.format_equity_curve(equity_curve)
    
    assert isinstance(output, str)
    assert '1000000' in output or '1,000,000' in output
    assert '1100000' in output or '1,100,000' in output
