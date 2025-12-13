"""Tests for backtest engine functionality."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_engine import BacktestEngine
from strategies.factor_strategy import FactorStrategy


def generate_sample_data(days=100):
    """Generate sample OHLCV data for testing.
    
    Args:
        days: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    base_price = 3700
    close_prices = base_price + np.cumsum(np.random.randn(days) * 10)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(days) * 5,
        'high': close_prices + abs(np.random.randn(days)) * 10,
        'low': close_prices - abs(np.random.randn(days)) * 10,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)
    
    return data


def test_backtest_engine_initialization():
    """Test BacktestEngine initialization."""
    engine = BacktestEngine()
    assert engine.initial_capital > 0
    assert engine.commission >= 0
    assert engine.slippage >= 0


def test_backtest_run():
    """Test running a basic backtest."""
    engine = BacktestEngine()
    strategy = FactorStrategy()
    
    # Generate sample data
    data = {
        'RB': generate_sample_data(100),
        'IF': generate_sample_data(100)
    }
    
    # Run backtest
    results = engine.run_backtest(strategy, data)
    
    # Verify results structure
    assert 'initial_capital' in results
    assert 'final_capital' in results
    assert 'total_return' in results
    assert 'annual_return' in results
    assert 'max_drawdown' in results
    assert 'sharpe_ratio' in results
    assert 'total_trades' in results
    assert 'win_rate' in results
    
    # Verify results validity
    assert results['initial_capital'] > 0
    assert results['final_capital'] > 0
    assert results['total_trades'] >= 0


def test_backtest_equity_curve():
    """Test equity curve generation."""
    engine = BacktestEngine()
    strategy = FactorStrategy()
    
    data = {'RB': generate_sample_data(50)}
    results = engine.run_backtest(strategy, data)
    
    # Check equity curve exists and has values
    assert 'equity_curve' in results
    assert len(results['equity_curve']) > 0
    assert results['equity_curve'][0] == engine.initial_capital


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
