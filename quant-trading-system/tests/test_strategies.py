"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.factor_strategy import FactorStrategy
from strategies.grid_strategy import GridStrategy
from strategies.combined_strategy import CombinedStrategy


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


def test_factor_strategy_initialization():
    """Test FactorStrategy initialization."""
    strategy = FactorStrategy()
    assert strategy.name == "FactorStrategy"
    assert strategy.model is not None


def test_factor_strategy_signals():
    """Test FactorStrategy signal generation."""
    strategy = FactorStrategy()
    
    data = {
        'RB': generate_sample_data(100),
        'IF': generate_sample_data(100)
    }
    
    signals = strategy.generate_signals(data)
    
    # Verify signals structure
    assert 'RB' in signals
    assert 'IF' in signals
    
    # Verify signal values are valid
    for symbol, signal in signals.items():
        assert signal in [-1, 0, 1]


def test_grid_strategy_initialization():
    """Test GridStrategy initialization."""
    strategy = GridStrategy()
    assert strategy.name == "GridStrategy"
    assert strategy.model is not None


def test_grid_strategy_signals():
    """Test GridStrategy signal generation."""
    strategy = GridStrategy()
    
    data = {
        'RB': generate_sample_data(100)
    }
    
    signals = strategy.generate_signals(data)
    
    # Verify signals structure
    assert 'RB' in signals
    assert signals['RB'] in [-1, 0, 1]


def test_combined_strategy_initialization():
    """Test CombinedStrategy initialization."""
    strategy = CombinedStrategy()
    assert strategy.name == "CombinedStrategy"
    assert strategy.factor_strategy is not None
    assert strategy.grid_strategy is not None


def test_combined_strategy_signals():
    """Test CombinedStrategy signal generation."""
    strategy = CombinedStrategy()
    
    data = {
        'RB': generate_sample_data(100),
        'IF': generate_sample_data(100)
    }
    
    signals = strategy.generate_signals(data)
    
    # Verify signals structure
    assert 'RB' in signals
    assert 'IF' in signals
    
    # Verify signal values are valid
    for symbol, signal in signals.items():
        assert signal in [-1, 0, 1]


def test_strategy_position_tracking():
    """Test strategy position tracking."""
    strategy = FactorStrategy()
    
    # Update positions
    strategy.update_position('RB', 5)
    assert strategy.get_position('RB') == 5
    
    strategy.update_position('RB', -2)
    assert strategy.get_position('RB') == 3
    
    # Reset positions
    strategy.reset_positions()
    assert strategy.get_position('RB') == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
