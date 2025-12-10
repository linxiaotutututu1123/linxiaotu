"""
测试回测引擎
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import DualMAStrategy
from src.data import BarData


def generate_test_data(days: int = 100) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')
    
    # 生成价格序列 (带有趋势和波动)
    returns = np.random.randn(len(dates)) * 0.002
    returns[len(dates)//3:2*len(dates)//3] += 0.001  # 中间段上涨趋势
    
    close = 4000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(len(dates)) * 0.003)),
        'low': close * (1 - np.abs(np.random.randn(len(dates)) * 0.003)),
        'close': close,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    df.set_index('datetime', inplace=True)
    df['symbol'] = 'rb2501'
    
    return df


class TestBacktestEngine:
    """回测引擎测试"""
    
    def test_basic_backtest(self):
        """基础回测测试"""
        # 生成测试数据
        df = generate_test_data(30)
        
        # 创建策略
        strategy = DualMAStrategy(
            name="TestDualMA",
            symbols=["rb2501"],
            fast_period=5,
            slow_period=20
        )
        
        # 配置回测
        config = BacktestConfig(
            initial_capital=1000000,
            commission_rate=0.0003,
            slippage=1.0
        )
        
        # 运行回测
        engine = BacktestEngine(config)
        result = engine.run(strategy, df)
        
        # 验证结果
        assert result is not None
        assert result.final_equity > 0
        assert result.total_trades >= 0
    
    def test_backtest_with_custom_strategy(self):
        """自定义策略回测"""
        from src.strategy import BaseStrategy
        from src.data import SignalData
        
        class SimpleStrategy(BaseStrategy):
            def on_bar(self, bar):
                # 简单策略：每10根K线交替做多做空
                if len(self._bar_cache) % 20 == 10:
                    return SignalData(
                        symbol=bar.symbol,
                        direction=1,
                        strength=0.8,
                        price=bar.close,
                        timestamp=bar.datetime
                    )
                elif len(self._bar_cache) % 20 == 0:
                    return SignalData(
                        symbol=bar.symbol,
                        direction=-1,
                        strength=0.8,
                        price=bar.close,
                        timestamp=bar.datetime
                    )
                return None
        
        df = generate_test_data(50)
        strategy = SimpleStrategy(name="Simple", symbols=["rb2501"])
        
        config = BacktestConfig(initial_capital=500000)
        engine = BacktestEngine(config)
        result = engine.run(strategy, df)
        
        assert result is not None
        assert result.total_trades > 0
    
    def test_backtest_metrics(self):
        """回测指标验证"""
        df = generate_test_data(60)
        strategy = DualMAStrategy(name="TestMA", symbols=["rb2501"])
        
        config = BacktestConfig(initial_capital=1000000)
        engine = BacktestEngine(config)
        result = engine.run(strategy, df)
        
        # 验证指标计算
        assert -1 <= result.total_return <= 10  # 收益率合理范围
        assert result.max_drawdown >= 0 and result.max_drawdown <= 1
        assert 0 <= result.win_rate <= 1 if result.total_trades > 0 else True


class TestStrategy:
    """策略测试"""
    
    def test_dual_ma_strategy(self):
        """双均线策略测试"""
        strategy = DualMAStrategy(
            name="DualMA",
            symbols=["rb2501"],
            fast_period=5,
            slow_period=10
        )
        
        # 模拟K线数据
        for i in range(50):
            bar = BarData(
                symbol="rb2501",
                datetime=datetime.now() - timedelta(minutes=50-i),
                open=4000 + i * 10,
                high=4010 + i * 10,
                low=3990 + i * 10,
                close=4005 + i * 10,
                volume=1000
            )
            signal = strategy.on_bar(bar)
        
        # 验证策略状态
        assert len(strategy._bar_cache) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
