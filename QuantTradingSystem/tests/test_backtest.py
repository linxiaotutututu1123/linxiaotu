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
from typing import Dict, List

from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import DualMAStrategy, BaseStrategy
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
    
    return df


class TestBacktestEngine:
    """回测引擎测试"""
    
    def test_basic_backtest(self):
        """基础回测测试"""
        # 生成测试数据
        df = generate_test_data(30)
        
        # 创建策略 - 使用正确的参数
        strategy = DualMAStrategy(
            fast_period=5,
            slow_period=20,
            symbols=["rb2501"]
        )
        
        # 配置回测 - 使用正确的参数名
        config = BacktestConfig(
            initial_capital=1000000,
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        # 运行回测 - 使用正确的API
        engine = BacktestEngine(config)
        engine.add_strategy(strategy)
        engine.add_data("rb2501", df)
        result = engine.run()
        
        # 验证结果 - 使用正确的属性名
        assert result is not None
        assert engine.get_account().balance > 0
        assert result.total_trades >= 0
    
    def test_backtest_with_custom_strategy(self):
        """自定义策略回测"""
        
        class SimpleStrategy(BaseStrategy):
            strategy_name = "SimpleTest"
            
            def __init__(self, symbols: List[str]):
                super().__init__()
                self.symbols = symbols
                self._counter = 0
            
            def on_bar(self, bars: Dict[str, BarData]) -> None:
                """每20根K线交替做多做空"""
                self._counter += 1
                for symbol, bar in bars.items():
                    if symbol not in self.symbols:
                        continue
                    
                    if self._counter % 20 == 10:
                        # 做多信号
                        self.buy(symbol, bar.close, 1)
                    elif self._counter % 20 == 0:
                        # 平仓信号
                        pos = self.get_position(symbol)
                        if pos and pos.volume > 0:
                            self.sell(symbol, bar.close, pos.volume)
        
        df = generate_test_data(50)
        strategy = SimpleStrategy(symbols=["rb2501"])
        
        config = BacktestConfig(initial_capital=500000)
        engine = BacktestEngine(config)
        engine.add_strategy(strategy)
        engine.add_data("rb2501", df)
        result = engine.run()
        
        assert result is not None
        assert result.total_trades >= 0
    
    def test_backtest_metrics(self):
        """回测指标验证"""
        df = generate_test_data(60)
        strategy = DualMAStrategy(
            fast_period=5,
            slow_period=20,
            symbols=["rb2501"]
        )
        
        config = BacktestConfig(initial_capital=1000000)
        engine = BacktestEngine(config)
        engine.add_strategy(strategy)
        engine.add_data("rb2501", df)
        result = engine.run()
        
        # 验证指标计算
        assert -1 <= result.total_return <= 10  # 收益率合理范围
        assert result.max_drawdown >= 0 and result.max_drawdown <= 1
        assert 0 <= result.win_rate <= 1 if result.total_trades > 0 else True


class TestStrategy:
    """策略测试"""
    
    def test_dual_ma_strategy(self):
        """双均线策略测试"""
        strategy = DualMAStrategy(
            fast_period=5,
            slow_period=10,
            symbols=["rb2501"]
        )
        
        # 初始化策略
        strategy.on_init()
        
        # 模拟K线数据
        for i in range(50):
            bar = BarData(
                symbol="rb2501",
                exchange="SHFE",
                datetime=datetime.now() - timedelta(minutes=50-i),
                interval="1m",
                open=4000 + i * 10,
                high=4010 + i * 10,
                low=3990 + i * 10,
                close=4005 + i * 10,
                volume=1000,
                turnover=0.0,
                open_interest=0
            )
            # on_bar 接收的是 Dict[str, BarData]
            strategy.on_bar({"rb2501": bar})
        
        # 验证策略状态 - 使用正确的属性名
        assert len(strategy._bar_buffers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
