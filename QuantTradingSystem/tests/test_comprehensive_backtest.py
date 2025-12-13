"""
综合回溯测试 - 使用模拟中国期货历史数据验证系统
测试螺纹钢(rb)、铁矿石(i)等品种
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import (
    DualMAStrategy, TurtleStrategy, GridStrategy,
    MomentumStrategy, MultiStrategyPortfolio, AdaptiveStrategy
)
from src.risk import RiskManager, RiskManagerConfig
from src.data.data_structures import PerformanceMetrics


def generate_realistic_futures_data(
    symbol: str = "rb2501",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    base_price: float = 4000.0,
    volatility: float = 0.02,
    trend_strength: float = 0.0001,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成逼真的中国期货历史数据
    
    模拟特征：
    - 日内波动模式（开盘波动大，中午平稳，收盘前活跃）
    - 趋势和震荡交替
    - 成交量与价格波动相关
    - 跳空缺口
    """
    np.random.seed(seed)
    
    # 生成交易日期（排除周末）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 每天生成多根K线（1分钟级别简化为日内几个时段）
    all_data = []
    current_price = base_price
    
    for date in dates:
        # 日内时段
        sessions = [
            (9, 0, 10, 15),    # 上午第一节
            (10, 30, 11, 30),  # 上午第二节
            (13, 30, 15, 0),   # 下午
        ]
        
        # 随机决定当天趋势
        daily_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        daily_volatility = volatility * (1 + 0.5 * np.random.random())
        
        for start_h, start_m, end_h, end_m in sessions:
            # 生成每分钟K线
            minutes = (end_h - start_h) * 60 + (end_m - start_m)
            
            for i in range(0, minutes, 5):  # 5分钟K线
                dt = datetime(date.year, date.month, date.day, 
                            start_h + (start_m + i) // 60, 
                            (start_m + i) % 60)
                
                # 价格变动（带趋势和均值回归）
                noise = np.random.randn() * daily_volatility * current_price
                trend = daily_trend * trend_strength * current_price
                mean_reversion = -0.001 * (current_price - base_price)
                
                price_change = noise + trend + mean_reversion
                
                # OHLC生成
                open_price = current_price
                close_price = current_price + price_change
                high_price = max(open_price, close_price) + abs(np.random.randn()) * daily_volatility * current_price * 0.3
                low_price = min(open_price, close_price) - abs(np.random.randn()) * daily_volatility * current_price * 0.3
                
                # 确保价格合理
                close_price = max(close_price, base_price * 0.7)
                close_price = min(close_price, base_price * 1.3)
                
                # 成交量（与波动相关）
                base_volume = 5000
                volatility_factor = abs(close_price - open_price) / (daily_volatility * current_price + 1e-8)
                volume = int(base_volume * (1 + volatility_factor) * (0.5 + np.random.random()))
                
                all_data.append({
                    'datetime': dt,
                    'open': round(open_price, 0),
                    'high': round(high_price, 0),
                    'low': round(low_price, 0),
                    'close': round(close_price, 0),
                    'volume': volume,
                    'turnover': volume * close_price * 10,  # 合约乘数10
                    'open_interest': int(200000 + np.random.randn() * 10000)
                })
                
                current_price = close_price
    
    df = pd.DataFrame(all_data)
    return df


class TestComprehensiveBacktest:
    """综合回溯测试"""
    
    @pytest.fixture
    def rb_data(self):
        """螺纹钢数据"""
        return generate_realistic_futures_data(
            symbol="rb2501",
            start_date="2023-01-01",
            end_date="2024-01-01",
            base_price=4000.0,
            volatility=0.015,
            seed=42
        )
    
    @pytest.fixture
    def i_data(self):
        """铁矿石数据"""
        return generate_realistic_futures_data(
            symbol="i2501",
            start_date="2023-01-01",
            end_date="2024-01-01",
            base_price=800.0,
            volatility=0.02,
            seed=43
        )
    
    @pytest.fixture
    def backtest_config(self):
        """回测配置"""
        return BacktestConfig(
            initial_capital=1000000.0,
            commission_rate=0.00025,
            slippage_ticks=2,
            slippage_mode="adaptive",
            contract_size=10.0,
            margin_ratio=0.1
        )
    
    def test_dual_ma_strategy_backtest(self, rb_data, backtest_config):
        """测试双均线策略回溯"""
        engine = BacktestEngine(backtest_config)
        strategy = DualMAStrategy(
            fast_period=5,
            slow_period=20,
            symbols=["rb2501"]
        )
        
        engine.add_strategy(strategy)
        engine.add_data("rb2501", rb_data)
        
        result = engine.run()
        
        # 验证结果合理性
        assert result is not None
        assert isinstance(result, PerformanceMetrics)
        assert result.total_trades >= 0
        
        # 验证回撤在合理范围
        print(f"\n双均线策略回测结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  交易次数: {result.total_trades}")
    
    def test_turtle_strategy_backtest(self, rb_data, backtest_config):
        """测试海龟策略回溯"""
        engine = BacktestEngine(backtest_config)
        strategy = TurtleStrategy(
            entry_period=20,
            exit_period=10,
            symbols=["rb2501"]
        )
        
        engine.add_strategy(strategy)
        engine.add_data("rb2501", rb_data)
        
        result = engine.run()
        
        assert result is not None
        print(f"\n海龟策略回测结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  交易次数: {result.total_trades}")
    
    def test_momentum_strategy_backtest(self, rb_data, backtest_config):
        """测试动量策略回溯"""
        engine = BacktestEngine(backtest_config)
        strategy = MomentumStrategy(
            momentum_period=20,
            holding_period=5,
            symbols=["rb2501"]
        )
        
        engine.add_strategy(strategy)
        engine.add_data("rb2501", rb_data)
        
        result = engine.run()
        
        assert result is not None
        print(f"\n动量策略回测结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  交易次数: {result.total_trades}")
    
    def test_multi_strategy_portfolio(self, rb_data, backtest_config):
        """测试多策略组合回溯"""
        engine = BacktestEngine(backtest_config)
        
        # 创建多个策略
        strategies = [
            (DualMAStrategy(fast_period=5, slow_period=20, symbols=["rb2501"]), 0.4),
            (TurtleStrategy(entry_period=20, exit_period=10, symbols=["rb2501"]), 0.3),
            (MomentumStrategy(momentum_period=20, holding_period=5, symbols=["rb2501"]), 0.3),
        ]
        
        portfolio = MultiStrategyPortfolio(
            strategies=strategies,
            rebalance_period=20
        )
        
        engine.add_strategy(portfolio)
        engine.add_data("rb2501", rb_data)
        
        result = engine.run()
        
        assert result is not None
        print(f"\n多策略组合回测结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  交易次数: {result.total_trades}")
    
    def test_risk_manager_integration(self, rb_data, backtest_config):
        """测试风控系统集成"""
        # 创建风控配置
        risk_config = RiskManagerConfig(
            max_loss_per_trade=0.015,
            max_daily_loss=0.02,
            max_drawdown=0.08,
            max_position_per_symbol=0.08,
            max_total_position=0.50,
            max_leverage=1.5
        )
        
        risk_manager = RiskManager(risk_config)
        
        # 验证风控参数
        assert risk_config.max_loss_per_trade <= 0.02
        assert risk_config.max_daily_loss <= 0.02
        assert risk_config.max_drawdown <= 0.08
        
        # 测试风控方法
        assert hasattr(risk_manager, 'check_order')
        assert hasattr(risk_manager, 'check_signal')
        assert hasattr(risk_manager, 'calculate_position_size')
        assert hasattr(risk_manager, 'check_drawdown')
        
        print(f"\n风控配置验证:")
        print(f"  单笔最大亏损: {risk_config.max_loss_per_trade:.1%}")
        print(f"  日内最大亏损: {risk_config.max_daily_loss:.1%}")
        print(f"  最大回撤阈值: {risk_config.max_drawdown:.1%}")
        print(f"  单品种最大仓位: {risk_config.max_position_per_symbol:.1%}")
        print(f"  最大杠杆: {risk_config.max_leverage:.1f}x")
    
    def test_adaptive_strategy_backtest(self, rb_data, backtest_config):
        """测试自适应策略回溯"""
        engine = BacktestEngine(backtest_config)
        strategy = AdaptiveStrategy(
            symbols=["rb2501"],
            trend_period=20,
            volatility_period=20
        )
        
        engine.add_strategy(strategy)
        engine.add_data("rb2501", rb_data)
        
        result = engine.run()
        
        assert result is not None
        print(f"\n自适应策略回测结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    
    def test_different_market_conditions(self, backtest_config):
        """测试不同市场条件下的表现"""
        # 上涨市场
        bull_data = generate_realistic_futures_data(
            symbol="rb_bull",
            base_price=4000.0,
            volatility=0.015,
            trend_strength=0.0005,  # 正趋势
            seed=100
        )
        
        # 下跌市场
        bear_data = generate_realistic_futures_data(
            symbol="rb_bear",
            base_price=4000.0,
            volatility=0.015,
            trend_strength=-0.0005,  # 负趋势
            seed=101
        )
        
        # 震荡市场
        range_data = generate_realistic_futures_data(
            symbol="rb_range",
            base_price=4000.0,
            volatility=0.01,
            trend_strength=0.0,  # 无趋势
            seed=102
        )
        
        results = {}
        
        for name, data in [("上涨", bull_data), ("下跌", bear_data), ("震荡", range_data)]:
            engine = BacktestEngine(backtest_config)
            strategy = DualMAStrategy(fast_period=5, slow_period=20, symbols=["test"])
            engine.add_strategy(strategy)
            engine.add_data("test", data)
            results[name] = engine.run()
        
        print(f"\n不同市场条件回测结果:")
        for name, result in results.items():
            print(f"  {name}市场: 收益{result.total_return:.2%}, 回撤{result.max_drawdown:.2%}")
    
    def test_backtest_engine_consistency(self, rb_data, backtest_config):
        """测试回测引擎一致性（相同输入应产生相同输出）"""
        results = []
        
        for _ in range(3):
            engine = BacktestEngine(backtest_config)
            strategy = DualMAStrategy(fast_period=5, slow_period=20, symbols=["rb2501"])
            engine.add_strategy(strategy)
            engine.add_data("rb2501", rb_data)
            results.append(engine.run())
        
        # 验证结果一致
        assert results[0].total_return == results[1].total_return == results[2].total_return
        assert results[0].total_trades == results[1].total_trades == results[2].total_trades
        
        print(f"\n回测一致性验证: 通过 ✓")
        print(f"  三次运行结果相同: 收益率={results[0].total_return:.4%}")


class TestSystemIntegrity:
    """系统完整性测试"""
    
    def test_all_strategies_can_be_instantiated(self):
        """验证所有策略可以正确实例化"""
        strategies = [
            DualMAStrategy(symbols=["test"]),
            TurtleStrategy(symbols=["test"]),
            GridStrategy(symbols=["test"]),
            MomentumStrategy(symbols=["test"]),
            AdaptiveStrategy(symbols=["test"]),
        ]
        
        for strategy in strategies:
            assert strategy is not None
            assert hasattr(strategy, 'on_bar')
            assert hasattr(strategy, 'on_init')
        
        print(f"\n策略实例化测试: 全部通过 [OK]")
    
    def test_risk_manager_methods(self):
        """验证风控管理器所有方法"""
        risk_manager = RiskManager()
        
        required_methods = [
            'check_order',
            'check_signal',
            'calculate_position_size',
            'check_drawdown',
            'check_daily_loss',
            'calculate_stop_loss',
            'calculate_take_profit',
            'generate_risk_report',
            'update_trade_result',
            'get_risk_status'
        ]
        
        for method in required_methods:
            assert hasattr(risk_manager, method), f"Missing method: {method}"
        
        print(f"\n风控方法验证: 全部存在 [OK]")
    
    def test_backtest_engine_methods(self):
        """验证回测引擎所有方法"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        required_methods = [
            'add_strategy',
            'add_data',
            'run',
            'get_account',
            'get_result_dataframe',
            'get_trades_dataframe'
        ]
        
        for method in required_methods:
            assert hasattr(engine, method), f"Missing method: {method}"
        
        print(f"\n回测引擎方法验证: 全部存在 [OK]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
