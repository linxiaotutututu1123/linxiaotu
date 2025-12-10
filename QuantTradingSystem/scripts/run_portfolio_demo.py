import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import (
    DualMAStrategy, 
    MomentumStrategy,
    MultiStrategyPortfolio
)

def generate_asset_data(symbol, seed, trend_factor=0.0002):
    np.random.seed(seed)
    start = pd.Timestamp('2023-01-02')
    periods = 252 * 2
    idx = pd.bdate_range(start, periods=periods)
    
    # Generate returns with some trend
    returns = np.random.randn(periods) * 0.01 + trend_factor
    prices = 4000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'datetime': idx,
        'open': prices * (1 + np.random.randn(periods) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(periods)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(periods)) * 0.003),
        'close': prices,
        'volume': np.random.randint(500, 2000, periods),
        'turnover': np.random.randint(1_000_000, 2_000_000, periods),
        'open_interest': np.random.randint(10_000, 20_000, periods),
        'symbol': symbol,
        'exchange': 'SHFE',
        'interval': '1d'
    })
    return df

def run_portfolio_demo():
    print('='*60)
    print('       量化交易系统 - 多品种组合策略示例')
    print('='*60)

    # 1. 生成多品种数据
    symbols = ['rb9999', 'cu9999', 'ag9999']
    data_map = {}
    
    print('[数据] 生成模拟数据:')
    for i, symbol in enumerate(symbols):
        df = generate_asset_data(symbol, seed=100+i, trend_factor=0.0001 * (i+1))
        data_map[symbol] = df
        print(f'  - {symbol}: {len(df)} bars, 最终价格 {df.close.iloc[-1]:.2f}')

    # 2. 构建组合策略
    # 策略1: 螺纹钢双均线
    strat1 = DualMAStrategy(fast_period=10, slow_period=30, symbols=['rb9999'])
    strat1.strategy_name = "DualMA_RB"
    
    # 策略2: 铜动量
    strat2 = MomentumStrategy(momentum_period=20, holding_period=10, top_n=1, symbols=['cu9999'])
    strat2.strategy_name = "Momentum_CU"
    
    # 策略3: 白银双均线
    strat3 = DualMAStrategy(fast_period=20, slow_period=60, symbols=['ag9999'])
    strat3.strategy_name = "DualMA_AG"
    
    # 组合: 40% RB, 30% CU, 30% AG
    portfolio = MultiStrategyPortfolio(
        strategies=[
            (strat1, 0.4),
            (strat2, 0.3),
            (strat3, 0.3)
        ],
        rebalance_period=20  # 每20天再平衡
    )

    # 3. 配置回测
    config = BacktestConfig(
        initial_capital=5_000_000,  # 500万资金
        commission_rate=0.0001,
        slippage_rate=0.0001,
        margin_ratio=0.1,
        contract_size=10,
        match_mode='next_bar'
    )

    engine = BacktestEngine(config)
    engine.add_strategy(portfolio, "Portfolio_Demo")
    
    for symbol, df in data_map.items():
        engine.add_data(symbol, df)

    print('\n[回测] 开始运行组合策略...')
    metrics = engine.run()

    print('\n' + '='*60)
    print('===== 组合策略绩效 =====')
    print(f'总收益率: {metrics.total_return:.2%}')
    print(f'夏普比率: {metrics.sharpe_ratio:.2f}')
    print(f'最大回撤: {metrics.max_drawdown:.2%}')
    print(f'胜率: {metrics.win_rate:.2%}')
    print(f'总交易数: {metrics.total_trades}')
    print('='*60)

if __name__ == "__main__":
    run_portfolio_demo()
