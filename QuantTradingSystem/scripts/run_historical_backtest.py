import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import DualMAStrategy

def run_historical_backtest():
    print('='*60)
    print('       量化交易系统 - 历史数据回测示例')
    print('='*60)

    # 1) 准备历史数据CSV（日线，2年）
    np.random.seed(123)
    start = pd.Timestamp('2023-01-02')
    periods = 252 * 2
    idx = pd.bdate_range(start, periods=periods)
    returns = np.random.randn(periods) * 0.01
    prices = 4000 * np.exp(np.cumsum(returns))

    bars = pd.DataFrame({
        'datetime': idx,
        'open': prices * (1 + np.random.randn(periods) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(periods)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(periods)) * 0.003),
        'close': prices,
        'volume': np.random.randint(500, 2000, periods),
        'turnover': np.random.randint(1_000_000, 2_000_000, periods),
        'open_interest': np.random.randint(10_000, 20_000, periods),
        'symbol': 'rb9999',
        'exchange': 'SHFE',
        'interval': '1d'
    })

    csv_path = project_root / 'data' / 'historical' / 'rb9999_demo.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(csv_path, index=False)

    print(f'已生成历史CSV: {csv_path}')
    print(f'行数: {len(bars)}')
    print(f'价格范围: {bars.close.min():.2f} -> {bars.close.max():.2f}')

    # 2) 读取CSV并运行回测
    bars_df = pd.read_csv(csv_path, parse_dates=['datetime'])

    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.0001,
        slippage_rate=0.0001,
        margin_ratio=0.1,
        contract_size=10,
        match_mode='next_bar'
    )

    strategy = DualMAStrategy(fast_period=15, slow_period=30, symbols=['rb9999'])
    engine = BacktestEngine(config)
    engine.add_strategy(strategy, 'DualMA_15_30')
    engine.add_data('rb9999', bars_df)

    print('\n[回测] 开始运行...')
    metrics = engine.run()

    print('\n' + '='*60)
    print('===== 回测结果 (CSV) =====')
    print(f'总收益率: {metrics.total_return:.2%}')
    print(f'夏普比率: {metrics.sharpe_ratio:.2f}')
    print(f'最大回撤: {metrics.max_drawdown:.2%}')
    print(f'胜率: {metrics.win_rate:.2%}')
    print(f'总交易数: {metrics.total_trades}')
    print('='*60)

if __name__ == "__main__":
    run_historical_backtest()
