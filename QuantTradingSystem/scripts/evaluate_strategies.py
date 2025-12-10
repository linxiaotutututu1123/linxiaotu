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
    MeanReversionStrategy,
    TurtleStrategy
)

def evaluate_strategies():
    print('='*60)
    print('       量化交易系统 - 多策略对比评估')
    print('='*60)

    # 1. 加载数据
    csv_path = project_root / 'data' / 'historical' / 'rb9999_demo.csv'
    if not csv_path.exists():
        print(f"Error: Data file {csv_path} not found. Please run run_historical_backtest.py first.")
        return

    bars_df = pd.read_csv(csv_path, parse_dates=['datetime'])
    print(f'已加载数据: {csv_path}, {len(bars_df)} bars')

    # 2. 定义策略配置
    strategies = [
        (
            "DualMA (10, 30)", 
            DualMAStrategy(fast_period=10, slow_period=30, symbols=['rb9999'])
        ),
        (
            "Momentum (20d)", 
            MomentumStrategy(momentum_period=20, holding_period=5, top_n=1, symbols=['rb9999'])
        ),
        (
            "MeanReversion (20d, 2std)", 
            MeanReversionStrategy(lookback_period=20, entry_std=2.0, exit_std=0.5, symbols=['rb9999'])
        ),
        (
            "Turtle (20d)", 
            TurtleStrategy(entry_period=20, exit_period=10, atr_period=20, symbols=['rb9999'])
        )
    ]

    # 3. 运行回测并收集结果
    results = []
    
    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=0.0001,
        slippage_rate=0.0001,
        margin_ratio=0.1,
        contract_size=10,
        match_mode='next_bar'
    )

    print('\n[评估] 开始运行策略回测...')
    
    for name, strategy in strategies:
        print(f'  Running {name}...')
        engine = BacktestEngine(config)
        engine.add_strategy(strategy, name)
        engine.add_data('rb9999', bars_df.copy())
        
        # 捕获输出以保持整洁
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            metrics = engine.run()
        finally:
            sys.stdout = old_stdout
            
        results.append({
            'Strategy': name,
            'Total Return': metrics.total_return,
            'Sharpe': metrics.sharpe_ratio,
            'Max Drawdown': metrics.max_drawdown,
            'Win Rate': metrics.win_rate,
            'Trades': metrics.total_trades
        })

    # 4. 展示结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Sharpe', ascending=False)
    
    print('\n' + '='*80)
    print('                  策略绩效对比排名')
    print('='*80)
    
    # 格式化输出
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 格式化百分比
    format_dict = {
        'Total Return': '{:.2%}',
        'Max Drawdown': '{:.2%}',
        'Win Rate': '{:.2%}',
        'Sharpe': '{:.2f}'
    }
    
    print(results_df.style.format(format_dict).to_string())
    
    best_strategy = results_df.iloc[0]
    print('\n' + '-'*80)
    print(f"🏆 最佳策略: {best_strategy['Strategy']}")
    print(f"   收益率: {best_strategy['Total Return']:.2%}")
    print(f"   夏普比率: {best_strategy['Sharpe']:.2f}")
    print('='*80)

if __name__ == "__main__":
    evaluate_strategies()
