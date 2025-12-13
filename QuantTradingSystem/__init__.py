"""
量化交易系统 (Quantitative Trading System)
==========================================

高性能量化交易系统，支持中国期货市场
目标：50%年化收益，最大回撤<8%，单笔亏损<2%

主要功能：
- 多策略组合（双均线、海龟、网格、均值回归、动量）
- AI增强预测（LSTM、Transformer）
- 多层风险管理（熔断、VaR、动态仓位）
- 高速回测引擎
- 实时监控前端

Author: QuantTrader
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "QuantTrader"

from src.strategies import (
    DualMAStrategy,
    TurtleStrategy,
    GridStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MultiStrategyPortfolio,
    AdaptiveStrategy,
)

from src.backtest import BacktestEngine
from src.risk import RiskManager

__all__ = [
    "DualMAStrategy",
    "TurtleStrategy", 
    "GridStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MultiStrategyPortfolio",
    "AdaptiveStrategy",
    "BacktestEngine",
    "RiskManager",
]
