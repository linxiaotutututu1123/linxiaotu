"""
策略模块初始化
"""

from .strategy_base import (
    BaseStrategy,
    DualMAStrategy,
    TurtleStrategy,
    GridStrategy,
    MeanReversionStrategy,
    MomentumStrategy
)

from .advanced_strategies import (
    SpreadArbitrageStrategy,
    CrossSymbolArbitrageStrategy,
    MultiStrategyPortfolio,
    AdaptiveStrategy
)

__all__ = [
    # 基类
    "BaseStrategy",
    # 经典策略
    "DualMAStrategy",
    "TurtleStrategy",
    "GridStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    # 高级策略
    "SpreadArbitrageStrategy",
    "CrossSymbolArbitrageStrategy",
    "MultiStrategyPortfolio",
    "AdaptiveStrategy"
]
