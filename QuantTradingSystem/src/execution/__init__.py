"""
交易执行模块
"""

from .execution_engine import (
    ExecutionConfig,
    OrderEvent,
    OrderEventData,
    ExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    IcebergAlgorithm,
    BrokerGateway,
    CTPGateway,
    TqsdkGateway,
    OrderManager,
    ExecutionEngine
)

from .auto_trader import (
    TradingSession,
    TRADING_HOURS,
    AutoTraderConfig,
    SignalProcessor,
    AutoTrader,
    TradeScheduler
)

__all__ = [
    # 执行引擎
    'ExecutionConfig',
    'OrderEvent',
    'OrderEventData',
    'ExecutionAlgorithm',
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    'IcebergAlgorithm',
    'BrokerGateway',
    'CTPGateway',
    'TqsdkGateway',
    'OrderManager',
    'ExecutionEngine',
    
    # 自动交易
    'TradingSession',
    'TRADING_HOURS',
    'AutoTraderConfig',
    'SignalProcessor',
    'AutoTrader',
    'TradeScheduler',
]
