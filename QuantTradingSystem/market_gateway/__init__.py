"""
行情网关模块。

职责：对接多个交易所柜台，提供统一的行情数据接口。
支持柜台：CTP / 飞马 / 易盛

# RISK: 各柜台SDK版本更新可能导致兼容性问题
"""

from market_gateway.core.models import (
    ExchangeType,
    TimestampSource,
    TickData,
    TickDataCore,
    TickDataDepth,
    TickDataView,
    PriceConfig,
    SnapshotData,
    GatewayConfig,
    ReconnectConfig,
)

__all__ = [
    "ExchangeType",
    "TimestampSource",
    "TickData",
    "TickDataCore",
    "TickDataDepth",
    "TickDataView",
    "PriceConfig",
    "SnapshotData",
    "GatewayConfig",
    "ReconnectConfig",
]
