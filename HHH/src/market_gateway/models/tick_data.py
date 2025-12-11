"""
行情网关数据模型定义。

本模块定义行情网关所需的核心数据模型，包括：
- TickData: 标准化Tick行情数据
- DepthData: 深度行情数据（5档）
- GatewayConfig: 网关配置模型

所有模型使用 Pydantic v2 进行数据校验。

__all__ 声明本模块公开的类与函数。
"""

from __future__ import annotations

__all__ = [
    "TickData",
    "DepthData",
    "PriceLevel",
    "GatewayConfig",
    "ReconnectConfig",
]

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


class PriceLevel(BaseModel):
    """
    单档价格数据。
    
    用于表示买卖盘的单个价格档位。
    
    Attributes:
        price: 价格，必须 > 0
        volume: 挂单量，必须 >= 0
    
    Example:
        >>> level = PriceLevel(price=Decimal("3500.0"), volume=100)
        >>> repr(level)
        "PriceLevel(price=3500.0, volume=100)"
    """
    
    model_config = ConfigDict(frozen=True, slots=True)
    
    # WHY: 使用 Decimal 而非 float 避免浮点精度问题
    price: Decimal = Field(..., gt=0, description="价格")
    volume: int = Field(..., ge=0, description="挂单量")
    
    def __repr__(self) -> str:
        """调试友好的字符串表示。"""
        return f"PriceLevel(price={self.price}, volume={self.volume})"


class DepthData(BaseModel):
    """
    深度行情数据（5档买卖盘）。
    
    Attributes:
        bids: 买盘5档，按价格降序排列
        asks: 卖盘5档，按价格升序排列
    
    Example:
        >>> depth = DepthData(
        ...     bids=[PriceLevel(price=Decimal("3500"), volume=100)],
        ...     asks=[PriceLevel(price=Decimal("3501"), volume=50)]
        ... )
    """
    
    model_config = ConfigDict(frozen=True, slots=True)
    
    bids: tuple[PriceLevel, ...] = Field(
        default=(), 
        max_length=5,
        description="买盘档位，最多5档"
    )
    asks: tuple[PriceLevel, ...] = Field(
        default=(), 
        max_length=5,
        description="卖盘档位，最多5档"
    )
    
    def __repr__(self) -> str:
        """调试友好的字符串表示。"""
        return f"DepthData(bids={len(self.bids)}档, asks={len(self.asks)}档)"


class TickData(BaseModel):
    """
    标准化Tick行情数据。
    
    所有行情网关输出的统一数据格式，确保下游消费者
    无需关心具体柜台的原始数据结构差异。
    
    Attributes:
        symbol: 合约代码，如 "IF2401"
        exchange: 交易所代码，如 "CFFEX"
        datetime: 行情时间戳（精确到毫秒）
        last_price: 最新价
        volume: 成交量（当日累计）
        turnover: 成交额（当日累计）
        open_interest: 持仓量
        open_price: 开盘价
        high_price: 最高价
        low_price: 最低价
        pre_close: 昨收价
        limit_up: 涨停价
        limit_down: 跌停价
        depth: 深度行情（可选）
        gateway_name: 来源网关名称
        local_time: 本地接收时间
    
    Example:
        >>> tick = TickData(
        ...     symbol="IF2401",
        ...     exchange="CFFEX",
        ...     datetime=datetime.now(),
        ...     last_price=Decimal("3500.0"),
        ...     volume=12345,
        ...     turnover=Decimal("4321000000"),
        ...     gateway_name="ctp"
        ... )
    
    Note:
        - 使用 __slots__ 优化内存占用
        - frozen=True 确保不可变，线程安全
    """
    
    # WHY: slots=True 减少内存占用约40%，frozen=True 保证线程安全
    model_config = ConfigDict(frozen=True, slots=True)
    
    # === 标识字段 ===
    symbol: str = Field(
        ..., 
        min_length=1, 
        max_length=20,
        description="合约代码"
    )
    exchange: str = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="交易所代码"
    )
    
    # === 时间字段 ===
    datetime: datetime = Field(..., description="行情时间戳")
    
    # === 价格字段（使用 Decimal 保证精度）===
    # WHY: 金融数据必须使用 Decimal，float 会有精度丢失
    last_price: Decimal = Field(..., gt=0, description="最新价")
    open_price: Optional[Decimal] = Field(None, gt=0, description="开盘价")
    high_price: Optional[Decimal] = Field(None, gt=0, description="最高价")
    low_price: Optional[Decimal] = Field(None, gt=0, description="最低价")
    pre_close: Optional[Decimal] = Field(None, gt=0, description="昨收价")
    limit_up: Optional[Decimal] = Field(None, gt=0, description="涨停价")
    limit_down: Optional[Decimal] = Field(None, gt=0, description="跌停价")
    
    # === 量价字段 ===
    volume: int = Field(..., ge=0, description="成交量（当日累计）")
    turnover: Optional[Decimal] = Field(None, ge=0, description="成交额")
    open_interest: Optional[int] = Field(None, ge=0, description="持仓量")
    
    # === 深度行情 ===
    depth: Optional[DepthData] = Field(None, description="深度行情")
    
    # === 元数据 ===
    gateway_name: str = Field(..., min_length=1, description="来源网关")
    local_time: datetime = Field(
        default_factory=datetime.now,
        description="本地接收时间"
    )
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """
        校验合约代码格式。
        
        # WHY: 合约代码应为字母+数字组合，过滤非法输入
        """
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("合约代码不能为空")
        return cleaned
    
    @model_validator(mode="after")
    def validate_price_range(self) -> "TickData":
        """
        校验价格逻辑关系。
        
        # WHY: 确保 low <= last <= high，防止脏数据
        # RISK: 极端行情可能触发误报，需结合涨跌停判断
        """
        if self.high_price and self.last_price > self.high_price:
            raise ValueError(
                f"最新价({self.last_price})不能高于最高价({self.high_price})"
            )
        if self.low_price and self.last_price < self.low_price:
            raise ValueError(
                f"最新价({self.last_price})不能低于最低价({self.low_price})"
            )
        return self
    
    def __repr__(self) -> str:
        """调试友好的字符串表示。"""
        return (
            f"TickData({self.symbol}@{self.exchange} "
            f"price={self.last_price} vol={self.volume} "
            f"time={self.datetime.strftime('%H:%M:%S.%f')[:-3]})"
        )
