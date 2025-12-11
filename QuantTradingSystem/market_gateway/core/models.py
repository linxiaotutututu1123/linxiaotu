"""
行情数据模型定义。

包含：
- TickData: 逐笔行情数据
- SnapshotData: 1秒快照数据
- GatewayConfig: 网关配置
- ReconnectConfig: 重连策略配置

# RISK: Decimal精度需与交易所一致，否则可能出现价格匹配失败
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, SecretStr, ConfigDict, model_validator


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


class ExchangeType(str, Enum):
    """
    交易所类型枚举。
    
    # 为什么用str继承：便于JSON序列化和日志输出
    """
    CFFEX = "CFFEX"      # 中金所
    SHFE = "SHFE"        # 上期所
    DCE = "DCE"          # 大商所
    CZCE = "CZCE"        # 郑商所
    INE = "INE"          # 上期能源
    GFEX = "GFEX"        # 广期所


class TimestampSource(str, Enum):
    """
    时间戳来源枚举。
    
    标识raw_timestamp的格式，便于解析时选择正确的解析器。
    
    # 为什么需要这个：不同柜台时间戳格式差异大
    # CTP: "HH:MM:SS" + 毫秒整数
    # 飞马: Unix微秒时间戳
    # 易盛: 自定义格式
    """
    CTP = "ctp"          # CTP格式: "HH:MM:SS|millisec"
    FEMAS = "femas"      # 飞马格式: Unix微秒时间戳
    ESUNNY = "esunny"    # 易盛格式: 待定义
    CUSTOM = "custom"    # 自定义格式（需额外说明）


class TickData(BaseModel):
    """
    逐笔行情数据（不可变值对象）。
    
    表示某一时刻某合约的完整行情快照。
    
    Attributes:
        symbol: 合约代码，如 "IF2312"
        exchange: 交易所代码
        last_price: 最新成交价
        volume: 累计成交量
        timestamp: 行情时间戳（交易所时间）
    
    # RISK: 不同柜台时间戳精度不同（CTP毫秒/飞马微秒）
    """
    
    # 为什么使用frozen：行情数据一旦生成不应被修改，防止策略层误操作
    model_config = ConfigDict(frozen=True)
    
    # === 数据版本控制 ===
    # 为什么需要版本：便于数据迁移和向后兼容
    # RISK: 版本升级时需要编写迁移脚本
    schema_version: int = Field(
        default=1,
        ge=1,
        description="数据模式版本号，用于兼容性检查",
    )
    
    # === 基础标识字段 ===
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="合约代码",
        examples=["IF2312", "rb2401"],
    )
    
    exchange: str = Field(
        ...,
        min_length=1,
        max_length=16,
        description="交易所代码",
        examples=["CFFEX", "SHFE"],
    )
    
    # === 价格字段（使用Decimal保证精度） ===
    # 为什么用Decimal：浮点数运算会丢失精度，金融场景必须精确
    last_price: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="最新成交价",
    )
    
    # === 量价字段 ===
    volume: int = Field(
        ...,
        ge=0,
        description="累计成交量",
    )
    
    open_interest: int = Field(
        default=0,
        ge=0,
        description="持仓量",
    )
    
    # === 时间戳 ===
    # RISK: 交易所时间可能与本地时间不同步
    timestamp: datetime = Field(
        ...,
        description="行情时间戳（交易所时间）",
    )
    
    # 为什么保留原始时间戳：不同柜台精度不同，防止信息丢失
    # CTP: "10:30:00" + 500(ms)  飞马: 1702267800123456(μs)
    raw_timestamp: Optional[str] = Field(
        default=None,
        description="柜台原始时间戳字符串（用于审计和精确回放）",
    )
    
    # 为什么需要来源标识：解析raw_timestamp时需要知道格式
    # RISK: 如果来源与格式不匹配，解析会失败
    timestamp_source: Optional[str] = Field(
        default=None,
        pattern=r"^(ctp|femas|esunny|custom)$",
        description="时间戳来源：ctp/femas/esunny/custom",
    )
    
    # 为什么记录本地时间：用于计算网络延迟
    local_timestamp: Optional[datetime] = Field(
        default=None,
        description="本地接收时间戳",
    )
    
    # === 盘口数据（一档） ===
    # 为什么只保留一档：减少内存占用，深度数据另存
    bid_price_1: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="买一价",
    )
    
    ask_price_1: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="卖一价",
    )
    
    bid_volume_1: Optional[int] = Field(
        default=None,
        ge=0,
        description="买一量",
    )
    
    ask_volume_1: Optional[int] = Field(
        default=None,
        ge=0,
        description="卖一量",
    )
    
    def __repr__(self) -> str:
        """
        返回便于调试的字符串表示。
        
        # 为什么自定义__repr__：快速查看关键信息而非全部字段
        """
        return (
            f"TickData("
            f"symbol={self.symbol!r}, "
            f"price={self.last_price}, "
            f"vol={self.volume}, "
            f"time={self.timestamp.strftime('%H:%M:%S.%f')}"
            f")"
        )
    
    @model_validator(mode='after')
    def validate_timestamp_consistency(self) -> 'TickData':
        """
        验证时间戳字段一致性。
        
        规则：
        - 如果有 timestamp_source，则必须有 raw_timestamp
        - 反过来可以（兼容未标记来源的历史数据）
        
        # 为什么这样设计：有来源标签却没有原始数据是逻辑矛盾
        """
        if self.timestamp_source is not None and self.raw_timestamp is None:
            raise ValueError(
                "timestamp_source requires raw_timestamp: "
                "if you specify a source, you must provide the raw timestamp"
            )
        return self


# ============================================================
# A. 拆分后的轻量级数据结构
# ============================================================

class TickDataCore(BaseModel):
    """
    核心行情数据（轻量级，高频使用）。
    
    只包含策略层最常用的字段，减少内存占用。
    适用场景：高频回测、实时信号计算。
    
    # 为什么拆分：原TickData 20+字段，高频场景内存压力大
    # RISK: 使用时需确认是否需要深度数据
    """
    
    model_config = ConfigDict(frozen=True)
    
    # === 必要字段（仅6个） ===
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="合约代码",
    )
    
    exchange: str = Field(
        ...,
        min_length=1,
        max_length=16,
        description="交易所代码",
    )
    
    last_price: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="最新成交价",
    )
    
    volume: int = Field(
        ...,
        ge=0,
        description="累计成交量",
    )
    
    timestamp: datetime = Field(
        ...,
        description="行情时间戳",
    )
    
    open_interest: int = Field(
        default=0,
        ge=0,
        description="持仓量",
    )
    
    def __repr__(self) -> str:
        """简洁的调试输出。"""
        return (
            f"TickDataCore({self.symbol} "
            f"price={self.last_price} vol={self.volume})"
        )


class TickDataDepth(BaseModel):
    """
    盘口深度数据（按需加载）。
    
    包含多档买卖盘信息，用于需要完整盘口的策略。
    
    # 为什么独立：深度数据占用大，很多策略不需要
    # RISK: 多档数据需与交易所档位对齐
    """
    
    model_config = ConfigDict(frozen=True)
    
    # === 关联标识 ===
    symbol: str = Field(..., min_length=1, description="合约代码")
    exchange: str = Field(..., min_length=1, description="交易所代码")
    timestamp: datetime = Field(..., description="盘口时间戳")
    
    # === 一档盘口 ===
    bid_price_1: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    ask_price_1: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    bid_volume_1: Optional[int] = Field(default=None, ge=0)
    ask_volume_1: Optional[int] = Field(default=None, ge=0)
    
    # === 二档盘口 ===
    bid_price_2: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    ask_price_2: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    bid_volume_2: Optional[int] = Field(default=None, ge=0)
    ask_volume_2: Optional[int] = Field(default=None, ge=0)
    
    # === 三档盘口 ===
    bid_price_3: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    ask_price_3: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    bid_volume_3: Optional[int] = Field(default=None, ge=0)
    ask_volume_3: Optional[int] = Field(default=None, ge=0)
    
    # === 四档盘口 ===
    bid_price_4: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    ask_price_4: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    bid_volume_4: Optional[int] = Field(default=None, ge=0)
    ask_volume_4: Optional[int] = Field(default=None, ge=0)
    
    # === 五档盘口 ===
    bid_price_5: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    ask_price_5: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    bid_volume_5: Optional[int] = Field(default=None, ge=0)
    ask_volume_5: Optional[int] = Field(default=None, ge=0)
    
    def __repr__(self) -> str:
        """调试输出。"""
        return (
            f"TickDataDepth({self.symbol} "
            f"bid1={self.bid_price_1} ask1={self.ask_price_1})"
        )


# ============================================================
# B. 价格类型配置
# ============================================================

class PriceConfig(BaseModel):
    """
    价格类型配置。
    
    控制系统使用 Decimal 还是 float 存储价格。
    
    # 为什么可配置：Decimal精确但慢，float快但有精度风险
    # RISK: 使用float时价格比较可能出现浮点误差
    """
    
    use_decimal: bool = Field(
        default=True,
        description="是否使用Decimal存储价格（True=精确，False=高性能）",
    )
    
    # 为什么添加警告：提醒使用float的用户注意精度问题
    @property
    def precision_warning(self) -> Optional[str]:
        """返回精度警告信息。"""
        if not self.use_decimal:
            return (
                "WARNING: float模式可能导致价格比较错误，"
                "请勿用于生产交易，仅限高频回测"
            )
        return None
    
    def __repr__(self) -> str:
        mode = "Decimal" if self.use_decimal else "float"
        return f"PriceConfig(mode={mode})"


# ============================================================
# C. 计算属性视图
# ============================================================

class TickDataView:
    """
    TickData的可变视图，提供计算属性。
    
    底层TickData保持不可变，视图提供：
    - mid_price: 中间价
    - spread: 买卖价差
    - spread_bps: 价差基点
    
    # 为什么用视图：frozen的TickData无法添加属性
    # 为什么不继承：避免破坏不可变性语义
    """
    
    __slots__ = ('_tick',)  # 为什么用slots：减少内存开销
    
    def __init__(self, tick: TickData) -> None:
        """
        初始化视图。
        
        Args:
            tick: 底层的不可变TickData对象
        """
        self._tick = tick
    
    # === 委托属性（直接访问底层tick） ===
    
    @property
    def symbol(self) -> str:
        """合约代码。"""
        return self._tick.symbol
    
    @property
    def exchange(self) -> str:
        """交易所代码。"""
        return self._tick.exchange
    
    @property
    def last_price(self) -> Decimal:
        """最新成交价。"""
        return self._tick.last_price
    
    @property
    def volume(self) -> int:
        """成交量。"""
        return self._tick.volume
    
    @property
    def timestamp(self) -> datetime:
        """时间戳。"""
        return self._tick.timestamp
    
    @property
    def bid_price_1(self) -> Optional[Decimal]:
        """买一价。"""
        return self._tick.bid_price_1
    
    @property
    def ask_price_1(self) -> Optional[Decimal]:
        """卖一价。"""
        return self._tick.ask_price_1
    
    # === 计算属性 ===
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """
        中间价 = (买一价 + 卖一价) / 2。
        
        # 为什么计算中间价：用于估算公允价格和滑点
        
        Returns:
            中间价，如果买卖价不完整则返回None
        """
        if self._tick.bid_price_1 is None or self._tick.ask_price_1 is None:
            return None
        return (self._tick.bid_price_1 + self._tick.ask_price_1) / 2
    
    @property
    def spread(self) -> Optional[Decimal]:
        """
        买卖价差 = 卖一价 - 买一价。
        
        # 为什么计算价差：流动性指标，价差越小流动性越好
        
        Returns:
            价差，如果买卖价不完整则返回None
        """
        if self._tick.bid_price_1 is None or self._tick.ask_price_1 is None:
            return None
        return self._tick.ask_price_1 - self._tick.bid_price_1
    
    @property
    def spread_bps(self) -> Optional[Decimal]:
        """
        价差基点 = (卖一 - 买一) / 中间价 * 10000。
        
        # 为什么用基点：便于跨品种比较流动性
        # 1个基点 = 0.01%
        
        Returns:
            价差基点，如果无法计算则返回None
        """
        mid = self.mid_price
        if mid is None or mid == 0:
            return None
        spread = self.spread
        if spread is None:
            return None
        return (spread / mid) * Decimal("10000")
    
    def __repr__(self) -> str:
        """调试输出。"""
        mid = self.mid_price
        mid_str = f"{mid:.2f}" if mid else "N/A"
        spread = self.spread_bps
        spread_str = f"{spread:.2f}bps" if spread else "N/A"
        return (
            f"TickDataView({self.symbol} "
            f"mid={mid_str} spread={spread_str})"
        )


class ReconnectConfig(BaseModel):
    """
    重连策略配置。
    
    三阶段策略：
    1. 立即重试阶段：快速尝试恢复连接
    2. 指数退避阶段：避免重连风暴
    3. 定时检测阶段：低频保持探测
    
    # RISK: 参数设置不当可能导致恢复过慢或压垮服务器
    """
    
    # === 立即重试阶段 ===
    immediate_retry_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="立即重试次数",
    )
    
    # 为什么限制上限：防止配置错误导致无限快速重试
    immediate_retry_interval_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="立即重试间隔（毫秒）",
    )
    
    # === 指数退避阶段 ===
    backoff_base_seconds: float = Field(
        default=1.0,
        gt=0,
        description="退避基数（秒）",
    )
    
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="退避乘数",
    )
    
    max_backoff_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300.0,
        description="最大退避时间（秒）",
    )
    
    # === 定时检测阶段 ===
    # 为什么需要定时检测：网络恢复后能自动重连
    scheduled_check_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="定时检测间隔（秒）",
    )


class GatewayConfig(BaseModel):
    """
    行情网关配置。
    
    包含柜台连接参数、认证信息、超时设置等。
    所有敏感信息应从环境变量加载，禁止硬编码。
    
    # RISK: 密码泄露风险，生产环境必须加密存储
    """
    
    # === 柜台连接信息 ===
    broker_id: str = Field(
        ...,
        min_length=1,
        max_length=16,
        description="经纪商代码",
    )
    
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="用户ID",
    )
    
    # 为什么用SecretStr：防止密码在日志/序列化中意外泄露
    # RISK: 调用get_secret_value()后仍需注意不要打印
    password: SecretStr = Field(
        ...,
        min_length=1,
        description="密码（应从环境变量加载）",
    )
    
    front_addr: str = Field(
        ...,
        pattern=r"^tcp://[\w\.\-]+:\d+$",
        description="前置地址，格式: tcp://ip:port",
        examples=["tcp://180.168.146.187:10131"],
    )
    
    # === 超时设置 ===
    # 为什么必须设置超时：防止网络异常时无限阻塞
    connect_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60.0,
        description="连接超时（秒）",
    )
    
    request_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=30.0,
        description="请求超时（秒）",
    )
    
    # === 重连配置 ===
    reconnect: ReconnectConfig = Field(
        default_factory=ReconnectConfig,
        description="重连策略配置",
    )
    
    def __repr__(self) -> str:
        """
        安全的字符串表示（隐藏密码）。
        
        # 为什么隐藏密码：防止日志泄露敏感信息
        """
        return (
            f"GatewayConfig("
            f"broker={self.broker_id!r}, "
            f"user={self.user_id!r}, "
            f"addr={self.front_addr!r}"
            f")"
        )


class SnapshotData(BaseModel):
    """
    1秒快照数据（聚合后的行情数据）。
    
    由多条TickData聚合生成，用于降低存储压力。
    
    # RISK: 聚合过程可能丢失极端价格信息
    """
    
    model_config = ConfigDict(frozen=True)
    
    symbol: str = Field(..., min_length=1, description="合约代码")
    exchange: str = Field(..., min_length=1, description="交易所代码")
    
    # === OHLC价格 ===
    # 为什么记录OHLC：捕获1秒内的完整价格波动
    open_price: Decimal = Field(..., ge=Decimal("0"), description="开盘价")
    high_price: Decimal = Field(..., ge=Decimal("0"), description="最高价")
    low_price: Decimal = Field(..., ge=Decimal("0"), description="最低价")
    close_price: Decimal = Field(..., ge=Decimal("0"), description="收盘价")
    
    # === 成交量 ===
    volume: int = Field(..., ge=0, description="1秒内成交量")
    tick_count: int = Field(..., ge=0, description="聚合的Tick数量")
    
    # === 时间戳 ===
    # 为什么用秒级时间戳：快照是秒级数据
    timestamp: datetime = Field(..., description="快照时间（秒级对齐）")
    
    def __repr__(self) -> str:
        """返回便于调试的字符串表示。"""
        return (
            f"SnapshotData("
            f"{self.symbol} "
            f"O={self.open_price} H={self.high_price} "
            f"L={self.low_price} C={self.close_price} "
            f"V={self.volume}"
            f")"
        )
