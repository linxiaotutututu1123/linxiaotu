"""
数据层 - 核心数据结构定义
定义Tick、Bar、Order、Position等基础数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


# ==================== 枚举类型定义 ====================

class Direction(Enum):
    """交易方向"""
    LONG = "多"
    SHORT = "空"

class Offset(Enum):
    """开平仓标志"""
    OPEN = "开仓"
    CLOSE = "平仓"
    CLOSE_TODAY = "平今"
    CLOSE_YESTERDAY = "平昨"

class OrderType(Enum):
    """订单类型"""
    LIMIT = "限价单"
    MARKET = "市价单"
    STOP = "止损单"
    STOP_LIMIT = "止损限价单"
    FAK = "即成剩撤"
    FOK = "全成全撤"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "待提交"
    SUBMITTED = "已提交"
    PARTIAL_FILLED = "部分成交"
    FILLED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "已拒绝"
    ERROR = "错误"

class StrategyStatus(Enum):
    """策略状态"""
    STOPPED = "已停止"
    RUNNING = "运行中"
    PAUSED = "已暂停"
    ERROR = "异常"


# ==================== 核心数据结构 ====================

@dataclass
class TickData:
    """
    Tick数据 - 逐笔行情
    包含最新价、成交量、持仓量、盘口五档等信息
    """
    symbol: str                           # 合约代码 如 rb2401
    exchange: str                         # 交易所
    datetime: datetime                    # 时间戳
    
    # 价格信息
    last_price: float                     # 最新价
    open_price: float                     # 开盘价
    high_price: float                     # 最高价
    low_price: float                      # 最低价
    pre_close: float                      # 昨收价
    pre_settlement: float                 # 昨结算价
    
    # 成交信息
    volume: int                           # 成交量
    turnover: float                       # 成交额
    open_interest: int                    # 持仓量
    
    # 涨跌停
    upper_limit: float                    # 涨停价
    lower_limit: float                    # 跌停价
    
    # 盘口数据 - 买方
    bid_price_1: float = 0.0
    bid_price_2: float = 0.0
    bid_price_3: float = 0.0
    bid_price_4: float = 0.0
    bid_price_5: float = 0.0
    bid_volume_1: int = 0
    bid_volume_2: int = 0
    bid_volume_3: int = 0
    bid_volume_4: int = 0
    bid_volume_5: int = 0
    
    # 盘口数据 - 卖方
    ask_price_1: float = 0.0
    ask_price_2: float = 0.0
    ask_price_3: float = 0.0
    ask_price_4: float = 0.0
    ask_price_5: float = 0.0
    ask_volume_1: int = 0
    ask_volume_2: int = 0
    ask_volume_3: int = 0
    ask_volume_4: int = 0
    ask_volume_5: int = 0
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        if self.ask_price_1 > 0 and self.bid_price_1 > 0:
            return self.ask_price_1 - self.bid_price_1
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """中间价"""
        if self.ask_price_1 > 0 and self.bid_price_1 > 0:
            return (self.ask_price_1 + self.bid_price_1) / 2
        return self.last_price


@dataclass
class BarData:
    """
    K线数据
    支持各种周期: 1分钟、5分钟、15分钟、30分钟、1小时、日线等
    """
    symbol: str                           # 合约代码
    exchange: str                         # 交易所
    datetime: datetime                    # K线时间
    interval: str                         # 周期: 1m, 5m, 15m, 30m, 1h, 1d
    
    # OHLCV数据
    open: float                           # 开盘价
    high: float                           # 最高价
    low: float                            # 最低价
    close: float                          # 收盘价
    volume: int                           # 成交量
    turnover: float                       # 成交额
    open_interest: int                    # 持仓量
    
    # 额外信息
    settlement: float = 0.0               # 结算价
    
    @property
    def range(self) -> float:
        """振幅"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """实体"""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """是否阳线"""
        return self.close > self.open
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "datetime": self.datetime.isoformat(),
            "interval": self.interval,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "turnover": self.turnover,
            "open_interest": self.open_interest
        }


@dataclass
class OrderData:
    """
    订单数据
    """
    order_id: str                         # 订单ID
    symbol: str                           # 合约代码
    exchange: str                         # 交易所
    direction: Direction                  # 方向
    offset: Offset                        # 开平
    price: float                          # 委托价格
    volume: int                           # 委托数量
    
    order_type: OrderType = OrderType.LIMIT  # 订单类型
    status: OrderStatus = OrderStatus.PENDING  # 订单状态
    traded: int = 0                       # 已成交数量
    remaining: int = 0                    # 剩余数量
    
    create_time: Optional[datetime] = None  # 创建时间
    update_time: Optional[datetime] = None  # 更新时间
    
    strategy_name: str = ""               # 策略名称
    remark: str = ""                      # 备注
    
    # 成交信息
    avg_price: float = 0.0                # 成交均价
    commission: float = 0.0               # 手续费
    slippage: float = 0.0                 # 滑点
    
    def __post_init__(self):
        self.remaining = self.volume - self.traded
        if self.create_time is None:
            self.create_time = datetime.now()


@dataclass
class TradeData:
    """
    成交数据
    """
    trade_id: str                         # 成交ID
    order_id: str                         # 订单ID
    symbol: str                           # 合约代码
    exchange: str                         # 交易所
    direction: Direction                  # 方向
    offset: Offset                        # 开平
    price: float                          # 成交价格
    volume: int                           # 成交数量
    datetime: datetime                    # 成交时间
    
    commission: float = 0.0               # 手续费
    strategy_name: str = ""               # 策略名称


@dataclass
class PositionData:
    """
    持仓数据
    """
    symbol: str                           # 合约代码
    exchange: str                         # 交易所
    direction: Direction                  # 方向
    
    volume: int = 0                       # 持仓量
    frozen: int = 0                       # 冻结量
    available: int = 0                    # 可用量
    
    price: float = 0.0                    # 持仓均价
    pnl: float = 0.0                      # 持仓盈亏
    
    # 今昨仓（区分上期所等需要区分今昨仓的交易所）
    yd_volume: int = 0                    # 昨仓
    td_volume: int = 0                    # 今仓
    
    # 浮动盈亏
    unrealized_pnl: float = 0.0           # 未实现盈亏
    realized_pnl: float = 0.0             # 已实现盈亏
    
    def __post_init__(self):
        self.available = self.volume - self.frozen


@dataclass
class AccountData:
    """
    账户资金数据
    """
    account_id: str                       # 账户ID
    
    balance: float = 0.0                  # 总资产
    available: float = 0.0                # 可用资金
    frozen: float = 0.0                   # 冻结资金
    
    margin: float = 0.0                   # 占用保证金
    commission: float = 0.0               # 今日手续费
    
    close_profit: float = 0.0             # 平仓盈亏
    position_profit: float = 0.0          # 持仓盈亏
    
    pre_balance: float = 0.0              # 昨日权益
    deposit: float = 0.0                  # 今日入金
    withdraw: float = 0.0                 # 今日出金
    
    @property
    def total_pnl(self) -> float:
        """总盈亏"""
        return self.close_profit + self.position_profit
    
    @property
    def daily_return(self) -> float:
        """日收益率"""
        if self.pre_balance > 0:
            return self.total_pnl / self.pre_balance
        return 0.0
    
    @property
    def margin_ratio(self) -> float:
        """保证金占用比例"""
        if self.balance > 0:
            return self.margin / self.balance
        return 0.0


@dataclass
class ContractData:
    """
    合约信息
    """
    symbol: str                           # 合约代码
    exchange: str                         # 交易所
    name: str                             # 合约名称
    product: str                          # 品种代码
    
    size: float                           # 合约乘数
    pricetick: float                      # 最小价格变动
    
    margin_ratio: float = 0.0             # 保证金比例
    commission_ratio: float = 0.0         # 手续费率
    commission_fixed: float = 0.0         # 固定手续费
    
    open_date: str = ""                   # 上市日期
    expire_date: str = ""                 # 到期日期
    
    # 涨跌停
    upper_limit: float = 0.0              # 涨停价
    lower_limit: float = 0.0              # 跌停价


@dataclass
class SignalData:
    """
    交易信号数据
    """
    signal_id: str                        # 信号ID
    strategy_name: str                    # 策略名称
    symbol: str                           # 合约代码
    datetime: datetime                    # 信号时间
    
    direction: Direction                  # 方向
    strength: float                       # 信号强度 -1.0 ~ 1.0
    
    entry_price: float = 0.0              # 入场价格
    stop_loss: float = 0.0                # 止损价格
    take_profit: float = 0.0              # 止盈价格
    
    confidence: float = 0.0               # 置信度 0.0 ~ 1.0
    reason: str = ""                      # 信号原因


@dataclass
class PerformanceMetrics:
    """
    绩效指标
    """
    # 收益指标
    total_return: float = 0.0             # 总收益率
    annual_return: float = 0.0            # 年化收益率
    monthly_return: float = 0.0           # 月度收益率
    daily_return_mean: float = 0.0        # 日均收益率
    daily_return_std: float = 0.0         # 日收益率标准差
    
    # 风险指标
    max_drawdown: float = 0.0             # 最大回撤
    max_drawdown_duration: int = 0        # 最大回撤持续天数
    var_95: float = 0.0                   # 95% VaR
    var_99: float = 0.0                   # 99% VaR
    
    # 风险调整收益
    sharpe_ratio: float = 0.0             # 夏普比率
    sortino_ratio: float = 0.0            # 索提诺比率
    calmar_ratio: float = 0.0             # 卡尔马比率
    information_ratio: float = 0.0        # 信息比率
    
    # 交易统计
    total_trades: int = 0                 # 总交易次数
    winning_trades: int = 0               # 盈利次数
    losing_trades: int = 0                # 亏损次数
    win_rate: float = 0.0                 # 胜率
    
    # 盈亏分析
    avg_profit: float = 0.0               # 平均盈利
    avg_loss: float = 0.0                 # 平均亏损
    profit_factor: float = 0.0            # 盈亏比
    max_consecutive_wins: int = 0         # 最大连胜次数
    max_consecutive_losses: int = 0       # 最大连亏次数
    
    # 持仓分析
    avg_holding_period: float = 0.0       # 平均持仓时间（分钟）
    avg_position_size: float = 0.0        # 平均仓位
    
    def is_valid(self) -> bool:
        """检查绩效是否达标"""
        return (
            self.sharpe_ratio >= 1.5 and
            self.max_drawdown <= 0.20 and
            self.win_rate >= 0.45 and
            self.profit_factor >= 1.5
        )


# ==================== 数据容器 ====================

class DataBuffer:
    """
    高性能数据缓冲区
    用于存储最近N个Bar或Tick数据
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._data: List[Any] = []
        
    def append(self, item: Any):
        """添加数据"""
        self._data.append(item)
        if len(self._data) > self.max_size:
            self._data.pop(0)
    
    def get(self, index: int) -> Optional[Any]:
        """获取指定位置数据"""
        if -len(self._data) <= index < len(self._data):
            return self._data[index]
        return None
    
    def get_last(self, n: int = 1) -> List[Any]:
        """获取最近N条数据"""
        return self._data[-n:]
    
    def get_array(self, field: str) -> np.ndarray:
        """获取指定字段的numpy数组"""
        values = [getattr(item, field) for item in self._data if hasattr(item, field)]
        return np.array(values)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int):
        return self._data[index]
    
    def clear(self):
        """清空数据"""
        self._data.clear()
