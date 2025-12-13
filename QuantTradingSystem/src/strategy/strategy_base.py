"""
策略框架 - 策略基类和常用策略模板
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..data.data_structures import (
    BarData, TickData, OrderData, TradeData, PositionData, AccountData,
    Direction, Offset, OrderType, SignalData, DataBuffer
)

logger = logging.getLogger(__name__)


# ==================== 策略基类 ====================

class BaseStrategy(ABC):
    """
    策略基类
    所有策略都应继承此类
    增强版：包含信号过滤、趋势确认、风险管理等功能
    """
    
    # 策略元信息
    strategy_name: str = "BaseStrategy"
    author: str = "QuantMaster"
    version: str = "2.0.0"
    
    def __init__(self):
        self.engine = None  # 回测/交易引擎引用
        self.parameters: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        
        # 数据缓冲
        self._bar_buffers: Dict[str, DataBuffer] = {}
        self._indicator_cache: Dict[str, np.ndarray] = {}
        
        # 策略状态
        self.is_trading = False
        self.positions: Dict[str, int] = {}  # symbol -> volume (正数多头，负数空头)
        
        # 统计
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # 信号过滤器配置
        self._signal_filter_enabled = True
        self._min_signal_interval = 5  # 最小信号间隔(K线数)
        self._last_signal_bar: Dict[str, int] = {}  # 记录上次信号的K线序号
        self._bar_counter: Dict[str, int] = {}  # K线计数器
        
        # 趋势确认配置
        self._trend_filter_enabled = True
        self._trend_period = 50  # 趋势判断周期
        
        # 波动率过滤
        self._volatility_filter_enabled = True
        self._min_volatility = 0.005  # 最低波动率(避免震荡行情)
        self._max_volatility = 0.08   # 最高波动率(避免极端行情)
        
        # 追踪止损
        self._trailing_stops: Dict[str, Dict] = {}  # symbol -> {entry_price, highest/lowest, stop_price}
        self._atr_stop_multiple = 2.0  # ATR止损倍数
        
        # 盈亏统计(用于动态调整)
        self._recent_trades: List[float] = []  # 最近交易盈亏
        self._max_recent_trades = 20
    
    def on_init(self):
        """
        策略初始化
        在回测开始前调用
        """
        logger.info(f"Strategy {self.strategy_name} initialized")
    
    @abstractmethod
    def on_bar(self, bars: Dict[str, BarData]):
        """
        K线数据更新回调
        核心策略逻辑在此实现
        
        Args:
            bars: 各品种的当前K线数据
        """
        pass
    
    def on_tick(self, tick: TickData):
        """
        Tick数据更新回调（高频策略使用）
        """
        pass
    
    def on_trade(self, trade: TradeData):
        """
        成交回调
        """
        self.trade_count += 1
        logger.info(f"Trade: {trade.symbol} {trade.direction.value} {trade.offset.value} "
                   f"price={trade.price} volume={trade.volume}")
    
    def on_order(self, order: OrderData):
        """
        订单状态更新回调
        """
        pass
    
    def on_stop(self):
        """
        策略停止回调
        """
        logger.info(f"Strategy {self.strategy_name} stopped")
    
    # ==================== 交易接口 ====================
    
    def buy(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT,
        stop_loss: float = 0,
        take_profit: float = 0
    ) -> str:
        """
        买入开多
        """
        return self.send_order(
            symbol, Direction.LONG, Offset.OPEN,
            price, volume, order_type
        )
    
    def sell(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> str:
        """
        卖出平多
        """
        return self.send_order(
            symbol, Direction.LONG, Offset.CLOSE,
            price, volume, order_type
        )
    
    def short(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> str:
        """
        卖出开空
        """
        return self.send_order(
            symbol, Direction.SHORT, Offset.OPEN,
            price, volume, order_type
        )
    
    def cover(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> str:
        """
        买入平空
        """
        return self.send_order(
            symbol, Direction.SHORT, Offset.CLOSE,
            price, volume, order_type
        )
    
    def send_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> str:
        """
        发送订单
        """
        if not self.engine:
            logger.error("Engine not set")
            return ""
        
        return self.engine.send_order(
            symbol=symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            order_type=order_type,
            strategy_name=self.strategy_name
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单
        """
        if not self.engine:
            return False
        return self.engine.cancel_order(order_id)
    
    # ==================== 数据接口 ====================
    
    def get_position(self, symbol: str, direction: Direction = None) -> Optional[PositionData]:
        """
        获取持仓
        """
        if not self.engine:
            return None
        return self.engine.get_position(symbol, direction)
    
    def get_account(self) -> Optional[AccountData]:
        """
        获取账户信息
        """
        if not self.engine:
            return None
        return self.engine.get_account()
    
    def get_bars(self, symbol: str, n: int = 100) -> Optional[DataBuffer]:
        """
        获取最近N根K线
        """
        return self._bar_buffers.get(symbol)
    
    # ==================== 技术指标 ====================
    
    def sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均"""
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        alpha = 2 / (period + 1)
        result = np.full(len(data), np.nan)
        result[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """平均真实波幅"""
        if len(high) < 2:
            return np.array([np.nan] * len(high))
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        
        atr_values = self.sma(tr, period)
        return atr_values
    
    def rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """相对强弱指标"""
        if len(data) < period + 1:
            return np.array([np.nan] * len(data))
        
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = self.sma(gain, period)
        avg_loss = self.sma(loss, period)
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        
        # 对齐长度
        result = np.full(len(data), np.nan)
        result[1:] = rsi_values
        
        return result
    
    def macd(
        self, 
        data: np.ndarray, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        dif = ema_fast - ema_slow
        dea = self.ema(dif[~np.isnan(dif)], signal)
        
        # 对齐长度
        dea_full = np.full(len(data), np.nan)
        dea_full[-len(dea):] = dea
        
        macd_hist = 2 * (dif - dea_full)
        
        return dif, dea_full, macd_hist
    
    def bollinger_bands(
        self, 
        data: np.ndarray, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        middle = self.sma(data, period)
        
        std = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def kdj(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray,
        n: int = 9,
        m1: int = 3,
        m2: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """KDJ指标"""
        length = len(close)
        rsv = np.full(length, np.nan)
        k = np.full(length, 50.0)
        d = np.full(length, 50.0)
        
        for i in range(n - 1, length):
            hh = np.max(high[i - n + 1:i + 1])
            ll = np.min(low[i - n + 1:i + 1])
            
            if hh != ll:
                rsv[i] = (close[i] - ll) / (hh - ll) * 100
            else:
                rsv[i] = 50
            
            k[i] = (m1 - 1) / m1 * k[i - 1] + rsv[i] / m1
            d[i] = (m2 - 1) / m2 * d[i - 1] + k[i] / m2
        
        j = 3 * k - 2 * d
        
        return k, d, j

    # ==================== 信号过滤与趋势确认 ====================
    
    def check_signal_filter(self, symbol: str) -> bool:
        """
        检查是否满足信号间隔要求
        避免过于频繁的交易
        """
        if not self._signal_filter_enabled:
            return True
        
        current_bar = self._bar_counter.get(symbol, 0)
        last_signal = self._last_signal_bar.get(symbol, -999)
        
        return (current_bar - last_signal) >= self._min_signal_interval
    
    def record_signal(self, symbol: str):
        """记录信号发出的K线"""
        self._last_signal_bar[symbol] = self._bar_counter.get(symbol, 0)
    
    def update_bar_counter(self, symbol: str):
        """更新K线计数器"""
        self._bar_counter[symbol] = self._bar_counter.get(symbol, 0) + 1
    
    def check_trend_filter(self, symbol: str, direction: int) -> bool:
        """
        趋势过滤器
        只在趋势方向交易，减少逆势交易
        
        Args:
            symbol: 合约代码
            direction: 1=做多, -1=做空
        
        Returns:
            是否符合趋势方向
        """
        if not self._trend_filter_enabled:
            return True
        
        buffer = self._bar_buffers.get(symbol)
        if not buffer or len(buffer) < self._trend_period:
            return True  # 数据不足时不过滤
        
        closes = buffer.get_array("close")
        
        # 计算长期均线判断趋势
        ma_long = self.sma(closes, self._trend_period)[-1]
        current_price = closes[-1]
        
        if np.isnan(ma_long):
            return True
        
        # 趋势判断：价格在均线上方为上涨趋势
        trend = 1 if current_price > ma_long else -1
        
        # 只允许顺势交易
        return direction == trend
    
    def check_volatility_filter(self, symbol: str) -> bool:
        """
        波动率过滤器
        避免在波动率过低(震荡)或过高(极端)的市场交易
        
        Returns:
            是否在合适的波动率范围内
        """
        if not self._volatility_filter_enabled:
            return True
        
        buffer = self._bar_buffers.get(symbol)
        if not buffer or len(buffer) < 20:
            return True
        
        closes = buffer.get_array("close")
        returns = np.diff(closes) / closes[:-1]
        
        if len(returns) < 10:
            return True
        
        volatility = np.std(returns[-20:])
        
        return self._min_volatility <= volatility <= self._max_volatility
    
    def calculate_dynamic_stop_loss(
        self, 
        symbol: str, 
        entry_price: float, 
        direction: Direction
    ) -> float:
        """
        计算动态止损价格(基于ATR)
        
        Args:
            symbol: 合约代码
            entry_price: 入场价格
            direction: 交易方向
        
        Returns:
            止损价格
        """
        buffer = self._bar_buffers.get(symbol)
        if not buffer or len(buffer) < 20:
            # 默认2%止损
            if direction == Direction.LONG:
                return entry_price * 0.98
            else:
                return entry_price * 1.02
        
        highs = buffer.get_array("high")
        lows = buffer.get_array("low")
        closes = buffer.get_array("close")
        
        atr_values = self.atr(highs, lows, closes, 14)
        current_atr = atr_values[-1]
        
        if np.isnan(current_atr):
            if direction == Direction.LONG:
                return entry_price * 0.98
            else:
                return entry_price * 1.02
        
        stop_distance = current_atr * self._atr_stop_multiple
        
        if direction == Direction.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def update_trailing_stop(
        self, 
        symbol: str, 
        current_price: float, 
        direction: Direction
    ) -> Optional[float]:
        """
        更新追踪止损
        
        Args:
            symbol: 合约代码
            current_price: 当前价格
            direction: 持仓方向
        
        Returns:
            触发止损返回止损价格，否则返回None
        """
        if symbol not in self._trailing_stops:
            return None
        
        stop_info = self._trailing_stops[symbol]
        entry_price = stop_info['entry_price']
        
        buffer = self._bar_buffers.get(symbol)
        if buffer and len(buffer) >= 14:
            highs = buffer.get_array("high")
            lows = buffer.get_array("low")
            closes = buffer.get_array("close")
            atr_values = self.atr(highs, lows, closes, 14)
            current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else entry_price * 0.02
        else:
            current_atr = entry_price * 0.02
        
        trailing_distance = current_atr * self._atr_stop_multiple
        
        if direction == Direction.LONG:
            # 多头：更新最高价，计算追踪止损
            if current_price > stop_info.get('highest', entry_price):
                stop_info['highest'] = current_price
                stop_info['stop_price'] = current_price - trailing_distance
            
            # 检查是否触发止损
            if current_price <= stop_info['stop_price']:
                return stop_info['stop_price']
        else:
            # 空头：更新最低价，计算追踪止损
            if current_price < stop_info.get('lowest', entry_price):
                stop_info['lowest'] = current_price
                stop_info['stop_price'] = current_price + trailing_distance
            
            # 检查是否触发止损
            if current_price >= stop_info['stop_price']:
                return stop_info['stop_price']
        
        return None
    
    def init_trailing_stop(self, symbol: str, entry_price: float, direction: Direction):
        """初始化追踪止损"""
        stop_price = self.calculate_dynamic_stop_loss(symbol, entry_price, direction)
        
        if direction == Direction.LONG:
            self._trailing_stops[symbol] = {
                'entry_price': entry_price,
                'highest': entry_price,
                'lowest': entry_price,
                'stop_price': stop_price,
                'direction': direction
            }
        else:
            self._trailing_stops[symbol] = {
                'entry_price': entry_price,
                'highest': entry_price,
                'lowest': entry_price,
                'stop_price': stop_price,
                'direction': direction
            }
    
    def clear_trailing_stop(self, symbol: str):
        """清除追踪止损"""
        if symbol in self._trailing_stops:
            del self._trailing_stops[symbol]
    
    def record_trade_result(self, pnl: float):
        """记录交易结果用于动态调整"""
        self._recent_trades.append(pnl)
        if len(self._recent_trades) > self._max_recent_trades:
            self._recent_trades.pop(0)
    
    def get_recent_win_rate(self) -> float:
        """获取近期胜率"""
        if not self._recent_trades:
            return 0.5
        wins = sum(1 for pnl in self._recent_trades if pnl > 0)
        return wins / len(self._recent_trades)
    
    def should_reduce_position(self) -> bool:
        """
        判断是否应该减仓
        基于近期交易表现
        """
        if len(self._recent_trades) < 5:
            return False
        
        # 连续亏损检查
        consecutive_losses = 0
        for pnl in reversed(self._recent_trades):
            if pnl < 0:
                consecutive_losses += 1
            else:
                break
        
        # 连续亏损3次以上，建议减仓
        if consecutive_losses >= 3:
            return True
        
        # 近期胜率过低
        if self.get_recent_win_rate() < 0.35:
            return True
        
        return False
    
    def calculate_position_size(
        self, 
        symbol: str, 
        base_size: int = 1
    ) -> int:
        """
        计算调整后的仓位大小
        基于波动率和近期表现动态调整
        """
        adjusted_size = base_size
        
        # 根据近期表现调整
        if self.should_reduce_position():
            adjusted_size = max(1, adjusted_size // 2)
            logger.info(f"{symbol}: Position reduced due to poor recent performance")
        
        # 根据波动率调整
        buffer = self._bar_buffers.get(symbol)
        if buffer and len(buffer) >= 20:
            closes = buffer.get_array("close")
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns[-20:])
            
            # 高波动率时减仓
            if volatility > 0.04:
                adjusted_size = max(1, int(adjusted_size * 0.7))
            elif volatility < 0.01:
                # 低波动率时可以略微加仓
                adjusted_size = int(adjusted_size * 1.2)
        
        return max(1, adjusted_size)


# ==================== 经典策略模板 ====================

class DualMAStrategy(BaseStrategy):
    """
    双均线策略 - 增强版
    金叉做多，死叉做空
    添加信号过滤、趋势确认、追踪止损
    """
    
    strategy_name = "DualMA"
    
    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        symbols: List[str] = None,
        use_trend_filter: bool = True,
        use_volatility_filter: bool = True,
        use_trailing_stop: bool = True
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.symbols = symbols or []
        
        # 增强配置
        self._trend_filter_enabled = use_trend_filter
        self._volatility_filter_enabled = use_volatility_filter
        self._use_trailing_stop = use_trailing_stop
        self._min_signal_interval = 3  # 最小信号间隔
        
        self.parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "use_trend_filter": use_trend_filter,
            "use_volatility_filter": use_volatility_filter
        }
        
        # 存储上一次的均线状态
        self._last_ma_state: Dict[str, str] = {}  # symbol -> "above" / "below"
        
        # 均线斜率确认(用于过滤假信号)
        self._ma_slope_period = 3  # 用于计算斜率的周期
    
    def on_init(self):
        super().on_init()
        # 需要更大的缓冲区以计算趋势
        buffer_size = max(self.slow_period * 3, self._trend_period + 10)
        for symbol in self.symbols:
            self._bar_buffers[symbol] = DataBuffer(max_size=buffer_size)
    
    def _check_ma_slope(self, symbol: str, direction: int) -> bool:
        """
        检查均线斜率确认信号
        金叉时慢线应该走平或上升，死叉时慢线应该走平或下降
        """
        buffer = self._bar_buffers.get(symbol)
        if not buffer or len(buffer) < self.slow_period + self._ma_slope_period:
            return True
        
        closes = buffer.get_array("close")
        
        # 计算最近几根K线的慢均线
        ma_slow_current = np.mean(closes[-self.slow_period:])
        ma_slow_prev = np.mean(closes[-self.slow_period-self._ma_slope_period:-self._ma_slope_period])
        
        slope = (ma_slow_current - ma_slow_prev) / ma_slow_prev if ma_slow_prev != 0 else 0
        
        # 做多时，慢线斜率应>=0；做空时，慢线斜率应<=0
        if direction == 1:
            return slope >= -0.001  # 允许轻微下降
        else:
            return slope <= 0.001   # 允许轻微上升
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                buffer_size = max(self.slow_period * 3, self._trend_period + 10)
                self._bar_buffers[symbol] = DataBuffer(max_size=buffer_size)
            self._bar_buffers[symbol].append(bar)
            self.update_bar_counter(symbol)
            
            # 数据不足
            if len(self._bar_buffers[symbol]) < self.slow_period:
                continue
            
            # 检查追踪止损
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                if position.direction == Direction.LONG:
                    current_pos = position.volume
                else:
                    current_pos = -position.volume
            
            # 如果有持仓，检查追踪止损
            if current_pos != 0 and self._use_trailing_stop:
                direction = Direction.LONG if current_pos > 0 else Direction.SHORT
                stop_price = self.update_trailing_stop(symbol, bar.close, direction)
                if stop_price is not None:
                    # 触发追踪止损
                    if current_pos > 0:
                        self.sell(symbol, bar.close, current_pos)
                        logger.info(f"{symbol} Trailing Stop: Sell @ {bar.close}")
                    else:
                        self.cover(symbol, bar.close, abs(current_pos))
                        logger.info(f"{symbol} Trailing Stop: Cover @ {bar.close}")
                    self.clear_trailing_stop(symbol)
                    self._last_ma_state[symbol] = ""  # 重置状态
                    continue
            
            # 计算均线
            closes = self._bar_buffers[symbol].get_array("close")
            ma_fast = self.sma(closes, self.fast_period)[-1]
            ma_slow = self.sma(closes, self.slow_period)[-1]
            
            if np.isnan(ma_fast) or np.isnan(ma_slow):
                continue
            
            # 判断当前状态
            current_state = "above" if ma_fast > ma_slow else "below"
            last_state = self._last_ma_state.get(symbol)
            
            # 信号判断
            if last_state == "below" and current_state == "above":
                # 金叉信号
                # 应用过滤器
                if not self.check_signal_filter(symbol):
                    logger.debug(f"{symbol} Golden Cross filtered: signal interval")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self.check_trend_filter(symbol, 1):
                    logger.debug(f"{symbol} Golden Cross filtered: against trend")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self.check_volatility_filter(symbol):
                    logger.debug(f"{symbol} Golden Cross filtered: volatility")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self._check_ma_slope(symbol, 1):
                    logger.debug(f"{symbol} Golden Cross filtered: MA slope")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                # 执行交易
                if current_pos < 0:
                    self.cover(symbol, bar.close, abs(current_pos))
                    self.clear_trailing_stop(symbol)
                if current_pos <= 0:
                    # 计算动态仓位
                    size = self.calculate_position_size(symbol, 1)
                    self.buy(symbol, bar.close, size)
                    self.record_signal(symbol)
                    # 初始化追踪止损
                    if self._use_trailing_stop:
                        self.init_trailing_stop(symbol, bar.close, Direction.LONG)
                    logger.info(f"{symbol} Golden Cross: Buy @ {bar.close}, size={size}")
            
            elif last_state == "above" and current_state == "below":
                # 死叉信号
                # 应用过滤器
                if not self.check_signal_filter(symbol):
                    logger.debug(f"{symbol} Death Cross filtered: signal interval")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self.check_trend_filter(symbol, -1):
                    logger.debug(f"{symbol} Death Cross filtered: against trend")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self.check_volatility_filter(symbol):
                    logger.debug(f"{symbol} Death Cross filtered: volatility")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                if not self._check_ma_slope(symbol, -1):
                    logger.debug(f"{symbol} Death Cross filtered: MA slope")
                    self._last_ma_state[symbol] = current_state
                    continue
                
                # 执行交易
                if current_pos > 0:
                    self.sell(symbol, bar.close, current_pos)
                    self.clear_trailing_stop(symbol)
                if current_pos >= 0:
                    # 计算动态仓位
                    size = self.calculate_position_size(symbol, 1)
                    self.short(symbol, bar.close, size)
                    self.record_signal(symbol)
                    # 初始化追踪止损
                    if self._use_trailing_stop:
                        self.init_trailing_stop(symbol, bar.close, Direction.SHORT)
                    logger.info(f"{symbol} Death Cross: Short @ {bar.close}, size={size}")
            
            self._last_ma_state[symbol] = current_state


class TurtleStrategy(BaseStrategy):
    """
    海龟交易法则
    突破N日高点做多，突破N日低点做空
    使用ATR进行动态止损和仓位管理
    """
    
    strategy_name = "Turtle"
    
    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        risk_percent: float = 0.01,
        symbols: List[str] = None
    ):
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_percent = risk_percent
        self.symbols = symbols or []
        
        self.parameters = {
            "entry_period": entry_period,
            "exit_period": exit_period,
            "atr_period": atr_period,
            "risk_percent": risk_percent
        }
        
        # 入场价格和ATR记录（用于止损）
        self._entry_prices: Dict[str, float] = {}
        self._entry_atr: Dict[str, float] = {}
    
    def on_init(self):
        super().on_init()
        buffer_size = max(self.entry_period, self.exit_period, self.atr_period) + 10
        for symbol in self.symbols:
            self._bar_buffers[symbol] = DataBuffer(max_size=buffer_size)
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                buffer_size = max(self.entry_period, self.exit_period, self.atr_period) + 10
                self._bar_buffers[symbol] = DataBuffer(max_size=buffer_size)
            self._bar_buffers[symbol].append(bar)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < self.entry_period:
                continue
            
            # 计算通道和ATR
            highs = buffer.get_array("high")
            lows = buffer.get_array("low")
            closes = buffer.get_array("close")
            
            entry_high = np.max(highs[-self.entry_period:-1])  # 不包含当前K线
            entry_low = np.min(lows[-self.entry_period:-1])
            exit_high = np.max(highs[-self.exit_period:-1])
            exit_low = np.min(lows[-self.exit_period:-1])
            
            atr_values = self.atr(highs, lows, closes, self.atr_period)
            current_atr = atr_values[-1]
            
            if np.isnan(current_atr):
                continue
            
            # 获取持仓
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                if position.direction == Direction.LONG:
                    current_pos = position.volume
                else:
                    current_pos = -position.volume
            
            # 计算仓位大小
            account = self.get_account()
            if account:
                unit_size = int(
                    account.balance * self.risk_percent / 
                    (current_atr * 10)  # 假设合约乘数10
                )
                unit_size = max(1, unit_size)
            else:
                unit_size = 1
            
            # 入场信号
            if current_pos == 0:
                if bar.close > entry_high:
                    # 突破高点做多
                    self.buy(symbol, bar.close, unit_size)
                    self._entry_prices[symbol] = bar.close
                    self._entry_atr[symbol] = current_atr
                    logger.info(f"{symbol} Turtle: Long breakout @ {bar.close}")
                
                elif bar.close < entry_low:
                    # 突破低点做空
                    self.short(symbol, bar.close, unit_size)
                    self._entry_prices[symbol] = bar.close
                    self._entry_atr[symbol] = current_atr
                    logger.info(f"{symbol} Turtle: Short breakout @ {bar.close}")
            
            # 出场信号
            elif current_pos > 0:
                entry_price = self._entry_prices.get(symbol, bar.close)
                entry_atr = self._entry_atr.get(symbol, current_atr)
                
                # ATR止损或突破低点出场
                stop_loss = entry_price - 2 * entry_atr
                if bar.close < exit_low or bar.close < stop_loss:
                    self.sell(symbol, bar.close, current_pos)
                    logger.info(f"{symbol} Turtle: Exit long @ {bar.close}")
            
            elif current_pos < 0:
                entry_price = self._entry_prices.get(symbol, bar.close)
                entry_atr = self._entry_atr.get(symbol, current_atr)
                
                # ATR止损或突破高点出场
                stop_loss = entry_price + 2 * entry_atr
                if bar.close > exit_high or bar.close > stop_loss:
                    self.cover(symbol, bar.close, abs(current_pos))
                    logger.info(f"{symbol} Turtle: Exit short @ {bar.close}")


class GridStrategy(BaseStrategy):
    """
    网格交易策略
    在价格区间内设置网格，低买高卖
    """
    
    strategy_name = "Grid"
    
    def __init__(
        self,
        grid_count: int = 10,
        grid_range_percent: float = 0.10,
        position_per_grid: int = 1,
        symbols: List[str] = None
    ):
        super().__init__()
        self.grid_count = grid_count
        self.grid_range_percent = grid_range_percent
        self.position_per_grid = position_per_grid
        self.symbols = symbols or []
        
        self.parameters = {
            "grid_count": grid_count,
            "grid_range_percent": grid_range_percent,
            "position_per_grid": position_per_grid
        }
        
        # 网格设置
        self._grid_prices: Dict[str, List[float]] = {}
        self._grid_positions: Dict[str, List[int]] = {}  # 每个网格的持仓
        self._base_price: Dict[str, float] = {}
        self._initialized: Dict[str, bool] = {}
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 初始化网格
            if not self._initialized.get(symbol):
                self._init_grid(symbol, bar.close)
            
            # 检查网格触发
            self._check_grid(symbol, bar)
    
    def _init_grid(self, symbol: str, base_price: float):
        """初始化网格"""
        self._base_price[symbol] = base_price
        
        # 计算网格价格
        upper = base_price * (1 + self.grid_range_percent / 2)
        lower = base_price * (1 - self.grid_range_percent / 2)
        grid_step = (upper - lower) / self.grid_count
        
        prices = []
        for i in range(self.grid_count + 1):
            prices.append(lower + i * grid_step)
        
        self._grid_prices[symbol] = prices
        self._grid_positions[symbol] = [0] * (self.grid_count + 1)
        self._initialized[symbol] = True
        
        logger.info(f"{symbol} Grid initialized: {lower:.2f} - {upper:.2f}, {self.grid_count} grids")
    
    def _check_grid(self, symbol: str, bar: BarData):
        """检查网格触发"""
        prices = self._grid_prices[symbol]
        positions = self._grid_positions[symbol]
        current_price = bar.close
        
        for i in range(len(prices) - 1):
            grid_low = prices[i]
            grid_high = prices[i + 1]
            
            # 价格进入网格区间
            if grid_low <= current_price < grid_high:
                # 如果该网格没有持仓，买入
                if positions[i] == 0:
                    self.buy(symbol, current_price, self.position_per_grid)
                    positions[i] = self.position_per_grid
                    logger.info(f"{symbol} Grid buy @ {current_price:.2f}, grid {i}")
            
            # 价格上穿网格
            elif current_price >= grid_high and positions[i] > 0:
                self.sell(symbol, current_price, positions[i])
                positions[i] = 0
                logger.info(f"{symbol} Grid sell @ {current_price:.2f}, grid {i}")


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    价格偏离均值过大时反向交易
    """
    
    strategy_name = "MeanReversion"
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        symbols: List[str] = None
    ):
        super().__init__()
        self.lookback_period = lookback_period
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.symbols = symbols or []
        
        self.parameters = {
            "lookback_period": lookback_period,
            "entry_std": entry_std,
            "exit_std": exit_std
        }
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                self._bar_buffers[symbol] = DataBuffer(max_size=self.lookback_period * 2)
            self._bar_buffers[symbol].append(bar)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < self.lookback_period:
                continue
            
            # 计算z-score
            closes = buffer.get_array("close")
            mean = np.mean(closes[-self.lookback_period:])
            std = np.std(closes[-self.lookback_period:])
            
            if std == 0:
                continue
            
            z_score = (bar.close - mean) / std
            
            # 获取持仓
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                if position.direction == Direction.LONG:
                    current_pos = position.volume
                else:
                    current_pos = -position.volume
            
            # 交易逻辑
            if current_pos == 0:
                if z_score > self.entry_std:
                    # 价格过高，做空
                    self.short(symbol, bar.close, 1)
                    logger.info(f"{symbol} Mean Reversion: Short @ {bar.close}, z={z_score:.2f}")
                
                elif z_score < -self.entry_std:
                    # 价格过低，做多
                    self.buy(symbol, bar.close, 1)
                    logger.info(f"{symbol} Mean Reversion: Long @ {bar.close}, z={z_score:.2f}")
            
            elif current_pos > 0:
                # 持多仓，回归均值时平仓
                if z_score > -self.exit_std:
                    self.sell(symbol, bar.close, current_pos)
                    logger.info(f"{symbol} Mean Reversion: Close long @ {bar.close}, z={z_score:.2f}")
            
            elif current_pos < 0:
                # 持空仓，回归均值时平仓
                if z_score < self.exit_std:
                    self.cover(symbol, bar.close, abs(current_pos))
                    logger.info(f"{symbol} Mean Reversion: Close short @ {bar.close}, z={z_score:.2f}")


class MomentumStrategy(BaseStrategy):
    """
    动量策略
    追涨杀跌，跟随趋势方向
    """
    
    strategy_name = "Momentum"
    
    def __init__(
        self,
        momentum_period: int = 20,
        holding_period: int = 5,
        top_n: int = 3,
        symbols: List[str] = None
    ):
        super().__init__()
        self.momentum_period = momentum_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.symbols = symbols or []
        
        self.parameters = {
            "momentum_period": momentum_period,
            "holding_period": holding_period,
            "top_n": top_n
        }
        
        self._holding_bars: Dict[str, int] = {}  # 持仓K线数
    
    def on_bar(self, bars: Dict[str, BarData]):
        # 计算所有品种的动量
        momentums = {}
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                self._bar_buffers[symbol] = DataBuffer(max_size=self.momentum_period + 10)
            self._bar_buffers[symbol].append(bar)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < self.momentum_period:
                continue
            
            # 计算动量（收益率）
            closes = buffer.get_array("close")
            momentum = (closes[-1] - closes[-self.momentum_period]) / closes[-self.momentum_period]
            momentums[symbol] = momentum
            
            # 更新持仓计数
            position = self.get_position(symbol)
            if position and position.volume > 0:
                self._holding_bars[symbol] = self._holding_bars.get(symbol, 0) + 1
        
        if not momentums:
            return
        
        # 排序选出动量最强/最弱的品种
        sorted_symbols = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        
        top_symbols = [s[0] for s in sorted_symbols[:self.top_n] if s[1] > 0]
        bottom_symbols = [s[0] for s in sorted_symbols[-self.top_n:] if s[1] < 0]
        
        # 处理持仓
        for symbol in self.symbols:
            if symbol not in bars:
                continue
            
            bar = bars[symbol]
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                if position.direction == Direction.LONG:
                    current_pos = position.volume
                else:
                    current_pos = -position.volume
            
            # 检查是否需要平仓（持仓到期或不在动量排名中）
            holding_bars = self._holding_bars.get(symbol, 0)
            
            if current_pos > 0:
                if holding_bars >= self.holding_period or symbol not in top_symbols:
                    self.sell(symbol, bar.close, current_pos)
                    self._holding_bars[symbol] = 0
                    logger.info(f"{symbol} Momentum: Close long @ {bar.close}")
            
            elif current_pos < 0:
                if holding_bars >= self.holding_period or symbol not in bottom_symbols:
                    self.cover(symbol, bar.close, abs(current_pos))
                    self._holding_bars[symbol] = 0
                    logger.info(f"{symbol} Momentum: Close short @ {bar.close}")
            
            # 开新仓
            if current_pos == 0:
                if symbol in top_symbols:
                    self.buy(symbol, bar.close, 1)
                    self._holding_bars[symbol] = 0
                    logger.info(f"{symbol} Momentum: Long @ {bar.close}, mom={momentums.get(symbol, 0):.4f}")
                
                elif symbol in bottom_symbols:
                    self.short(symbol, bar.close, 1)
                    self._holding_bars[symbol] = 0
                    logger.info(f"{symbol} Momentum: Short @ {bar.close}, mom={momentums.get(symbol, 0):.4f}")
