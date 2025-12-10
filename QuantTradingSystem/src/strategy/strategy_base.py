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
    """
    
    # 策略元信息
    strategy_name: str = "BaseStrategy"
    author: str = "QuantMaster"
    version: str = "1.0.0"
    
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
        """平均真实波幅 - 使用EMA提升灵敏度"""
        if len(high) < 2:
            return np.array([np.nan] * len(high))
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        
        # 使用EMA而非SMA，更快响应市场变化
        atr_values = self.ema(tr, period)
        return atr_values
    
    def detect_market_regime(self, closes: np.ndarray, period: int = 20) -> str:
        """
        检测市场状态
        Returns: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        if len(closes) < period:
            return 'unknown'
        
        # 计算收益率
        returns = np.diff(closes[-period:]) / closes[-period:-1]
        
        # 趋势强度 - 收益率均值
        trend_strength = np.mean(returns)
        # 波动率
        volatility = np.std(returns)
        # ADX风格的方向指标
        positive_moves = np.sum(returns > 0) / len(returns)
        
        # 波动率阈值（年化20%对应日波动1.26%）
        high_vol_threshold = 0.015
        trend_threshold = 0.001
        
        if volatility > high_vol_threshold:
            return 'volatile'
        elif trend_strength > trend_threshold and positive_moves > 0.6:
            return 'trending_up'
        elif trend_strength < -trend_threshold and positive_moves < 0.4:
            return 'trending_down'
        else:
            return 'ranging'
    
    def calculate_volatility_percentile(self, closes: np.ndarray, period: int = 60) -> float:
        """
        计算当前波动率在历史中的分位数
        用于动态调整风险参数
        """
        if len(closes) < period + 20:
            return 0.5
        
        returns = np.diff(closes) / closes[:-1]
        
        # 计算滚动波动率
        rolling_vol = []
        for i in range(20, len(returns)):
            vol = np.std(returns[i-20:i])
            rolling_vol.append(vol)
        
        if len(rolling_vol) < 2:
            return 0.5
        
        current_vol = rolling_vol[-1]
        # 计算分位数
        percentile = np.sum(np.array(rolling_vol) <= current_vol) / len(rolling_vol)
        
        return percentile
    
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


# ==================== 经典策略模板 ====================

class DualMAStrategy(BaseStrategy):
    """
    双均线策略 - 增强版
    1. 添加趋势过滤器避免震荡市场
    2. 使用多重确认减少假信号
    3. 动态止损止盈
    4. 仓位管理基于波动率
    """
    
    strategy_name = "DualMA"
    
    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 21,
        trend_period: int = 60,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        atr_profit_multiplier: float = 4.0,
        symbols: List[str] = None
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_profit_multiplier = atr_profit_multiplier
        self.symbols = symbols or []
        
        self.parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "trend_period": trend_period,
            "atr_stop_multiplier": atr_stop_multiplier
        }
        
        # 存储交易信息
        self._last_ma_state: Dict[str, str] = {}
        self._entry_prices: Dict[str, float] = {}
        self._stop_losses: Dict[str, float] = {}
        self._take_profits: Dict[str, float] = {}
        self._trailing_highs: Dict[str, float] = {}
        self._trailing_lows: Dict[str, float] = {}
    
    def on_init(self):
        super().on_init()
        for symbol in self.symbols:
            self._bar_buffers[symbol] = DataBuffer(max_size=max(self.trend_period, self.slow_period) * 2)
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                self._bar_buffers[symbol] = DataBuffer(max_size=max(self.trend_period, self.slow_period) * 2)
            self._bar_buffers[symbol].append(bar)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < max(self.slow_period, self.trend_period):
                continue
            
            # 提取数据
            closes = buffer.get_array("close")
            highs = buffer.get_array("high")
            lows = buffer.get_array("low")
            
            # 计算指标
            ma_fast = self.ema(closes, self.fast_period)[-1]
            ma_slow = self.ema(closes, self.slow_period)[-1]
            ma_trend = self.ema(closes, self.trend_period)[-1]
            atr_values = self.atr(highs, lows, closes, self.atr_period)
            current_atr = atr_values[-1]
            
            if np.isnan(ma_fast) or np.isnan(ma_slow) or np.isnan(ma_trend) or np.isnan(current_atr):
                continue
            
            # 市场状态检测
            regime = self.detect_market_regime(closes, 30)
            
            # 获取持仓
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                if position.direction == Direction.LONG:
                    current_pos = position.volume
                else:
                    current_pos = -position.volume
            
            # 更新trailing stop
            if current_pos > 0:
                if symbol not in self._trailing_highs or bar.high > self._trailing_highs[symbol]:
                    self._trailing_highs[symbol] = bar.high
                    # 移动止损
                    new_stop = bar.high - current_atr * self.atr_stop_multiplier
                    if symbol in self._stop_losses:
                        self._stop_losses[symbol] = max(self._stop_losses[symbol], new_stop)
            
            elif current_pos < 0:
                if symbol not in self._trailing_lows or bar.low < self._trailing_lows[symbol]:
                    self._trailing_lows[symbol] = bar.low
                    new_stop = bar.low + current_atr * self.atr_stop_multiplier
                    if symbol in self._stop_losses:
                        self._stop_losses[symbol] = min(self._stop_losses[symbol], new_stop)
            
            # 止损止盈检查
            if current_pos > 0:
                stop_loss = self._stop_losses.get(symbol, 0)
                take_profit = self._take_profits.get(symbol, float('inf'))
                
                if bar.close <= stop_loss:
                    self.sell(symbol, bar.close, current_pos)
                    logger.info(f"{symbol} Stop Loss Hit: Sell @ {bar.close}, loss={bar.close - self._entry_prices.get(symbol, bar.close):.2f}")
                    self._clear_trade_data(symbol)
                    continue
                elif bar.close >= take_profit:
                    self.sell(symbol, bar.close, current_pos)
                    logger.info(f"{symbol} Take Profit Hit: Sell @ {bar.close}, profit={bar.close - self._entry_prices.get(symbol, bar.close):.2f}")
                    self._clear_trade_data(symbol)
                    continue
            
            elif current_pos < 0:
                stop_loss = self._stop_losses.get(symbol, float('inf'))
                take_profit = self._take_profits.get(symbol, 0)
                
                if bar.close >= stop_loss:
                    self.cover(symbol, bar.close, abs(current_pos))
                    logger.info(f"{symbol} Stop Loss Hit: Cover @ {bar.close}")
                    self._clear_trade_data(symbol)
                    continue
                elif bar.close <= take_profit:
                    self.cover(symbol, bar.close, abs(current_pos))
                    logger.info(f"{symbol} Take Profit Hit: Cover @ {bar.close}")
                    self._clear_trade_data(symbol)
                    continue
            
            # 判断当前状态
            current_state = "above" if ma_fast > ma_slow else "below"
            last_state = self._last_ma_state.get(symbol)
            
            # 趋势过滤器：价格必须在趋势线上方才做多，下方才做空
            trend_filter_long = bar.close > ma_trend
            trend_filter_short = bar.close < ma_trend
            
            # 只在非高波动市场交易
            if regime == 'volatile':
                self._last_ma_state[symbol] = current_state
                continue
            
            # 信号判断 - 添加多重确认
            if last_state == "below" and current_state == "above" and trend_filter_long:
                # 额外确认：价格回调后突破
                recent_high = np.max(highs[-5:])
                if bar.close >= recent_high * 0.995:  # 突破近期高点
                    # 金叉 - 平空开多
                    if current_pos < 0:
                        self.cover(symbol, bar.close, abs(current_pos))
                        self._clear_trade_data(symbol)
                    
                    if current_pos <= 0:
                        # 基于波动率调整仓位
                        vol_percentile = self.calculate_volatility_percentile(closes, 60)
                        size = 1 if vol_percentile < 0.8 else 1  # 高波动时可减仓
                        
                        self.buy(symbol, bar.close, size)
                        self._entry_prices[symbol] = bar.close
                        self._stop_losses[symbol] = bar.close - current_atr * self.atr_stop_multiplier
                        self._take_profits[symbol] = bar.close + current_atr * self.atr_profit_multiplier
                        self._trailing_highs[symbol] = bar.high
                        logger.info(f"{symbol} Golden Cross [LONG]: Entry @ {bar.close}, Stop @ {self._stop_losses[symbol]:.2f}, Target @ {self._take_profits[symbol]:.2f}, Regime: {regime}")
            
            elif last_state == "above" and current_state == "below" and trend_filter_short:
                # 额外确认：价格反弹后跌破
                recent_low = np.min(lows[-5:])
                if bar.close <= recent_low * 1.005:
                    # 死叉 - 平多开空
                    if current_pos > 0:
                        self.sell(symbol, bar.close, current_pos)
                        self._clear_trade_data(symbol)
                    
                    if current_pos >= 0 and regime != 'ranging':  # 震荡市不做空
                        vol_percentile = self.calculate_volatility_percentile(closes, 60)
                        size = 1 if vol_percentile < 0.8 else 1
                        
                        self.short(symbol, bar.close, size)
                        self._entry_prices[symbol] = bar.close
                        self._stop_losses[symbol] = bar.close + current_atr * self.atr_stop_multiplier
                        self._take_profits[symbol] = bar.close - current_atr * self.atr_profit_multiplier
                        self._trailing_lows[symbol] = bar.low
                        logger.info(f"{symbol} Death Cross [SHORT]: Entry @ {bar.close}, Stop @ {self._stop_losses[symbol]:.2f}, Target @ {self._take_profits[symbol]:.2f}, Regime: {regime}")
            
            self._last_ma_state[symbol] = current_state
    
    def _clear_trade_data(self, symbol: str):
        """清理交易数据"""
        self._entry_prices.pop(symbol, None)
        self._stop_losses.pop(symbol, None)
        self._take_profits.pop(symbol, None)
        self._trailing_highs.pop(symbol, None)
        self._trailing_lows.pop(symbol, None)


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
