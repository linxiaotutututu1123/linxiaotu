"""
高级策略 - 套利策略和组合策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .strategy_base import BaseStrategy
from ..data.data_structures import (
    BarData, Direction, Offset, OrderType, DataBuffer
)

logger = logging.getLogger(__name__)


class SpreadArbitrageStrategy(BaseStrategy):
    """
    跨期套利策略
    利用不同合约之间的价差进行套利
    """
    
    strategy_name = "SpreadArbitrage"
    
    def __init__(
        self,
        near_symbol: str,      # 近月合约
        far_symbol: str,       # 远月合约
        spread_mean: float = 0,     # 价差均值
        spread_std: float = 50,     # 价差标准差
        entry_threshold: float = 2.0,  # 入场阈值（标准差倍数）
        exit_threshold: float = 0.5,   # 出场阈值
        position_size: int = 1
    ):
        super().__init__()
        self.near_symbol = near_symbol
        self.far_symbol = far_symbol
        self.spread_mean = spread_mean
        self.spread_std = spread_std
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        
        self.parameters = {
            "near_symbol": near_symbol,
            "far_symbol": far_symbol,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold
        }
        
        # 价差缓冲区
        self._spread_buffer = DataBuffer(max_size=100)
        self._position_state = 0  # 0=无仓位, 1=做多价差, -1=做空价差
    
    def on_bar(self, bars: Dict[str, BarData]):
        # 确保两个合约都有数据
        if self.near_symbol not in bars or self.far_symbol not in bars:
            return
        
        near_bar = bars[self.near_symbol]
        far_bar = bars[self.far_symbol]
        
        # 计算价差（远月 - 近月）
        spread = far_bar.close - near_bar.close
        self._spread_buffer.append(spread)
        
        # 需要足够数据来计算统计量
        if len(self._spread_buffer) < 20:
            return
        
        # 动态更新价差统计
        spreads = np.array(list(self._spread_buffer._data))
        current_mean = np.mean(spreads)
        current_std = np.std(spreads)
        
        if current_std == 0:
            return
        
        # 计算z-score
        z_score = (spread - current_mean) / current_std
        
        # 交易逻辑
        if self._position_state == 0:
            if z_score > self.entry_threshold:
                # 价差过大，做空价差（卖远买近）
                self.short(self.far_symbol, far_bar.close, self.position_size)
                self.buy(self.near_symbol, near_bar.close, self.position_size)
                self._position_state = -1
                logger.info(f"Spread Arb: Short spread @ {spread:.2f}, z={z_score:.2f}")
            
            elif z_score < -self.entry_threshold:
                # 价差过小，做多价差（买远卖近）
                self.buy(self.far_symbol, far_bar.close, self.position_size)
                self.short(self.near_symbol, near_bar.close, self.position_size)
                self._position_state = 1
                logger.info(f"Spread Arb: Long spread @ {spread:.2f}, z={z_score:.2f}")
        
        elif self._position_state == 1:
            # 持有多价差仓位
            if z_score > -self.exit_threshold:
                # 价差回归，平仓
                self.sell(self.far_symbol, far_bar.close, self.position_size)
                self.cover(self.near_symbol, near_bar.close, self.position_size)
                self._position_state = 0
                logger.info(f"Spread Arb: Close long spread @ {spread:.2f}")
        
        elif self._position_state == -1:
            # 持有空价差仓位
            if z_score < self.exit_threshold:
                # 价差回归，平仓
                self.cover(self.far_symbol, far_bar.close, self.position_size)
                self.sell(self.near_symbol, near_bar.close, self.position_size)
                self._position_state = 0
                logger.info(f"Spread Arb: Close short spread @ {spread:.2f}")


class CrossSymbolArbitrageStrategy(BaseStrategy):
    """
    跨品种套利策略
    利用相关品种之间的价格关系进行套利
    """
    
    strategy_name = "CrossSymbolArbitrage"
    
    def __init__(
        self,
        symbol_a: str,           # 品种A
        symbol_b: str,           # 品种B
        ratio: float = 1.0,      # 价格比例系数
        lookback: int = 60,      # 回溯期
        entry_std: float = 2.0,  # 入场标准差
        exit_std: float = 0.5,   # 出场标准差
        position_size: int = 1
    ):
        super().__init__()
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.ratio = ratio
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.position_size = position_size
        
        self.parameters = {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "ratio": ratio,
            "entry_std": entry_std
        }
        
        self._ratio_buffer = DataBuffer(max_size=lookback * 2)
        self._position_state = 0
    
    def on_bar(self, bars: Dict[str, BarData]):
        if self.symbol_a not in bars or self.symbol_b not in bars:
            return
        
        bar_a = bars[self.symbol_a]
        bar_b = bars[self.symbol_b]
        
        # 计算价格比率
        if bar_b.close == 0:
            return
        price_ratio = bar_a.close / bar_b.close
        self._ratio_buffer.append(price_ratio)
        
        if len(self._ratio_buffer) < self.lookback:
            return
        
        # 计算统计量
        ratios = np.array(list(self._ratio_buffer._data[-self.lookback:]))
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        if std_ratio == 0:
            return
        
        z_score = (price_ratio - mean_ratio) / std_ratio
        
        # 交易逻辑
        if self._position_state == 0:
            if z_score > self.entry_std:
                # A相对B过贵，卖A买B
                self.short(self.symbol_a, bar_a.close, self.position_size)
                self.buy(self.symbol_b, bar_b.close, self.position_size)
                self._position_state = -1
                logger.info(f"Cross Arb: Short {self.symbol_a}, Long {self.symbol_b}, z={z_score:.2f}")
            
            elif z_score < -self.entry_std:
                # A相对B过便宜，买A卖B
                self.buy(self.symbol_a, bar_a.close, self.position_size)
                self.short(self.symbol_b, bar_b.close, self.position_size)
                self._position_state = 1
                logger.info(f"Cross Arb: Long {self.symbol_a}, Short {self.symbol_b}, z={z_score:.2f}")
        
        elif self._position_state == 1:
            if z_score > -self.exit_std:
                self.sell(self.symbol_a, bar_a.close, self.position_size)
                self.cover(self.symbol_b, bar_b.close, self.position_size)
                self._position_state = 0
                logger.info(f"Cross Arb: Close position, z={z_score:.2f}")
        
        elif self._position_state == -1:
            if z_score < self.exit_std:
                self.cover(self.symbol_a, bar_a.close, self.position_size)
                self.sell(self.symbol_b, bar_b.close, self.position_size)
                self._position_state = 0
                logger.info(f"Cross Arb: Close position, z={z_score:.2f}")


class MultiStrategyPortfolio(BaseStrategy):
    """
    多策略组合
    同时运行多个策略，并进行资金分配
    """
    
    strategy_name = "MultiStrategyPortfolio"
    
    def __init__(
        self,
        strategies: List[Tuple[BaseStrategy, float]],  # (策略, 资金比例)
        rebalance_period: int = 20  # 再平衡周期（K线数）
    ):
        super().__init__()
        self.strategies = strategies
        self.rebalance_period = rebalance_period
        
        # 验证资金比例
        total_weight = sum(weight for _, weight in strategies)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Strategy weights sum to {total_weight}, not 1.0")
        
        self._bar_count = 0
        
        # 策略绩效跟踪
        self._strategy_returns: Dict[str, List[float]] = {}
        self._strategy_allocations: Dict[str, float] = {}
        
        for strategy, weight in strategies:
            self._strategy_allocations[strategy.strategy_name] = weight
            self._strategy_returns[strategy.strategy_name] = []
    
    def on_init(self):
        super().on_init()
        for strategy, _ in self.strategies:
            strategy.engine = self.engine
            strategy.on_init()
    
    def on_bar(self, bars: Dict[str, BarData]):
        self._bar_count += 1
        
        # 调用各子策略
        for strategy, weight in self.strategies:
            try:
                strategy.on_bar(bars)
            except Exception as e:
                logger.error(f"Strategy {strategy.strategy_name} error: {e}")
        
        # 定期再平衡
        if self._bar_count % self.rebalance_period == 0:
            self._rebalance()
    
    def _rebalance(self):
        """
        根据各策略表现动态调整资金分配
        表现好的策略分配更多资金
        """
        logger.info("Portfolio rebalancing...")
        
        # 计算各策略的夏普比率
        sharpe_ratios = {}
        for name, returns in self._strategy_returns.items():
            if len(returns) >= 20:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                sharpe_ratios[name] = max(sharpe, 0.1)  # 最小0.1
        
        if not sharpe_ratios:
            return
        
        # 根据夏普比率重新分配
        total_sharpe = sum(sharpe_ratios.values())
        if total_sharpe > 0:
            for name in sharpe_ratios:
                new_weight = sharpe_ratios[name] / total_sharpe
                old_weight = self._strategy_allocations.get(name, 0)
                # 平滑调整
                self._strategy_allocations[name] = 0.8 * old_weight + 0.2 * new_weight
        
        logger.info(f"New allocations: {self._strategy_allocations}")
    
    def on_trade(self, trade):
        """记录交易收益用于再平衡"""
        super().on_trade(trade)
        
        if trade.strategy_name in self._strategy_returns:
            # 简化：将成交记录为收益
            self._strategy_returns[trade.strategy_name].append(0)  # 实际应计算真实收益


class AdaptiveStrategy(BaseStrategy):
    """
    自适应策略
    根据市场状态自动切换策略模式
    """
    
    strategy_name = "Adaptive"
    
    def __init__(
        self,
        symbols: List[str],
        trend_period: int = 20,
        volatility_period: int = 20,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.03
    ):
        super().__init__()
        self.symbols = symbols
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
        self.parameters = {
            "trend_period": trend_period,
            "volatility_period": volatility_period
        }
        
        # 市场状态
        self._market_state: Dict[str, str] = {}  # trending / ranging / volatile
        
        # 各状态对应的策略逻辑
        self._trend_ma_fast = 5
        self._trend_ma_slow = 20
        self._range_upper_std = 2.0
        self._range_lower_std = -2.0
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                self._bar_buffers[symbol] = DataBuffer(max_size=100)
            self._bar_buffers[symbol].append(bar)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < max(self.trend_period, self.volatility_period):
                continue
            
            # 识别市场状态
            market_state = self._identify_market_state(symbol)
            self._market_state[symbol] = market_state
            
            # 根据市场状态执行相应策略
            if market_state == "trending":
                self._execute_trend_strategy(symbol, bar)
            elif market_state == "ranging":
                self._execute_range_strategy(symbol, bar)
            elif market_state == "volatile":
                self._execute_volatile_strategy(symbol, bar)
    
    def _identify_market_state(self, symbol: str) -> str:
        """识别市场状态"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        
        # 计算趋势强度（使用收益率的均值）
        returns = np.diff(closes[-self.trend_period:]) / closes[-self.trend_period:-1]
        trend_strength = abs(np.mean(returns))
        
        # 计算波动率
        volatility = np.std(returns)
        
        # 判断市场状态
        if volatility > self.volatility_threshold:
            return "volatile"
        elif trend_strength > self.trend_threshold:
            return "trending"
        else:
            return "ranging"
    
    def _execute_trend_strategy(self, symbol: str, bar: BarData):
        """趋势市场策略 - 使用均线跟踪"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        
        ma_fast = self.sma(closes, self._trend_ma_fast)[-1]
        ma_slow = self.sma(closes, self._trend_ma_slow)[-1]
        
        if np.isnan(ma_fast) or np.isnan(ma_slow):
            return
        
        position = self.get_position(symbol)
        current_pos = position.volume if position else 0
        
        if ma_fast > ma_slow and current_pos <= 0:
            if current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
            self.buy(symbol, bar.close, 1)
            logger.info(f"{symbol} Adaptive[Trend]: Long @ {bar.close}")
        
        elif ma_fast < ma_slow and current_pos >= 0:
            if current_pos > 0:
                self.sell(symbol, bar.close, current_pos)
            self.short(symbol, bar.close, 1)
            logger.info(f"{symbol} Adaptive[Trend]: Short @ {bar.close}")
    
    def _execute_range_strategy(self, symbol: str, bar: BarData):
        """震荡市场策略 - 使用布林带均值回归"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        
        upper, middle, lower = self.bollinger_bands(closes, 20, 2.0)
        
        if np.isnan(upper[-1]) or np.isnan(lower[-1]):
            return
        
        position = self.get_position(symbol)
        current_pos = position.volume if position else 0
        
        if bar.close <= lower[-1] and current_pos <= 0:
            if current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
            self.buy(symbol, bar.close, 1)
            logger.info(f"{symbol} Adaptive[Range]: Long @ {bar.close} (near lower band)")
        
        elif bar.close >= upper[-1] and current_pos >= 0:
            if current_pos > 0:
                self.sell(symbol, bar.close, current_pos)
            self.short(symbol, bar.close, 1)
            logger.info(f"{symbol} Adaptive[Range]: Short @ {bar.close} (near upper band)")
        
        elif middle[-1] * 0.99 < bar.close < middle[-1] * 1.01 and current_pos != 0:
            # 回到中轨附近，平仓
            if current_pos > 0:
                self.sell(symbol, bar.close, current_pos)
            elif current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
            logger.info(f"{symbol} Adaptive[Range]: Close @ {bar.close} (near middle)")
    
    def _execute_volatile_strategy(self, symbol: str, bar: BarData):
        """高波动市场策略 - 减仓或观望"""
        position = self.get_position(symbol)
        
        if position and position.volume > 0:
            # 高波动时减仓
            reduce_volume = max(1, position.volume // 2)
            if position.direction == Direction.LONG:
                self.sell(symbol, bar.close, reduce_volume)
            else:
                self.cover(symbol, bar.close, reduce_volume)
            logger.info(f"{symbol} Adaptive[Volatile]: Reduce position by {reduce_volume}")
