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
    多策略组合 - 增强版
    同时运行多个策略，并进行智能资金分配
    
    特性：
    - 基于风险调整收益的动态权重
    - 策略相关性考虑
    - 回撤控制
    - 策略表现衰退检测
    """
    
    strategy_name = "MultiStrategyPortfolio"
    
    def __init__(
        self,
        strategies: List = None,  # 可以是 List[BaseStrategy] 或 List[Tuple[BaseStrategy, float]]
        rebalance_period: int = 20,  # 再平衡周期（K线数）
        min_weight: float = 0.05,    # 最小权重
        max_weight: float = 0.40,    # 最大权重
        use_risk_parity: bool = True,  # 使用风险平价
        decay_detection: bool = True,  # 策略衰退检测
        name: str = None,            # 组合名称（兼容参数）
        symbols: List[str] = None    # 交易品种（兼容参数）
    ):
        super().__init__()
        
        # 处理兼容性参数
        if name:
            self.strategy_name = name
        if symbols:
            self.symbols = symbols
        
        # 处理策略列表格式
        if strategies is None:
            strategies = []
        
        # 兼容两种输入格式
        processed_strategies = []
        if strategies and len(strategies) > 0:
            if isinstance(strategies[0], tuple):
                # 已经是 (strategy, weight) 格式
                processed_strategies = strategies
            else:
                # 是纯策略列表，自动分配等权重
                equal_weight = 1.0 / len(strategies)
                processed_strategies = [(s, equal_weight) for s in strategies]
        
        self.strategies = processed_strategies
        self.rebalance_period = rebalance_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_risk_parity = use_risk_parity
        self.decay_detection = decay_detection
        
        # 验证资金比例
        if self.strategies:
            total_weight = sum(weight for _, weight in self.strategies)
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Strategy weights sum to {total_weight}, normalizing...")
                # 归一化
                self.strategies = [(s, w/total_weight) for s, w in self.strategies]
        
        self._bar_count = 0
        
        # 策略绩效跟踪
        self._strategy_returns: Dict[str, List[float]] = {}
        self._strategy_allocations: Dict[str, float] = {}
        self._strategy_volatilities: Dict[str, float] = {}
        self._strategy_drawdowns: Dict[str, float] = {}
        self._strategy_peak_values: Dict[str, float] = {}
        
        for strategy, weight in self.strategies:
            self._strategy_allocations[strategy.strategy_name] = weight
            self._strategy_returns[strategy.strategy_name] = []
            self._strategy_volatilities[strategy.strategy_name] = 0.02
            self._strategy_drawdowns[strategy.strategy_name] = 0
            self._strategy_peak_values[strategy.strategy_name] = 1.0
    
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
            self._rebalance_enhanced()
    
    def _calculate_strategy_metrics(self, name: str) -> Dict[str, float]:
        """计算策略的各项指标"""
        returns = self._strategy_returns.get(name, [])
        
        if len(returns) < 10:
            return {
                "sharpe": 0,
                "sortino": 0,
                "volatility": 0.02,
                "drawdown": 0,
                "win_rate": 0.5
            }
        
        returns_arr = np.array(returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)
        
        # 夏普比率
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = returns_arr[returns_arr < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_ret
        sortino = mean_ret / downside_std if downside_std > 0 else 0
        
        # 胜率
        win_rate = np.sum(returns_arr > 0) / len(returns_arr)
        
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "volatility": std_ret,
            "drawdown": self._strategy_drawdowns.get(name, 0),
            "win_rate": win_rate
        }
    
    def _detect_strategy_decay(self, name: str) -> bool:
        """检测策略是否出现衰退"""
        if not self.decay_detection:
            return False
        
        returns = self._strategy_returns.get(name, [])
        if len(returns) < 40:
            return False
        
        # 比较近期表现和历史表现
        recent_returns = returns[-20:]
        historical_returns = returns[-40:-20]
        
        recent_sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
        historical_sharpe = np.mean(historical_returns) / (np.std(historical_returns) + 1e-8)
        
        # 如果近期夏普明显低于历史水平，认为策略衰退
        if recent_sharpe < historical_sharpe * 0.5:
            return True
        
        # 检查连续亏损
        consecutive_losses = 0
        for ret in reversed(recent_returns):
            if ret < 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= 5:
            return True
        
        return False
    
    def _rebalance_enhanced(self):
        """
        增强版再平衡
        综合考虑多个因素：
        1. 风险调整收益（夏普/索提诺）
        2. 波动率（风险平价）
        3. 回撤控制
        4. 策略衰退检测
        """
        logger.info("Portfolio rebalancing (enhanced)...")
        
        # 计算各策略指标
        strategy_scores = {}
        for strategy, _ in self.strategies:
            name = strategy.strategy_name
            metrics = self._calculate_strategy_metrics(name)
            
            # 综合评分 = 夏普 * 0.4 + 索提诺 * 0.3 + 胜率 * 0.2 - 回撤 * 0.1
            score = (
                metrics["sharpe"] * 0.4 +
                metrics["sortino"] * 0.3 +
                metrics["win_rate"] * 0.2 -
                metrics["drawdown"] * 0.1
            )
            
            # 策略衰退惩罚
            if self._detect_strategy_decay(name):
                score *= 0.5
                logger.warning(f"Strategy {name} shows decay signs, reducing weight")
            
            strategy_scores[name] = max(score, 0.01)  # 最小分数
            self._strategy_volatilities[name] = metrics["volatility"]
        
        if not strategy_scores:
            return
        
        # 计算新权重
        if self.use_risk_parity:
            # 风险平价：波动率越低，权重越高
            inv_vols = {
                name: 1.0 / (self._strategy_volatilities[name] + 0.01)
                for name in strategy_scores
            }
            total_inv_vol = sum(inv_vols.values())
            
            # 结合评分和风险平价
            new_weights = {}
            for name in strategy_scores:
                risk_parity_weight = inv_vols[name] / total_inv_vol
                score_weight = strategy_scores[name] / sum(strategy_scores.values())
                # 50% 风险平价 + 50% 评分
                new_weights[name] = 0.5 * risk_parity_weight + 0.5 * score_weight
        else:
            # 纯评分权重
            total_score = sum(strategy_scores.values())
            new_weights = {
                name: score / total_score
                for name, score in strategy_scores.items()
            }
        
        # 应用权重限制
        for name in new_weights:
            new_weights[name] = max(self.min_weight, min(new_weights[name], self.max_weight))
        
        # 归一化
        total_weight = sum(new_weights.values())
        for name in new_weights:
            new_weights[name] /= total_weight
        
        # 平滑调整（避免剧烈变化）
        for name in new_weights:
            old_weight = self._strategy_allocations.get(name, new_weights[name])
            self._strategy_allocations[name] = 0.7 * old_weight + 0.3 * new_weights[name]
        
        logger.info(f"New allocations: {self._strategy_allocations}")
    
    def on_trade(self, trade):
        """记录交易收益用于再平衡"""
        super().on_trade(trade)
        
        if trade.strategy_name in self._strategy_returns:
            # 计算收益率
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                # 如果有盈亏数据
                pnl_pct = trade.pnl / (trade.price * trade.volume + 1e-8)
            else:
                # 简化：使用相对价格变化
                pnl_pct = 0.0
            
            self._strategy_returns[trade.strategy_name].append(pnl_pct)
            
            # 更新策略的峰值和回撤
            name = trade.strategy_name
            cum_value = 1.0 + sum(self._strategy_returns[name])
            
            if cum_value > self._strategy_peak_values.get(name, 1.0):
                self._strategy_peak_values[name] = cum_value
            
            peak = self._strategy_peak_values.get(name, 1.0)
            current_dd = (peak - cum_value) / peak
            self._strategy_drawdowns[name] = current_dd
            
            # 保持最近的收益记录（避免内存增长）
            if len(self._strategy_returns[name]) > 200:
                self._strategy_returns[name] = self._strategy_returns[name][-100:]
    
    def update_strategy_return(self, strategy_name: str, return_pct: float):
        """
        外部接口：更新策略收益
        可以在策略外部调用，用于更准确的收益追踪
        """
        if strategy_name in self._strategy_returns:
            self._strategy_returns[strategy_name].append(return_pct)
            
            # 更新回撤
            cum_value = 1.0 + sum(self._strategy_returns[strategy_name])
            if cum_value > self._strategy_peak_values.get(strategy_name, 1.0):
                self._strategy_peak_values[strategy_name] = cum_value
            
            peak = self._strategy_peak_values.get(strategy_name, 1.0)
            self._strategy_drawdowns[strategy_name] = (peak - cum_value) / peak
    
    def get_strategy_status(self) -> Dict[str, Dict]:
        """获取所有策略的当前状态"""
        status = {}
        for strategy, _ in self.strategies:
            name = strategy.strategy_name
            metrics = self._calculate_strategy_metrics(name)
            status[name] = {
                "allocation": self._strategy_allocations.get(name, 0),
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "volatility": metrics["volatility"],
                "drawdown": metrics["drawdown"],
                "win_rate": metrics["win_rate"],
                "is_decaying": self._detect_strategy_decay(name)
            }
        return status


class AdaptiveStrategy(BaseStrategy):
    """
    自适应策略 - 增强版
    根据市场状态自动切换策略模式
    添加多时间框架确认和智能仓位管理
    """
    
    strategy_name = "Adaptive"
    
    def __init__(
        self,
        symbols: List[str],
        trend_period: int = 20,
        volatility_period: int = 20,
        trend_threshold: float = 0.015,  # 更敏感的趋势识别
        volatility_threshold: float = 0.025,  # 更保守的波动率阈值
        use_mtf_confirmation: bool = True  # 多时间框架确认
    ):
        super().__init__()
        self.symbols = symbols
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.use_mtf_confirmation = use_mtf_confirmation
        
        self.parameters = {
            "trend_period": trend_period,
            "volatility_period": volatility_period,
            "trend_threshold": trend_threshold,
            "volatility_threshold": volatility_threshold
        }
        
        # 市场状态
        self._market_state: Dict[str, str] = {}  # trending / ranging / volatile
        self._state_confidence: Dict[str, float] = {}  # 状态置信度
        
        # 各状态对应的策略逻辑
        self._trend_ma_fast = 5
        self._trend_ma_slow = 20
        self._range_upper_std = 2.0
        self._range_lower_std = -2.0
        
        # 多时间框架周期
        self._mtf_period = 50  # 更长周期用于确认
        
        # 状态持续计数
        self._state_duration: Dict[str, int] = {}
        self._min_state_duration = 3  # 最少持续3根K线才切换策略
        
        # 每个状态的仓位系数
        self._state_position_factor = {
            "trending": 1.0,    # 趋势市场全仓
            "ranging": 0.7,     # 震荡市场7成仓
            "volatile": 0.3     # 高波动3成仓
        }
    
    def on_bar(self, bars: Dict[str, BarData]):
        for symbol, bar in bars.items():
            if symbol not in self.symbols:
                continue
            
            # 更新K线缓冲
            if symbol not in self._bar_buffers:
                self._bar_buffers[symbol] = DataBuffer(max_size=max(100, self._mtf_period + 20))
            self._bar_buffers[symbol].append(bar)
            self.update_bar_counter(symbol)
            
            buffer = self._bar_buffers[symbol]
            if len(buffer) < max(self.trend_period, self.volatility_period, self._mtf_period):
                continue
            
            # 检查追踪止损
            position = self.get_position(symbol)
            if position and position.volume > 0:
                direction = position.direction
                stop_price = self.update_trailing_stop(symbol, bar.close, direction)
                if stop_price is not None:
                    # 触发追踪止损
                    if direction == Direction.LONG:
                        self.sell(symbol, bar.close, position.volume)
                    else:
                        self.cover(symbol, bar.close, position.volume)
                    self.clear_trailing_stop(symbol)
                    logger.info(f"{symbol} Adaptive Trailing Stop @ {bar.close}")
                    continue
            
            # 识别市场状态
            new_state, confidence = self._identify_market_state_enhanced(symbol)
            old_state = self._market_state.get(symbol, "")
            
            # 状态持续计数
            if new_state == old_state:
                self._state_duration[symbol] = self._state_duration.get(symbol, 0) + 1
            else:
                self._state_duration[symbol] = 1
            
            self._market_state[symbol] = new_state
            self._state_confidence[symbol] = confidence
            
            # 只有状态持续足够长才执行对应策略
            if self._state_duration.get(symbol, 0) < self._min_state_duration:
                continue
            
            # 根据市场状态执行相应策略
            if new_state == "trending":
                self._execute_trend_strategy_enhanced(symbol, bar, confidence)
            elif new_state == "ranging":
                self._execute_range_strategy_enhanced(symbol, bar, confidence)
            elif new_state == "volatile":
                self._execute_volatile_strategy_enhanced(symbol, bar)
    
    def _identify_market_state_enhanced(self, symbol: str) -> Tuple[str, float]:
        """识别市场状态 - 增强版，返回状态和置信度"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        highs = buffer.get_array("high")
        lows = buffer.get_array("low")
        
        # 计算短期趋势强度
        returns_short = np.diff(closes[-self.trend_period:]) / closes[-self.trend_period:-1]
        trend_strength_short = np.mean(returns_short)
        
        # 计算长期趋势强度（多时间框架确认）
        if self.use_mtf_confirmation and len(closes) >= self._mtf_period:
            returns_long = np.diff(closes[-self._mtf_period:]) / closes[-self._mtf_period:-1]
            trend_strength_long = np.mean(returns_long)
        else:
            trend_strength_long = trend_strength_short
        
        # 计算波动率
        volatility = np.std(returns_short)
        
        # 计算ADX类似的趋势强度指标
        atr = self.atr(highs, lows, closes, 14)[-1]
        if not np.isnan(atr) and closes[-1] > 0:
            normalized_atr = atr / closes[-1]
        else:
            normalized_atr = volatility
        
        # 判断市场状态和置信度
        confidence = 0.5  # 默认置信度
        
        if volatility > self.volatility_threshold or normalized_atr > 0.03:
            state = "volatile"
            confidence = min(volatility / self.volatility_threshold, 1.0)
        elif abs(trend_strength_short) > self.trend_threshold:
            # 需要长短期趋势方向一致
            if self.use_mtf_confirmation:
                if trend_strength_short * trend_strength_long > 0:  # 同向
                    state = "trending"
                    confidence = min(abs(trend_strength_short) / self.trend_threshold, 1.0)
                else:
                    state = "ranging"  # 长短期不一致，视为震荡
                    confidence = 0.6
            else:
                state = "trending"
                confidence = min(abs(trend_strength_short) / self.trend_threshold, 1.0)
        else:
            state = "ranging"
            confidence = 1.0 - abs(trend_strength_short) / self.trend_threshold
        
        return state, confidence
    
    def _execute_trend_strategy_enhanced(self, symbol: str, bar: BarData, confidence: float):
        """趋势市场策略 - 增强版"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        
        ma_fast = self.sma(closes, self._trend_ma_fast)[-1]
        ma_slow = self.sma(closes, self._trend_ma_slow)[-1]
        
        if np.isnan(ma_fast) or np.isnan(ma_slow):
            return
        
        position = self.get_position(symbol)
        current_pos = 0
        if position:
            if position.direction == Direction.LONG:
                current_pos = position.volume
            else:
                current_pos = -position.volume
        
        if ma_fast > ma_slow and current_pos <= 0:
            if not self.check_signal_filter(symbol):
                return
            # 根据置信度调整仓位
            base_size = self.calculate_position_size(symbol, 1)
            size = max(1, int(base_size * confidence * self._state_position_factor["trending"]))
            
            if current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
                self.clear_trailing_stop(symbol)
            self.buy(symbol, bar.close, size)
            self.init_trailing_stop(symbol, bar.close, Direction.LONG)
            self.record_signal(symbol)
            logger.info(f"{symbol} Adaptive[Trend]: Long @ {bar.close}, size={size}, conf={confidence:.2f}")
        
        elif ma_fast < ma_slow and current_pos >= 0:
            if not self.check_signal_filter(symbol):
                return
            # 根据置信度调整仓位
            base_size = self.calculate_position_size(symbol, 1)
            size = max(1, int(base_size * confidence * self._state_position_factor["trending"]))
            
            if current_pos > 0:
                self.sell(symbol, bar.close, current_pos)
                self.clear_trailing_stop(symbol)
            self.short(symbol, bar.close, size)
            self.init_trailing_stop(symbol, bar.close, Direction.SHORT)
            self.record_signal(symbol)
            logger.info(f"{symbol} Adaptive[Trend]: Short @ {bar.close}, size={size}, conf={confidence:.2f}")
    
    def _execute_range_strategy_enhanced(self, symbol: str, bar: BarData, confidence: float):
        """震荡市场策略 - 增强版"""
        buffer = self._bar_buffers[symbol]
        closes = buffer.get_array("close")
        
        upper, middle, lower = self.bollinger_bands(closes, 20, 2.0)
        
        if np.isnan(upper[-1]) or np.isnan(lower[-1]):
            return
        
        position = self.get_position(symbol)
        current_pos = 0
        if position:
            if position.direction == Direction.LONG:
                current_pos = position.volume
            else:
                current_pos = -position.volume
        
        # 震荡策略使用更保守的仓位
        base_size = self.calculate_position_size(symbol, 1)
        size = max(1, int(base_size * self._state_position_factor["ranging"]))
        
        if bar.close <= lower[-1] and current_pos <= 0:
            if not self.check_signal_filter(symbol):
                return
            if current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
                self.clear_trailing_stop(symbol)
            self.buy(symbol, bar.close, size)
            self.init_trailing_stop(symbol, bar.close, Direction.LONG)
            self.record_signal(symbol)
            logger.info(f"{symbol} Adaptive[Range]: Long @ {bar.close} (near lower band), size={size}")
        
        elif bar.close >= upper[-1] and current_pos >= 0:
            if not self.check_signal_filter(symbol):
                return
            if current_pos > 0:
                self.sell(symbol, bar.close, current_pos)
                self.clear_trailing_stop(symbol)
            self.short(symbol, bar.close, size)
            self.init_trailing_stop(symbol, bar.close, Direction.SHORT)
            self.record_signal(symbol)
            logger.info(f"{symbol} Adaptive[Range]: Short @ {bar.close} (near upper band), size={size}")
        
        elif middle[-1] * 0.995 < bar.close < middle[-1] * 1.005 and current_pos != 0:
            # 回到中轨附近，平仓（更精确的判断）
            if current_pos > 0:
                self.sell(symbol, bar.close, abs(current_pos))
            elif current_pos < 0:
                self.cover(symbol, bar.close, abs(current_pos))
            self.clear_trailing_stop(symbol)
            logger.info(f"{symbol} Adaptive[Range]: Close @ {bar.close} (near middle)")
    
    def _execute_volatile_strategy_enhanced(self, symbol: str, bar: BarData):
        """高波动市场策略 - 增强版：更积极地减仓和风险控制"""
        position = self.get_position(symbol)
        
        if position and position.volume > 0:
            # 高波动时大幅减仓
            reduce_ratio = self._state_position_factor["volatile"]
            target_volume = max(0, int(position.volume * reduce_ratio))
            reduce_volume = position.volume - target_volume
            
            if reduce_volume > 0:
                if position.direction == Direction.LONG:
                    self.sell(symbol, bar.close, reduce_volume)
                else:
                    self.cover(symbol, bar.close, reduce_volume)
                logger.info(f"{symbol} Adaptive[Volatile]: Reduce position by {reduce_volume}")
            
            # 更新追踪止损（高波动时使用更紧的止损）
            if symbol in self._trailing_stops:
                # 缩紧止损距离
                self._atr_stop_multiple = 1.5  # 临时使用更紧的止损
                self.update_trailing_stop(symbol, bar.close, position.direction)
                self._atr_stop_multiple = 2.0  # 恢复


# ==================== AI增强策略 ====================

class AIEnhancedStrategy(BaseStrategy):
    """
    AI增强策略
    结合深度学习预测和传统技术指标的智能交易策略
    
    特性：
    - 使用LSTM/Transformer预测价格方向
    - 预测信号与技术指标双重确认
    - 动态置信度调整仓位
    - 自适应学习市场状态
    """
    
    strategy_name = "AIEnhanced"
    
    def __init__(
        self,
        symbols: List[str],
        model_type: str = "lstm",  # lstm / transformer / ensemble
        prediction_threshold: float = 0.6,  # 预测置信度阈值
        use_technical_filter: bool = True,  # 是否使用技术指标过滤
        retrain_period: int = 500,  # 重新训练周期
        lookback_period: int = 60,  # 回看周期
    ):
        super().__init__()
        self.symbols = symbols
        self.model_type = model_type
        self.prediction_threshold = prediction_threshold
        self.use_technical_filter = use_technical_filter
        self.retrain_period = retrain_period
        self.lookback_period = lookback_period
        
        self.parameters = {
            "model_type": model_type,
            "prediction_threshold": prediction_threshold,
            "use_technical_filter": use_technical_filter,
            "retrain_period": retrain_period,
        }
        
        # AI相关
        self._prediction_manager = None
        self._bar_count = 0
        self._is_model_trained = False
        
        # 数据缓存
        self._data_cache: Dict[str, List[Dict]] = {}
        for symbol in symbols:
            self._data_cache[symbol] = []
        
        # 预测历史
        self._predictions: Dict[str, List[float]] = {s: [] for s in symbols}
        self._prediction_accuracy: Dict[str, float] = {s: 0.5 for s in symbols}
    
    def on_init(self):
        """策略初始化"""
        super().on_init()
        
        try:
            from ..ai.predictors import PredictionManager, ModelConfig
            
            config = ModelConfig(
                sequence_length=self.lookback_period,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                epochs=50,
                early_stopping_patience=5
            )
            self._prediction_manager = PredictionManager(config)
            logger.info(f"AI Enhanced Strategy initialized with {self.model_type} model")
        except ImportError as e:
            logger.warning(f"AI module not available: {e}")
            self._prediction_manager = None
    
    def on_bar(self, bar: BarData):
        """处理K线数据"""
        symbol = bar.symbol
        if symbol not in self.symbols:
            return
        
        self._bar_count += 1
        self.update_bar_counter()
        
        # 缓存数据
        self._cache_bar_data(symbol, bar)
        
        # 更新追踪止损
        position = self.get_position(symbol)
        if position and position.volume > 0:
            self.update_trailing_stop(symbol, bar.close, position.direction)
        
        # 检查是否需要训练模型
        if self._bar_count % self.retrain_period == 0:
            self._train_models()
        
        # 生成交易信号
        if self._is_model_trained and len(self._data_cache[symbol]) >= self.lookback_period:
            self._generate_ai_signal(symbol, bar)
    
    def _cache_bar_data(self, symbol: str, bar: BarData):
        """缓存K线数据"""
        bar_dict = {
            'datetime': bar.datetime,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        self._data_cache[symbol].append(bar_dict)
        
        # 保留最近的数据
        max_cache = self.retrain_period + self.lookback_period + 100
        if len(self._data_cache[symbol]) > max_cache:
            self._data_cache[symbol] = self._data_cache[symbol][-max_cache:]
    
    def _train_models(self):
        """训练预测模型"""
        if self._prediction_manager is None:
            return
        
        for symbol in self.symbols:
            if len(self._data_cache[symbol]) < self.lookback_period + 50:
                continue
            
            try:
                import pandas as pd
                df = pd.DataFrame(self._data_cache[symbol])
                df.set_index('datetime', inplace=True)
                
                self._prediction_manager.train_model(
                    df, 
                    model_type=self.model_type,
                    target_column="returns"
                )
                self._is_model_trained = True
                logger.info(f"AI model trained for {symbol}")
            except Exception as e:
                logger.error(f"Failed to train AI model for {symbol}: {e}")
    
    def _generate_ai_signal(self, symbol: str, bar: BarData):
        """生成AI交易信号"""
        if self._prediction_manager is None or not self._is_model_trained:
            return
        
        try:
            import pandas as pd
            
            # 准备预测数据
            df = pd.DataFrame(self._data_cache[symbol])
            df.set_index('datetime', inplace=True)
            
            # 获取预测
            predictions = self._prediction_manager.predict(df)
            
            if len(predictions) == 0:
                return
            
            # 获取最新预测
            latest_pred = predictions[-1]
            self._predictions[symbol].append(latest_pred)
            
            # 计算预测置信度
            confidence = abs(latest_pred)
            
            # 技术指标过滤
            tech_signal = 1  # 默认通过
            if self.use_technical_filter:
                tech_signal = self._get_technical_signal(symbol, bar)
            
            # 综合信号
            position = self.get_position(symbol)
            current_pos = 0
            if position:
                current_pos = position.volume if position.direction == Direction.LONG else -position.volume
            
            # 信号过滤检查
            if not self.check_signal_filter(symbol):
                return
            
            # 计算仓位大小（基于置信度）
            base_size = self.calculate_position_size(symbol, 1)
            confidence_factor = min(1.0, confidence / 0.02)  # 归一化置信度
            size = max(1, int(base_size * confidence_factor))
            
            # 执行交易
            if latest_pred > 0.005 and confidence > self.prediction_threshold / 100:
                # 预测上涨
                if tech_signal >= 0 and current_pos <= 0:
                    if current_pos < 0:
                        self.cover(symbol, bar.close, abs(current_pos))
                        self.clear_trailing_stop(symbol)
                    self.buy(symbol, bar.close, size)
                    self.init_trailing_stop(symbol, bar.close, Direction.LONG)
                    self.record_signal(symbol)
                    logger.info(f"{symbol} AI: Long @ {bar.close}, pred={latest_pred:.4f}, conf={confidence:.4f}")
            
            elif latest_pred < -0.005 and confidence > self.prediction_threshold / 100:
                # 预测下跌
                if tech_signal <= 0 and current_pos >= 0:
                    if current_pos > 0:
                        self.sell(symbol, bar.close, current_pos)
                        self.clear_trailing_stop(symbol)
                    self.short(symbol, bar.close, size)
                    self.init_trailing_stop(symbol, bar.close, Direction.SHORT)
                    self.record_signal(symbol)
                    logger.info(f"{symbol} AI: Short @ {bar.close}, pred={latest_pred:.4f}, conf={confidence:.4f}")
            
            # 更新预测准确率
            self._update_prediction_accuracy(symbol)
            
        except Exception as e:
            logger.error(f"AI signal generation failed for {symbol}: {e}")
    
    def _get_technical_signal(self, symbol: str, bar: BarData) -> int:
        """
        获取技术指标信号
        返回: 1=看多, -1=看空, 0=中性
        """
        closes = self.get_array(symbol, "close", 30)
        if closes is None or len(closes) < 30:
            return 0
        
        # 短期均线
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        
        # 趋势判断
        trend = 0
        if ma5 > ma10 > ma20:
            trend = 1  # 上升趋势
        elif ma5 < ma10 < ma20:
            trend = -1  # 下降趋势
        
        # RSI
        rsi = self._calculate_rsi(closes, 14)
        
        rsi_signal = 0
        if rsi < 30:
            rsi_signal = 1  # 超卖
        elif rsi > 70:
            rsi_signal = -1  # 超买
        
        # 综合信号
        if trend == 1 and rsi_signal >= 0:
            return 1
        elif trend == -1 and rsi_signal <= 0:
            return -1
        
        return 0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_prediction_accuracy(self, symbol: str):
        """更新预测准确率"""
        preds = self._predictions[symbol]
        if len(preds) < 10:
            return
        
        closes = self.get_array(symbol, "close", len(preds) + 1)
        if closes is None or len(closes) < len(preds) + 1:
            return
        
        # 计算实际收益率
        actual_returns = np.diff(closes[-len(preds)-1:]) / closes[-len(preds)-1:-1]
        
        # 方向准确率
        pred_directions = np.array(preds[-len(actual_returns):]) > 0
        actual_directions = actual_returns > 0
        
        accuracy = np.mean(pred_directions == actual_directions)
        self._prediction_accuracy[symbol] = accuracy
        
        if len(preds) % 50 == 0:
            logger.info(f"{symbol} AI prediction accuracy: {accuracy:.2%}")
    
    def get_ai_status(self) -> Dict[str, Any]:
        """获取AI策略状态"""
        return {
            "model_trained": self._is_model_trained,
            "model_type": self.model_type,
            "bar_count": self._bar_count,
            "prediction_accuracy": self._prediction_accuracy,
            "symbols": self.symbols
        }
