"""
风险控制系统 - 多层次风险管理
包含事前风控、事中风控、事后风控
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

from ..data.data_structures import (
    OrderData, PositionData, AccountData, BarData, TickData,
    Direction, Offset, OrderStatus
)

logger = logging.getLogger(__name__)


# ==================== 风险类型枚举 ====================

class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    CRITICAL = "极端风险"

class RiskAction(Enum):
    """风险处理动作"""
    PASS = "通过"
    WARN = "警告"
    REDUCE = "减仓"
    REJECT = "拒绝"
    LIQUIDATE = "强平"


# ==================== 风险检查结果 ====================

@dataclass
class RiskCheckResult:
    """风险检查结果"""
    passed: bool = True
    action: RiskAction = RiskAction.PASS
    level: RiskLevel = RiskLevel.LOW
    messages: List[str] = field(default_factory=list)
    suggested_volume: int = 0  # 建议的交易量（如果需要减仓）
    
    def add_message(self, msg: str):
        self.messages.append(msg)
    
    def fail(self, action: RiskAction, level: RiskLevel, msg: str):
        self.passed = False
        self.action = action
        self.level = level
        self.messages.append(msg)


# ==================== 风控配置 ====================

@dataclass
class RiskManagerConfig:
    """风控配置 - 优化版本"""
    # 单笔风险（更保守）
    max_loss_per_trade: float = 0.015      # 单笔最大亏损1.5%（从2%降低）
    max_order_value: float = 0.08          # 单笔最大委托8%（从10%降低）
    
    # 日内风险
    max_daily_loss: float = 0.04           # 日内最大亏损4%（从5%降低）
    max_daily_trades: int = 30             # 日内最大交易30次（从50降低）
    max_daily_turnover: float = 3.0        # 日内最大换手率3倍（从5降低）
    
    # 仓位风险（更分散）
    max_position_per_symbol: float = 0.12  # 单品种最大12%（从15%降低）
    max_total_position: float = 0.70       # 总仓位70%（从80%降低）
    max_correlated_exposure: float = 0.25  # 相关品种最大25%（从30%降低）
    
    # 杠杆风险（更保守）
    max_leverage: float = 2.5              # 最大杠杆2.5倍（从3倍降低）
    
    # 流动性风险
    min_volume_threshold: int = 3000       # 最小成交量3000（从2000提高）
    max_volume_participation: float = 0.03 # 最大成交量占比3%（从5%降低）
    
    # 回撤风险（更严格）
    max_drawdown: float = 0.08             # 最大回撤8%（从10%降低）
    drawdown_reduce_ratio: float = 0.6     # 回撤达阈值时减仓60%（从50%提高）
    drawdown_warning_ratio: float = 0.6    # 回撤预警阈值（60%时开始警告）
    
    # 波动风险
    volatility_scaling: bool = True        # 启用波动率缩放
    max_volatility_multiple: float = 1.5   # 最大波动率倍数1.5（从2降低）
    min_volatility_multiple: float = 0.5   # 最小波动率倍数
    
    # 熔断配置（更敏感）
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.06  # 6%亏损触发熔断（从7%降低）
    circuit_breaker_duration_hours: int = 24 # 熔断持续24小时
    
    # 连续亏损控制
    max_consecutive_losses: int = 4        # 最大连续亏损次数
    consecutive_loss_reduce_ratio: float = 0.5  # 连续亏损时减仓50%
    
    # 盈利保护
    profit_protection_threshold: float = 0.10  # 盈利10%后启用保护
    profit_protection_ratio: float = 0.50      # 保护50%已有盈利
    
    # 相关性控制
    correlation_check_enabled: bool = True
    max_correlation_threshold: float = 0.7  # 相关性超过0.7时限制开仓


# ==================== 仓位管理器 ====================

class PositionSizer:
    """
    仓位管理器
    基于Kelly公式和风险预算动态计算仓位
    """
    
    def __init__(self, config: RiskManagerConfig):
        self.config = config
        self._strategy_stats: Dict[str, Dict] = {}
    
    def update_strategy_stats(
        self, 
        strategy_name: str, 
        win_rate: float, 
        profit_factor: float,
        avg_win: float,
        avg_loss: float
    ):
        """更新策略统计数据"""
        self._strategy_stats[strategy_name] = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }
    
    def calculate_kelly_fraction(self, strategy_name: str) -> float:
        """
        计算Kelly最优仓位比例 - 改进版
        
        Kelly公式: f* = (p * b - q) / b
        考虑估计误差，使用更保守的分数Kelly
        """
        stats = self._strategy_stats.get(strategy_name)
        if not stats or stats["win_rate"] == 0:
            return 0.01  # 默认1%仓位（更保守）
        
        p = stats["win_rate"]
        q = 1 - p
        
        # 使用盈亏比而非profit_factor
        if stats["avg_loss"] == 0:
            return 0.01
        
        b = abs(stats["avg_win"] / stats["avg_loss"])
        
        if b <= 0 or p <= 0:
            return 0.01
        
        # Kelly公式
        kelly = (p * b - q) / b
        
        # 考虑估计误差：样本量越小，越保守
        # 假设至少需要30个样本才能合理估计
        sample_adj = min(1.0, self.trade_count.get(strategy_name, 10) / 30.0)
        
        # 使用1/4 Kelly（更保守，减少方差）
        kelly = kelly * 0.25 * sample_adj
        
        # 严格限制范围
        kelly = max(0.005, min(kelly, 0.15))
        
        return kelly
    
    @property
    def trade_count(self) -> Dict[str, int]:
        """获取各策略交易次数"""
        return getattr(self, '_trade_counts', {})
    
    def calculate_volatility_adjusted_size(
        self,
        base_size: int,
        current_volatility: float,
        target_volatility: float = 0.02
    ) -> int:
        """
        根据波动率调整仓位
        高波动时减仓，低波动时可适当加仓
        """
        if not self.config.volatility_scaling:
            return base_size
        
        if current_volatility <= 0:
            return base_size
        
        # 波动率调整因子
        vol_factor = target_volatility / current_volatility
        
        # 限制调整幅度
        vol_factor = max(0.5, min(vol_factor, self.config.max_volatility_multiple))
        
        adjusted_size = int(base_size * vol_factor)
        return max(1, adjusted_size)
    
    def calculate_optimal_size(
        self,
        account: AccountData,
        symbol: str,
        price: float,
        contract_size: float,
        margin_ratio: float,
        strategy_name: str = "",
        current_volatility: float = 0
    ) -> int:
        """
        计算最优仓位
        
        综合考虑：
        - Kelly公式
        - 风险预算
        - 波动率调整
        - 最大仓位限制
        """
        available = account.available
        
        # 1. 基于风险预算计算
        risk_budget = available * self.config.max_loss_per_trade
        # 假设止损点为2个ATR，每单位合约的风险
        per_unit_risk = price * contract_size * 0.02  # 简化计算
        if per_unit_risk > 0:
            risk_based_size = int(risk_budget / per_unit_risk)
        else:
            risk_based_size = 1
        
        # 2. 基于Kelly公式计算
        kelly_fraction = self.calculate_kelly_fraction(strategy_name)
        kelly_capital = available * kelly_fraction
        kelly_based_size = int(kelly_capital / (price * contract_size * margin_ratio))
        
        # 3. 取较小值
        base_size = min(risk_based_size, kelly_based_size)
        
        # 4. 波动率调整
        if current_volatility > 0:
            base_size = self.calculate_volatility_adjusted_size(
                base_size, current_volatility
            )
        
        # 5. 最大仓位限制
        max_position_value = available * self.config.max_position_per_symbol
        max_size = int(max_position_value / (price * contract_size * margin_ratio))
        
        final_size = min(base_size, max_size)
        
        return max(1, final_size)


# ==================== 风险管理器 ====================

class RiskManager:
    """
    风险管理器
    实现多层次风险控制
    """
    
    def __init__(self, config: RiskManagerConfig = None):
        self.config = config or RiskManagerConfig()
        self.position_sizer = PositionSizer(self.config)
        
        # 日内统计
        self._daily_trades: int = 0
        self._daily_pnl: float = 0
        self._daily_turnover: float = 0
        self._daily_date: Optional[datetime] = None
        
        # 回撤追踪
        self._peak_balance: float = 0
        self._current_drawdown: float = 0
        self._protected_profit: float = 0  # 受保护的利润
        
        # 连续亏损追踪
        self._consecutive_losses: int = 0
        self._last_trade_profit: float = 0
        
        # 品种相关性矩阵
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # 熔断状态
        self._circuit_breaker_triggered: bool = False
        self._circuit_breaker_until: Optional[datetime] = None
        
        # 风险事件日志
        self._risk_events: List[Dict] = []
        
        # 仓位快照（用于计算相关性风险）
        self._position_snapshot: Dict[str, float] = {}
        
        # 交易统计（用于Kelly公式）
        self._trade_counts: Dict[str, int] = {}
    
    def check_order(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        current_bar: Optional[BarData] = None
    ) -> RiskCheckResult:
        """
        事前风控 - 订单提交前检查（增强版）
        """
        result = RiskCheckResult()
        
        # 0. 熔断检查
        if self._check_circuit_breaker(result):
            return result
        
        # 0.1 连续亏损检查
        if self._check_consecutive_losses(result):
            return result
        
        # 1. 资金检查
        self._check_capital(order, account, result)
        if not result.passed:
            return result
        
        # 2. 单笔风险检查
        self._check_single_trade_risk(order, account, result)
        if not result.passed:
            return result
        
        # 3. 仓位检查
        self._check_position_limit(order, account, positions, result)
        if not result.passed:
            return result
        
        # 4. 日内交易限制检查
        self._check_daily_limit(result)
        if not result.passed:
            return result
        
        # 5. 流动性检查
        if current_bar:
            self._check_liquidity(order, current_bar, result)
            if not result.passed:
                return result
        
        # 6. 杠杆检查
        self._check_leverage(order, account, positions, result)
        if not result.passed:
            return result
        
        # 7. 相关性风险检查（增强）
        if self.config.correlation_check_enabled:
            self._check_correlation_risk_enhanced(order, account, positions, result)
            if not result.passed:
                return result
        
        # 8. 盈利保护检查
        self._check_profit_protection(account, result)
        
        return result
    
    def _check_circuit_breaker(self, result: RiskCheckResult) -> bool:
        """检查熔断状态"""
        if self._circuit_breaker_triggered:
            if self._circuit_breaker_until and datetime.now() < self._circuit_breaker_until:
                result.fail(
                    RiskAction.REJECT,
                    RiskLevel.CRITICAL,
                    f"熔断中，恢复时间: {self._circuit_breaker_until}"
                )
                return True
            else:
                # 熔断恢复
                self._circuit_breaker_triggered = False
                self._circuit_breaker_until = None
        return False
    
    def _check_capital(self, order: OrderData, account: AccountData, result: RiskCheckResult):
        """检查资金是否充足"""
        # 计算订单所需保证金（简化计算）
        margin_ratio = 0.1  # 假设10%保证金
        contract_size = 10  # 假设合约乘数10
        required_margin = order.price * order.volume * contract_size * margin_ratio
        
        if required_margin > account.available:
            result.fail(
                RiskAction.REJECT,
                RiskLevel.HIGH,
                f"资金不足: 需要{required_margin:.2f}, 可用{account.available:.2f}"
            )
    
    def _check_single_trade_risk(
        self, 
        order: OrderData, 
        account: AccountData, 
        result: RiskCheckResult
    ):
        """检查单笔交易风险"""
        contract_size = 10
        order_value = order.price * order.volume * contract_size
        
        # 单笔委托金额限制
        max_order_value = account.balance * self.config.max_order_value
        if order_value > max_order_value:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.MEDIUM,
                f"单笔委托金额超限: {order_value:.2f} > {max_order_value:.2f}"
            )
            # 计算建议的交易量
            result.suggested_volume = int(max_order_value / (order.price * contract_size))
    
    def _check_position_limit(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        result: RiskCheckResult
    ):
        """检查仓位限制"""
        # 只检查开仓订单
        if order.offset != Offset.OPEN:
            return
        
        contract_size = 10
        margin_ratio = 0.1
        
        # 单品种仓位检查
        symbol_position = positions.get(order.symbol)
        current_position_value = 0
        if symbol_position:
            current_position_value = (
                symbol_position.volume * symbol_position.price * contract_size * margin_ratio
            )
        
        new_position_value = order.price * order.volume * contract_size * margin_ratio
        total_symbol_value = current_position_value + new_position_value
        
        max_symbol_value = account.balance * self.config.max_position_per_symbol
        if total_symbol_value > max_symbol_value:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.MEDIUM,
                f"单品种仓位超限: {total_symbol_value:.2f} > {max_symbol_value:.2f}"
            )
            available_value = max_symbol_value - current_position_value
            result.suggested_volume = max(0, int(available_value / (order.price * contract_size * margin_ratio)))
            return
        
        # 总仓位检查
        total_position_value = sum(
            pos.volume * pos.price * contract_size * margin_ratio
            for pos in positions.values()
        )
        total_position_value += new_position_value
        
        max_total_value = account.balance * self.config.max_total_position
        if total_position_value > max_total_value:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.MEDIUM,
                f"总仓位超限: {total_position_value:.2f} > {max_total_value:.2f}"
            )
    
    def _check_daily_limit(self, result: RiskCheckResult):
        """检查日内交易限制"""
        if self._daily_trades >= self.config.max_daily_trades:
            result.fail(
                RiskAction.REJECT,
                RiskLevel.MEDIUM,
                f"日内交易次数超限: {self._daily_trades} >= {self.config.max_daily_trades}"
            )
    
    def _check_liquidity(
        self, 
        order: OrderData, 
        bar: BarData, 
        result: RiskCheckResult
    ):
        """检查流动性"""
        if bar.volume < self.config.min_volume_threshold:
            result.fail(
                RiskAction.WARN,
                RiskLevel.MEDIUM,
                f"品种流动性不足: 成交量{bar.volume} < {self.config.min_volume_threshold}"
            )
            return
        
        # 检查订单占成交量比例
        participation = order.volume / bar.volume if bar.volume > 0 else 1
        if participation > self.config.max_volume_participation:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.MEDIUM,
                f"订单量占比过大: {participation:.2%} > {self.config.max_volume_participation:.2%}"
            )
            result.suggested_volume = int(bar.volume * self.config.max_volume_participation)
    
    def _check_leverage(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        result: RiskCheckResult
    ):
        """检查杠杆"""
        contract_size = 10
        margin_ratio = 0.1
        
        # 计算当前杠杆
        total_exposure = sum(
            pos.volume * pos.price * contract_size
            for pos in positions.values()
        )
        
        if order.offset == Offset.OPEN:
            total_exposure += order.price * order.volume * contract_size
        
        current_leverage = total_exposure / account.balance if account.balance > 0 else 0
        
        if current_leverage > self.config.max_leverage:
            result.fail(
                RiskAction.REJECT,
                RiskLevel.HIGH,
                f"杠杆超限: {current_leverage:.2f}x > {self.config.max_leverage:.2f}x"
            )
    
    def _check_correlation_risk(
        self,
        order: OrderData,
        positions: Dict[str, PositionData],
        result: RiskCheckResult
    ):
        """检查相关性风险（同板块风险集中）"""
        # 简化实现：检查同类品种持仓
        # 实际应用中应使用真实的相关性矩阵
        
        # 定义板块
        sectors = {
            "黑色系": ["rb", "hc", "i", "j", "jm"],
            "有色金属": ["cu", "al", "zn", "pb", "ni"],
            "能源化工": ["sc", "bu", "fu", "ta", "ma", "eg"],
            "农产品": ["c", "cs", "a", "m", "y", "p"],
            "贵金属": ["au", "ag"],
        }
        
        # 获取订单品种所属板块
        symbol_lower = order.symbol.lower()
        order_sector = None
        for sector, symbols in sectors.items():
            if any(s in symbol_lower for s in symbols):
                order_sector = sector
                break
        
        if not order_sector:
            return
        
        # 计算同板块持仓
        sector_exposure = 0
        for pos_symbol, pos in positions.items():
            for s in sectors.get(order_sector, []):
                if s in pos_symbol.lower():
                    sector_exposure += pos.volume * pos.price * 10  # 简化计算
        
        if order.offset == Offset.OPEN:
            sector_exposure += order.price * order.volume * 10
        
        # 检查板块敞口
        # 这里简化处理，实际应该比较总资金
        if sector_exposure > 0:
            result.add_message(f"注意: {order_sector}板块敞口较大")
    
    def _check_consecutive_losses(self, result: RiskCheckResult) -> bool:
        """检查连续亏损"""
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            result.fail(
                RiskAction.WARN,
                RiskLevel.HIGH,
                f"连续亏损{self._consecutive_losses}次，建议暂停交易或减小仓位"
            )
            result.suggested_volume = 0  # 建议停止交易
            self._log_risk_event("CONSECUTIVE_LOSSES", {
                "count": self._consecutive_losses
            })
            return True
        elif self._consecutive_losses >= self.config.max_consecutive_losses * 0.75:
            result.add_message(f"警告: 已连续亏损{self._consecutive_losses}次")
        return False
    
    def _check_profit_protection(self, account: AccountData, result: RiskCheckResult):
        """盈利保护检查"""
        if self._peak_balance == 0:
            return
        
        total_return = (account.balance - self._peak_balance) / self._peak_balance
        
        # 如果有足够盈利，启用保护
        if total_return >= self.config.profit_protection_threshold:
            # 计算受保护的利润
            profit = account.balance - self._peak_balance
            self._protected_profit = profit * self.config.profit_protection_ratio
            
            # 当前回撤如果侵蚀受保护利润，发出警告
            current_profit = account.balance - self._peak_balance
            if current_profit < self._protected_profit:
                result.add_message(
                    f"盈利保护警告: 当前盈利{current_profit:.2f}低于保护线{self._protected_profit:.2f}"
                )
                result.action = RiskAction.WARN
                result.level = RiskLevel.MEDIUM
    
    def _check_correlation_risk_enhanced(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        result: RiskCheckResult
    ):
        """增强的相关性风险检查"""
        if order.offset != Offset.OPEN:
            return
        
        # 定义板块及相关性
        sectors = {
            "黑色系": ["rb", "hc", "i", "j", "jm"],
            "有色金属": ["cu", "al", "zn", "pb", "ni"],
            "能源化工": ["sc", "bu", "fu", "ta", "ma", "eg"],
            "农产品": ["c", "cs", "a", "m", "y", "p"],
            "贵金属": ["au", "ag"],
        }
        
        # 获取订单品种所属板块
        symbol_lower = order.symbol.lower()
        order_sector = None
        for sector, symbols in sectors.items():
            if any(s in symbol_lower for s in symbols):
                order_sector = sector
                break
        
        if not order_sector:
            return
        
        # 计算同板块持仓价值
        contract_size = 10
        margin_ratio = 0.1
        sector_exposure = 0
        sector_count = 0
        
        for pos_symbol, pos in positions.items():
            for s in sectors.get(order_sector, []):
                if s in pos_symbol.lower():
                    sector_exposure += pos.volume * pos.price * contract_size * margin_ratio
                    sector_count += 1
        
        # 添加新订单
        new_exposure = order.price * order.volume * contract_size * margin_ratio
        total_sector_exposure = sector_exposure + new_exposure
        
        # 检查板块集中度
        sector_ratio = total_sector_exposure / account.balance if account.balance > 0 else 0
        
        if sector_ratio > self.config.max_correlated_exposure:
            result.fail(
                RiskAction.REJECT,
                RiskLevel.HIGH,
                f"{order_sector}板块集中度过高: {sector_ratio:.2%} > {self.config.max_correlated_exposure:.2%}"
            )
            self._log_risk_event("SECTOR_CONCENTRATION", {
                "sector": order_sector,
                "ratio": sector_ratio,
                "threshold": self.config.max_correlated_exposure
            })
        elif sector_ratio > self.config.max_correlated_exposure * 0.8:
            result.add_message(f"警告: {order_sector}板块集中度{sector_ratio:.2%}接近上限")
            result.action = RiskAction.WARN
            result.level = RiskLevel.MEDIUM
    
    def update_daily_stats(self, trade_pnl: float, turnover: float, current_date: datetime):
        """更新日内统计"""
        # 日期变化时重置
        if self._daily_date is None or current_date.date() != self._daily_date.date():
            self._daily_trades = 0
            self._daily_pnl = 0
            self._daily_turnover = 0
            self._daily_date = current_date
            self._consecutive_losses = 0  # 每日重置连续亏损
        
        self._daily_trades += 1
        self._daily_pnl += trade_pnl
        self._daily_turnover += turnover
        
        # 更新连续亏损
        if trade_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        self._last_trade_profit = trade_pnl
    
    def check_drawdown(self, current_balance: float) -> RiskCheckResult:
        """
        事中风控 - 回撤检查
        """
        result = RiskCheckResult()
        
        # 更新峰值
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        
        # 计算当前回撤
        if self._peak_balance > 0:
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance
        
        # 检查回撤
        if self._current_drawdown >= self.config.max_drawdown:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.HIGH,
                f"回撤达到阈值: {self._current_drawdown:.2%} >= {self.config.max_drawdown:.2%}"
            )
            self._log_risk_event("DRAWDOWN_ALERT", {
                "drawdown": self._current_drawdown,
                "threshold": self.config.max_drawdown
            })
        elif self._current_drawdown >= self.config.max_drawdown * 0.8:
            result.add_message(f"警告: 回撤接近阈值 {self._current_drawdown:.2%}")
        
        return result
    
    def check_daily_loss(self, account: AccountData) -> RiskCheckResult:
        """
        事中风控 - 日内亏损检查
        """
        result = RiskCheckResult()
        
        daily_return = account.daily_return
        
        # 日内亏损检查
        if daily_return <= -self.config.max_daily_loss:
            # 触发熔断
            if self.config.circuit_breaker_enabled and daily_return <= -self.config.circuit_breaker_threshold:
                self._circuit_breaker_triggered = True
                self._circuit_breaker_until = datetime.now() + timedelta(hours=24)
                result.fail(
                    RiskAction.LIQUIDATE,
                    RiskLevel.CRITICAL,
                    f"触发熔断! 日内亏损: {daily_return:.2%}"
                )
                self._log_risk_event("CIRCUIT_BREAKER", {
                    "daily_loss": daily_return,
                    "until": self._circuit_breaker_until.isoformat()
                })
            else:
                result.fail(
                    RiskAction.REDUCE,
                    RiskLevel.HIGH,
                    f"日内亏损超限: {daily_return:.2%} <= {-self.config.max_daily_loss:.2%}"
                )
        
        return result
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        atr_multiple: float = 2.0
    ) -> float:
        """
        计算动态止损价
        基于ATR（平均真实波幅）
        """
        stop_distance = atr * atr_multiple
        
        if direction == Direction.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self,
        entry_price: float,
        direction: Direction,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        计算止盈价
        基于风险收益比
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if direction == Direction.LONG:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def generate_risk_report(self, account: AccountData) -> Dict[str, Any]:
        """
        生成风险报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "account_status": {
                "balance": account.balance,
                "available": account.available,
                "margin_ratio": account.margin_ratio,
                "daily_return": account.daily_return
            },
            "risk_metrics": {
                "current_drawdown": self._current_drawdown,
                "peak_balance": self._peak_balance,
                "daily_trades": self._daily_trades,
                "daily_pnl": self._daily_pnl
            },
            "risk_status": {
                "circuit_breaker": self._circuit_breaker_triggered,
                "drawdown_warning": self._current_drawdown >= self.config.max_drawdown * 0.8,
                "daily_loss_warning": self._daily_pnl < 0 and abs(self._daily_pnl / account.balance) >= self.config.max_daily_loss * 0.8
            },
            "recent_events": self._risk_events[-10:] if self._risk_events else []
        }
        
        return report
    
    def _log_risk_event(self, event_type: str, details: Dict):
        """记录风险事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        self._risk_events.append(event)
        logger.warning(f"Risk event: {event_type} - {details}")
        
        # 保留最近100条记录
        if len(self._risk_events) > 100:
            self._risk_events = self._risk_events[-100:]
    
    def reset_daily(self):
        """重置日内统计"""
        self._daily_trades = 0
        self._daily_pnl = 0
        self._daily_turnover = 0
    
    def set_correlation_matrix(self, matrix: Dict[str, Dict[str, float]]):
        """设置品种相关性矩阵"""
        self._correlation_matrix = matrix


# ==================== VaR计算器 ====================

class VaRCalculator:
    """
    Value at Risk计算器
    支持历史模拟法、参数法、蒙特卡洛模拟
    """
    
    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        历史模拟法计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            VaR值（负数表示损失）
        """
        if len(returns) == 0:
            return 0
        
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        return var
    
    @staticmethod
    def parametric_var(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        参数法计算VaR（假设正态分布）
        """
        from scipy import stats
        
        if len(returns) == 0:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std
        
        return var
    
    @staticmethod
    def monte_carlo_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        time_horizon: int = 1
    ) -> float:
        """
        蒙特卡洛模拟法计算VaR
        """
        if len(returns) == 0:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # 生成模拟收益
        simulated_returns = np.random.normal(
            mean * time_horizon,
            std * np.sqrt(time_horizon),
            n_simulations
        )
        
        percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, percentile)
        
        return var
    
    @staticmethod
    def expected_shortfall(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        计算期望损失（CVaR/ES）
        超过VaR的平均损失
        """
        if len(returns) == 0:
            return 0
        
        var = VaRCalculator.historical_var(returns, confidence_level)
        
        # 计算超过VaR的平均损失
        tail_losses = returns[returns <= var]
        
        if len(tail_losses) == 0:
            return var
        
        es = np.mean(tail_losses)
        return es
