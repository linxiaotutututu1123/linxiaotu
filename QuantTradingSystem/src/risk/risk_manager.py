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
    """风控配置 - 优化目标: 年化50%+, 最大回撤<8%, 单笔亏损<2%"""
    # 单笔风险 - 严格控制在2%以内
    max_loss_per_trade: float = 0.015      # 单笔最大亏损比例(1.5%)
    max_order_value: float = 0.06          # 单笔最大委托金额比例(6%)
    
    # 日内风险 - 严格控制
    max_daily_loss: float = 0.02           # 日内最大亏损比例(2%)
    max_daily_trades: int = 20             # 日内最大交易次数(减少过度交易)
    max_daily_turnover: float = 2.0        # 日内最大换手率
    
    # 仓位风险 - 分散化
    max_position_per_symbol: float = 0.08  # 单品种最大仓位比例(8%)
    max_total_position: float = 0.50       # 总仓位上限(50%)
    max_correlated_exposure: float = 0.15  # 相关品种最大敞口(15%)
    
    # 杠杆风险
    max_leverage: float = 1.5              # 最大杠杆倍数(保守)
    
    # 流动性风险
    min_volume_threshold: int = 5000       # 最小成交量阈值
    max_volume_participation: float = 0.02 # 最大成交量占比(2%)
    
    # 回撤风险 - 严格控制在8%以内
    max_drawdown: float = 0.08             # 最大回撤阈值(8%)
    drawdown_reduce_ratio: float = 0.6     # 回撤达阈值时减仓比例
    drawdown_warning_level: float = 0.04   # 回撤预警水平(4%)
    
    # 波动风险
    volatility_scaling: bool = True        # 是否启用波动率缩放
    max_volatility_multiple: float = 1.3   # 最大波动率倍数
    target_volatility: float = 0.015       # 目标波动率(1.5%)
    
    # 熔断配置 - 更敏感的触发
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.03  # 触发熔断的亏损比例(3%)
    circuit_breaker_duration_hours: int = 6  # 熔断持续时间(小时)
    
    # 追踪止损配置
    trailing_stop_enabled: bool = True     # 是否启用追踪止损
    trailing_stop_atr_multiple: float = 1.8  # 追踪止损ATR倍数(更紧)
    trailing_stop_activation: float = 0.015  # 追踪止损激活收益(1.5%)
    
    # 盈利保护
    profit_protection_enabled: bool = True  # 是否启用盈利保护
    profit_protection_threshold: float = 0.02  # 盈利保护阈值(2%)
    profit_protection_ratio: float = 0.6    # 保护比例(保护60%盈利)


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
        计算Kelly最优仓位比例
        
        Kelly公式: f* = (p * b - q) / b
        其中: p=胜率, q=败率, b=盈亏比
        
        使用1/4 Kelly以降低风险(更保守)
        """
        stats = self._strategy_stats.get(strategy_name)
        if not stats:
            return 0.01  # 默认1%仓位(更保守)
        
        p = stats["win_rate"]
        q = 1 - p
        b = stats["profit_factor"]
        
        # 验证统计数据的有效性
        if b <= 0 or p <= 0 or p >= 1:
            return 0.01
        
        # 计算原始Kelly
        kelly = (p * b - q) / b
        
        # 如果Kelly为负,说明策略期望为负,不应该下注
        if kelly <= 0:
            return 0
        
        # 使用1/4 Kelly（更保守，降低波动）
        kelly = kelly * 0.25
        
        # 根据策略表现动态调整
        # 如果胜率高于60%且盈亏比大于1.5,可以适当提高
        if p > 0.6 and b > 1.5:
            kelly = kelly * 1.2
        # 如果胜率低于50%或盈亏比低于1.2,进一步降低
        elif p < 0.5 or b < 1.2:
            kelly = kelly * 0.7
        
        # 限制范围(最低0.5%,最高15%)
        kelly = max(0.005, min(kelly, 0.15))
        
        return kelly
    
    def calculate_volatility_adjusted_size(
        self,
        base_size: int,
        current_volatility: float,
        target_volatility: float = None
    ) -> int:
        """
        根据波动率调整仓位
        高波动时减仓，低波动时可适当加仓
        
        使用目标波动率缩放法
        """
        if not self.config.volatility_scaling:
            return base_size
        
        if current_volatility <= 0:
            return base_size
        
        target_vol = target_volatility or self.config.target_volatility
        
        # 波动率调整因子
        vol_factor = target_vol / current_volatility
        
        # 限制调整幅度 - 更保守的范围
        vol_factor = max(0.3, min(vol_factor, self.config.max_volatility_multiple))
        
        # 非线性调整：高波动时更激进地减仓
        if current_volatility > target_vol * 2:
            vol_factor = vol_factor * 0.7  # 额外减少30%
        
        adjusted_size = int(base_size * vol_factor)
        return max(1, adjusted_size)
    
    def calculate_drawdown_adjusted_size(
        self,
        base_size: int,
        current_drawdown: float
    ) -> int:
        """
        根据当前回撤调整仓位
        回撤越大，仓位越小
        """
        if current_drawdown <= 0:
            return base_size
        
        # 回撤调整因子
        # 回撤在预警水平以下：正常
        # 回撤在预警水平到最大回撤之间：线性减仓
        # 回撤超过最大回撤：大幅减仓
        
        warning_level = self.config.drawdown_warning_level
        max_dd = self.config.max_drawdown
        
        if current_drawdown <= warning_level:
            dd_factor = 1.0
        elif current_drawdown <= max_dd:
            # 线性插值减仓
            ratio = (current_drawdown - warning_level) / (max_dd - warning_level)
            dd_factor = 1.0 - ratio * self.config.drawdown_reduce_ratio
        else:
            # 超过最大回撤，强制大幅减仓
            dd_factor = 0.25
        
        adjusted_size = int(base_size * dd_factor)
        return max(1, adjusted_size)
    
    def calculate_optimal_size(
        self,
        account: AccountData,
        symbol: str,
        price: float,
        contract_size: float,
        margin_ratio: float,
        strategy_name: str = "",
        current_volatility: float = 0,
        current_drawdown: float = 0,
        atr: float = 0
    ) -> int:
        """
        计算最优仓位 - 增强版
        
        综合考虑：
        - Kelly公式
        - 风险预算(基于ATR)
        - 波动率调整
        - 回撤调整
        - 最大仓位限制
        """
        available = account.available
        
        if available <= 0 or price <= 0:
            return 1
        
        # 1. 基于ATR的风险预算计算（更精确）
        risk_budget = available * self.config.max_loss_per_trade
        
        if atr > 0:
            # 使用ATR计算每单位风险
            stop_distance = atr * 2.0  # 2倍ATR止损
            per_unit_risk = stop_distance * contract_size
        else:
            # 默认使用2%止损
            per_unit_risk = price * contract_size * 0.02
        
        if per_unit_risk > 0:
            risk_based_size = int(risk_budget / per_unit_risk)
        else:
            risk_based_size = 1
        
        # 2. 基于Kelly公式计算
        kelly_fraction = self.calculate_kelly_fraction(strategy_name)
        kelly_capital = available * kelly_fraction
        if price * contract_size * margin_ratio > 0:
            kelly_based_size = int(kelly_capital / (price * contract_size * margin_ratio))
        else:
            kelly_based_size = 1
        
        # 3. 取较小值作为基础仓位
        base_size = min(risk_based_size, kelly_based_size)
        base_size = max(1, base_size)
        
        # 4. 波动率调整
        if current_volatility > 0:
            base_size = self.calculate_volatility_adjusted_size(
                base_size, current_volatility
            )
        
        # 5. 回撤调整
        if current_drawdown > 0:
            base_size = self.calculate_drawdown_adjusted_size(
                base_size, current_drawdown
            )
        
        # 6. 最大仓位限制
        max_position_value = available * self.config.max_position_per_symbol
        if price * contract_size * margin_ratio > 0:
            max_size = int(max_position_value / (price * contract_size * margin_ratio))
        else:
            max_size = 1
        
        final_size = min(base_size, max_size)
        
        return max(1, final_size)


# ==================== 风险管理器 ====================

class RiskManager:
    """
    风险管理器 - 增强版
    实现多层次风险控制，包含：
    - 事前风控（订单检查）
    - 事中风控（实时监控）
    - 事后风控（复盘分析）
    - 多级熔断机制
    - VaR风险监控
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
        
        # 品种相关性矩阵
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # 多级熔断状态
        self._circuit_breaker_level: int = 0  # 0=正常, 1=警告, 2=限制, 3=熔断
        self._circuit_breaker_triggered: bool = False
        self._circuit_breaker_until: Optional[datetime] = None
        
        # 连续亏损追踪
        self._consecutive_losses: int = 0
        self._recent_trades_pnl: List[float] = []
        
        # 风险事件日志
        self._risk_events: List[Dict] = []
        
        # 收益率历史（用于VaR计算）
        self._returns_history: List[float] = []
        
        # 时间段熔断（避免特定高风险时段交易）
        self._blocked_periods: List[Tuple[int, int]] = [
            (14, 15),  # 收盘前一小时
            (9, 9),    # 开盘第一分钟
        ]
    
    def check_order(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        current_bar: Optional[BarData] = None
    ) -> RiskCheckResult:
        """
        事前风控 - 订单提交前检查（增强版）
        
        Args:
            order: 待提交订单
            account: 账户信息
            positions: 当前持仓
            current_bar: 当前K线数据
        
        Returns:
            RiskCheckResult
        """
        result = RiskCheckResult()
        
        # 0. 熔断检查（多级）
        if self._check_circuit_breaker_level(result):
            return result
        
        # 1. 时间段风控检查
        self._check_trading_time(result)
        if not result.passed:
            return result
        
        # 2. 资金检查
        self._check_capital(order, account, result)
        if not result.passed:
            return result
        
        # 3. 单笔风险检查
        self._check_single_trade_risk(order, account, result)
        if not result.passed:
            return result
        
        # 4. 仓位检查
        self._check_position_limit(order, account, positions, result)
        if not result.passed:
            return result
        
        # 5. 日内交易限制检查
        self._check_daily_limit(result)
        if not result.passed:
            return result
        
        # 6. 流动性检查
        if current_bar:
            self._check_liquidity(order, current_bar, result)
            if not result.passed:
                return result
        
        # 7. 杠杆检查
        self._check_leverage(order, account, positions, result)
        if not result.passed:
            return result
        
        # 8. 相关性风险检查
        self._check_correlation_risk(order, positions, result)
        
        # 9. VaR检查
        self._check_var_limit(order, account, positions, result)
        
        # 10. 连续亏损检查
        self._check_consecutive_losses(result)
        
        return result
    
    def _check_circuit_breaker_level(self, result: RiskCheckResult) -> bool:
        """多级熔断检查"""
        now = datetime.now()
        
        # 检查熔断是否恢复
        if self._circuit_breaker_until and now >= self._circuit_breaker_until:
            self._reset_circuit_breaker()
        
        if self._circuit_breaker_level >= 3:
            result.fail(
                RiskAction.REJECT,
                RiskLevel.CRITICAL,
                f"熔断中 (等级3)，恢复时间: {self._circuit_breaker_until}"
            )
            return True
        elif self._circuit_breaker_level == 2:
            result.add_message(f"限制交易模式 (等级2)：仅允许平仓")
            # 只允许平仓
            if hasattr(result, 'order') and result.order.offset == Offset.OPEN:
                result.fail(
                    RiskAction.REJECT,
                    RiskLevel.HIGH,
                    "限制交易模式：仅允许平仓操作"
                )
                return True
        elif self._circuit_breaker_level == 1:
            result.add_message("警告模式 (等级1)：仓位限制50%")
        
        return False
    
    def _check_trading_time(self, result: RiskCheckResult):
        """检查是否在高风险交易时段"""
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        for start_hour, end_hour in self._blocked_periods:
            if start_hour == end_hour:
                # 精确到分钟的检查
                if current_hour == start_hour and current_minute < 5:
                    result.add_message(f"注意: 当前为高风险时段 ({current_hour}:{current_minute:02d})")
            elif start_hour <= current_hour < end_hour:
                result.add_message(f"注意: 当前为高风险时段 ({start_hour}:00-{end_hour}:00)")
    
    def _check_consecutive_losses(self, result: RiskCheckResult):
        """检查连续亏损"""
        if self._consecutive_losses >= 5:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.HIGH,
                f"连续亏损 {self._consecutive_losses} 次，建议减少交易"
            )
            self._log_risk_event("CONSECUTIVE_LOSSES", {
                "count": self._consecutive_losses
            })
        elif self._consecutive_losses >= 3:
            result.add_message(f"警告: 连续亏损 {self._consecutive_losses} 次")
    
    def _check_var_limit(
        self,
        order: OrderData,
        account: AccountData,
        positions: Dict[str, PositionData],
        result: RiskCheckResult
    ):
        """VaR风险检查"""
        if len(self._returns_history) < 20:
            return  # 数据不足，跳过VaR检查
        
        returns_array = np.array(self._returns_history[-100:])
        
        # 计算当前VaR
        var_95 = VaRCalculator.historical_var(returns_array, 0.95)
        
        # 如果VaR表明风险较高，发出警告
        if var_95 < -0.03:  # 日VaR超过3%
            result.add_message(f"VaR警告: 95% VaR = {var_95:.2%}")
            
            # 如果VaR特别高且是开仓订单，建议减少仓位
            if var_95 < -0.05 and order.offset == Offset.OPEN:
                result.fail(
                    RiskAction.REDUCE,
                    RiskLevel.HIGH,
                    f"VaR风险过高: {var_95:.2%}，建议减少开仓"
                )
                result.suggested_volume = max(1, order.volume // 2)
    
    def update_trade_result(self, pnl: float, balance: float):
        """更新交易结果，用于连续亏损和熔断判断"""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # 记录收益率
        return_pct = pnl / balance if balance > 0 else 0
        self._recent_trades_pnl.append(pnl)
        self._returns_history.append(return_pct)
        
        # 保持最近的记录
        if len(self._recent_trades_pnl) > 50:
            self._recent_trades_pnl = self._recent_trades_pnl[-50:]
        if len(self._returns_history) > 500:
            self._returns_history = self._returns_history[-500:]
        
        # 检查是否需要升级熔断等级
        self._update_circuit_breaker_level()
    
    def _update_circuit_breaker_level(self):
        """根据当前状态更新熔断等级"""
        # 计算近期表现
        if len(self._recent_trades_pnl) < 5:
            return
        
        recent_pnl = sum(self._recent_trades_pnl[-10:])
        recent_win_rate = sum(1 for p in self._recent_trades_pnl[-10:] if p > 0) / min(10, len(self._recent_trades_pnl))
        
        # 根据多个指标判断熔断等级
        old_level = self._circuit_breaker_level
        
        if self._consecutive_losses >= 7 or (recent_pnl < 0 and abs(recent_pnl) > self._peak_balance * 0.05):
            self._circuit_breaker_level = 3
            self._circuit_breaker_triggered = True
            self._circuit_breaker_until = datetime.now() + timedelta(hours=self.config.circuit_breaker_duration_hours)
        elif self._consecutive_losses >= 5 or recent_win_rate < 0.2:
            self._circuit_breaker_level = 2
            self._circuit_breaker_until = datetime.now() + timedelta(hours=2)
        elif self._consecutive_losses >= 3 or recent_win_rate < 0.35:
            self._circuit_breaker_level = 1
        else:
            self._circuit_breaker_level = 0
        
        if self._circuit_breaker_level != old_level:
            self._log_risk_event("CIRCUIT_BREAKER_LEVEL_CHANGE", {
                "old_level": old_level,
                "new_level": self._circuit_breaker_level,
                "consecutive_losses": self._consecutive_losses,
                "recent_win_rate": recent_win_rate
            })
    
    def _reset_circuit_breaker(self):
        """重置熔断状态"""
        self._circuit_breaker_level = 0
        self._circuit_breaker_triggered = False
        self._circuit_breaker_until = None
        self._consecutive_losses = 0
        self._log_risk_event("CIRCUIT_BREAKER_RESET", {})
    
    def get_risk_status(self) -> Dict[str, Any]:
        """获取当前风险状态摘要"""
        return {
            "circuit_breaker_level": self._circuit_breaker_level,
            "circuit_breaker_triggered": self._circuit_breaker_triggered,
            "circuit_breaker_until": self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None,
            "consecutive_losses": self._consecutive_losses,
            "current_drawdown": self._current_drawdown,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "peak_balance": self._peak_balance
        }
    
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
    
    def update_daily_stats(self, trade_pnl: float, turnover: float, current_date: datetime):
        """更新日内统计"""
        # 日期变化时重置
        if self._daily_date is None or current_date.date() != self._daily_date.date():
            self._daily_trades = 0
            self._daily_pnl = 0
            self._daily_turnover = 0
            self._daily_date = current_date
        
        self._daily_trades += 1
        self._daily_pnl += trade_pnl
        self._daily_turnover += turnover
    
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
    
    def check_signal(self, signal: Any) -> RiskCheckResult:
        """
        检查交易信号是否符合风控要求
        
        Args:
            signal: SignalData对象
        
        Returns:
            RiskCheckResult
        """
        result = RiskCheckResult()
        
        # 检查熔断状态
        if self._check_circuit_breaker_level(result):
            return result
        
        # 检查信号强度
        if hasattr(signal, 'strength'):
            if abs(signal.strength) < 0.3:
                result.add_message(f"信号强度较弱: {signal.strength:.2f}")
            if abs(signal.strength) < 0.2:
                result.fail(
                    RiskAction.REJECT,
                    RiskLevel.LOW,
                    f"信号强度不足: {signal.strength:.2f}"
                )
                return result
        
        # 检查连续亏损
        self._check_consecutive_losses(result)
        if not result.passed:
            return result
        
        # 检查回撤
        if self._current_drawdown >= self.config.max_drawdown:
            result.fail(
                RiskAction.REDUCE,
                RiskLevel.HIGH,
                f"当前回撤过高: {self._current_drawdown:.2%}"
            )
        elif self._current_drawdown >= self.config.drawdown_warning_level:
            result.add_message(f"回撤警告: {self._current_drawdown:.2%}")
        
        return result
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        signal_strength: float = 1.0,
        account_balance: float = None,
        contract_size: float = 10,
        margin_ratio: float = 0.1
    ) -> int:
        """
        计算仓位大小
        
        基于风险预算和信号强度计算最优仓位
        
        Args:
            symbol: 合约代码
            price: 当前价格
            stop_loss: 止损价格
            signal_strength: 信号强度 (0-1)
            account_balance: 账户余额
            contract_size: 合约乘数
            margin_ratio: 保证金比例
        
        Returns:
            建议手数
        """
        if account_balance is None:
            account_balance = self._peak_balance or 1000000
        
        if account_balance <= 0 or price <= 0:
            return 1
        
        # 1. 基于风险预算计算
        risk_budget = account_balance * self.config.max_loss_per_trade
        
        # 计算每手风险
        risk_per_lot = abs(price - stop_loss) * contract_size
        if risk_per_lot <= 0:
            risk_per_lot = price * 0.02 * contract_size  # 默认2%止损
        
        risk_based_size = int(risk_budget / risk_per_lot)
        
        # 2. 基于信号强度调整
        signal_factor = 0.5 + 0.5 * abs(signal_strength)  # 0.5 ~ 1.0
        adjusted_size = int(risk_based_size * signal_factor)
        
        # 3. 基于回撤调整
        if self._current_drawdown > 0:
            dd_factor = self.position_sizer.calculate_drawdown_adjusted_size(
                100, self._current_drawdown
            ) / 100.0
            adjusted_size = int(adjusted_size * dd_factor)
        
        # 4. 基于熔断等级调整
        if self._circuit_breaker_level == 1:
            adjusted_size = int(adjusted_size * 0.5)
        elif self._circuit_breaker_level >= 2:
            adjusted_size = 0
        
        # 5. 单品种仓位限制
        max_position_value = account_balance * self.config.max_position_per_symbol
        max_lots = int(max_position_value / (price * contract_size * margin_ratio))
        
        final_size = min(adjusted_size, max_lots)
        
        return max(1, final_size) if final_size > 0 else 0


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
