"""
自动交易管理器
整合策略、风控和执行的自动化交易
"""

import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
import logging
import time

from ..data.data_structures import (
    BarData, SignalData, OrderData, PositionData,
    Direction, OrderType
)
from ..strategy.strategy_base import BaseStrategy
from ..risk.risk_manager import RiskManager, RiskCheckResult

logger = logging.getLogger(__name__)


# ==================== 交易时段 ====================

class TradingSession(Enum):
    """交易时段"""
    PRE_MARKET = "pre_market"       # 盘前
    MORNING_1 = "morning_1"          # 上午第一节 9:00-10:15
    MORNING_2 = "morning_2"          # 上午第二节 10:30-11:30
    AFTERNOON = "afternoon"          # 下午 13:30-15:00
    NIGHT = "night"                  # 夜盘 21:00-次日
    CLOSED = "closed"                # 休市


# 交易时段配置（期货）
TRADING_HOURS = {
    TradingSession.MORNING_1: (dtime(9, 0), dtime(10, 15)),
    TradingSession.MORNING_2: (dtime(10, 30), dtime(11, 30)),
    TradingSession.AFTERNOON: (dtime(13, 30), dtime(15, 0)),
    TradingSession.NIGHT: (dtime(21, 0), dtime(23, 0)),  # 简化，实际根据品种不同
}


@dataclass
class AutoTraderConfig:
    """自动交易配置"""
    # 交易配置
    enabled: bool = True
    paper_trading: bool = True           # 模拟交易
    
    # 执行配置
    max_daily_trades: int = 100          # 每日最大交易次数
    min_signal_strength: float = 0.5     # 最小信号强度
    
    # 仓位配置
    default_volume: int = 1              # 默认手数
    use_kelly_sizing: bool = True        # 使用凯利公式
    
    # 订单配置
    order_type: str = "limit"            # limit/market
    chase_price: bool = True             # 追价
    max_chase_ticks: int = 2             # 最大追价跳数
    
    # 时间配置
    close_before_end: int = 300          # 收盘前平仓(秒)
    
    # 品种配置
    symbols: List[str] = field(default_factory=list)


# ==================== 信号处理器 ====================

class SignalProcessor:
    """
    信号处理器
    处理策略信号，生成交易指令
    """
    
    def __init__(self, config: AutoTraderConfig):
        self.config = config
        self._signal_history: List[SignalData] = []
    
    def process_signal(
        self, 
        signal: SignalData,
        current_position: float = 0
    ) -> Optional[Dict]:
        """
        处理信号
        
        Returns:
            交易指令 {'action': 'buy'/'sell'/'close', 'volume': float} 或 None
        """
        # 过滤弱信号
        if abs(signal.strength) < self.config.min_signal_strength:
            return None
        
        # 记录信号
        self._signal_history.append(signal)
        
        action = None
        volume = self.config.default_volume
        
        # 根据信号方向和当前仓位决定动作
        if signal.direction == 1:  # 做多信号
            if current_position <= 0:
                action = 'buy'
                if current_position < 0:
                    volume += abs(current_position)  # 先平空再开多
                    
        elif signal.direction == -1:  # 做空信号
            if current_position >= 0:
                action = 'sell'
                if current_position > 0:
                    volume += current_position  # 先平多再开空
                    
        elif signal.direction == 0:  # 平仓信号
            if current_position != 0:
                action = 'close'
                volume = abs(current_position)
        
        if action:
            return {
                'action': action,
                'volume': volume,
                'signal': signal,
                'timestamp': datetime.now()
            }
        
        return None
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计"""
        if not self._signal_history:
            return {}
        
        buy_signals = sum(1 for s in self._signal_history if s.direction == 1)
        sell_signals = sum(1 for s in self._signal_history if s.direction == -1)
        
        return {
            'total_signals': len(self._signal_history),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_strength': sum(abs(s.strength) for s in self._signal_history) / len(self._signal_history)
        }


# ==================== 自动交易器 ====================

class AutoTrader:
    """
    自动交易器
    自动执行交易策略
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        risk_manager: RiskManager,
        execution_engine: Any,  # ExecutionEngine
        config: AutoTraderConfig = None
    ):
        self.strategies = {s.name: s for s in strategies}
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.config = config or AutoTraderConfig()
        
        self.signal_processor = SignalProcessor(config)
        
        # 状态
        self._running = False
        self._paused = False
        self._daily_trades = 0
        self._last_trade_date: Optional[datetime] = None
        
        # 持仓跟踪
        self._positions: Dict[str, float] = {}
        
        # 回调
        self._trade_callbacks: List[Callable] = []
    
    async def start(self):
        """启动自动交易"""
        if not self.config.enabled:
            logger.warning("Auto trader is disabled")
            return
        
        logger.info("Starting auto trader...")
        self._running = True
        
        # 重置每日计数
        self._reset_daily_counters()
        
        # 启动监控循环
        asyncio.create_task(self._monitor_loop())
        
        logger.info("Auto trader started")
    
    async def stop(self):
        """停止自动交易"""
        logger.info("Stopping auto trader...")
        self._running = False
        
        # 平掉所有仓位
        if not self.config.paper_trading:
            await self._close_all_positions()
        
        logger.info("Auto trader stopped")
    
    def pause(self):
        """暂停交易"""
        self._paused = True
        logger.info("Auto trader paused")
    
    def resume(self):
        """恢复交易"""
        self._paused = False
        logger.info("Auto trader resumed")
    
    async def on_bar(self, bar: BarData):
        """
        处理K线数据
        """
        if not self._running or self._paused:
            return
        
        # 检查交易时段
        if not self._is_trading_time():
            return
        
        # 检查每日交易限制
        if self._daily_trades >= self.config.max_daily_trades:
            return
        
        symbol = bar.symbol
        
        # 获取所有策略的信号
        signals = []
        for name, strategy in self.strategies.items():
            if symbol in strategy.symbols or not strategy.symbols:
                signal = strategy.on_bar(bar)
                if signal:
                    signals.append(signal)
        
        # 处理信号
        for signal in signals:
            await self._process_signal(signal)
    
    async def _process_signal(self, signal: SignalData):
        """处理单个信号"""
        symbol = signal.symbol
        current_position = self._positions.get(symbol, 0)
        
        # 信号处理
        trade_cmd = self.signal_processor.process_signal(signal, current_position)
        if not trade_cmd:
            return
        
        # 风控检查
        risk_result = self.risk_manager.check_signal(signal)
        if not risk_result.passed:
            logger.warning(f"Signal rejected by risk manager: {risk_result.message}")
            return
        
        # 计算仓位
        if self.config.use_kelly_sizing:
            volume = self.risk_manager.calculate_position_size(
                symbol=symbol,
                price=signal.price,
                stop_loss=signal.stop_loss or signal.price * 0.98,
                signal_strength=signal.strength
            )
        else:
            volume = trade_cmd['volume']
        
        # 执行交易
        await self._execute_trade(
            symbol=symbol,
            action=trade_cmd['action'],
            volume=volume,
            price=signal.price,
            signal=signal
        )
    
    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        volume: float,
        price: float,
        signal: SignalData
    ):
        """执行交易"""
        # 确定方向
        if action == 'buy':
            direction = Direction.LONG
        elif action == 'sell':
            direction = Direction.SHORT
        elif action == 'close':
            current_pos = self._positions.get(symbol, 0)
            direction = Direction.SHORT if current_pos > 0 else Direction.LONG
        else:
            return
        
        # 订单类型
        order_type = OrderType.LIMIT if self.config.order_type == 'limit' else OrderType.MARKET
        
        if self.config.paper_trading:
            # 模拟交易
            order_id = f"PAPER_{int(time.time()*1000)}"
            logger.info(f"[PAPER] {action} {volume} {symbol} @ {price}")
            
            # 更新虚拟仓位
            if action == 'buy':
                self._positions[symbol] = self._positions.get(symbol, 0) + volume
            elif action == 'sell':
                self._positions[symbol] = self._positions.get(symbol, 0) - volume
            elif action == 'close':
                self._positions[symbol] = 0
        else:
            # 实盘交易
            try:
                order_id = await self.execution_engine.submit_order(
                    symbol=symbol,
                    direction=direction,
                    volume=volume,
                    price=price,
                    order_type=order_type
                )
                logger.info(f"[LIVE] {action} {volume} {symbol} @ {price}, order_id={order_id}")
                
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
                return
        
        # 更新统计
        self._daily_trades += 1
        
        # 回调通知
        for callback in self._trade_callbacks:
            try:
                callback({
                    'symbol': symbol,
                    'action': action,
                    'volume': volume,
                    'price': price,
                    'signal': signal,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def _is_trading_time(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now().time()
        
        for session, (start, end) in TRADING_HOURS.items():
            if start <= now <= end:
                return True
        
        return False
    
    def _get_current_session(self) -> TradingSession:
        """获取当前交易时段"""
        now = datetime.now().time()
        
        for session, (start, end) in TRADING_HOURS.items():
            if start <= now <= end:
                return session
        
        return TradingSession.CLOSED
    
    def _reset_daily_counters(self):
        """重置每日计数器"""
        today = datetime.now().date()
        if self._last_trade_date != today:
            self._daily_trades = 0
            self._last_trade_date = today
    
    async def _close_all_positions(self):
        """平掉所有仓位"""
        for symbol, position in self._positions.items():
            if position != 0:
                await self._execute_trade(
                    symbol=symbol,
                    action='close',
                    volume=abs(position),
                    price=0,  # 市价
                    signal=SignalData(
                        symbol=symbol,
                        direction=0,
                        strength=1.0,
                        price=0,
                        timestamp=datetime.now()
                    )
                )
    
    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 重置每日计数
                self._reset_daily_counters()
                
                # 检查收盘前平仓
                session = self._get_current_session()
                if session == TradingSession.AFTERNOON:
                    now = datetime.now()
                    close_time = datetime.combine(now.date(), dtime(15, 0))
                    seconds_to_close = (close_time - now).total_seconds()
                    
                    if seconds_to_close <= self.config.close_before_end:
                        logger.info("Closing positions before market close")
                        await self._close_all_positions()
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            await asyncio.sleep(60)
    
    def register_trade_callback(self, callback: Callable):
        """注册交易回调"""
        self._trade_callbacks.append(callback)
    
    def get_positions(self) -> Dict[str, float]:
        """获取当前仓位"""
        return self._positions.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """获取交易器状态"""
        return {
            'running': self._running,
            'paused': self._paused,
            'paper_trading': self.config.paper_trading,
            'session': self._get_current_session().value,
            'daily_trades': self._daily_trades,
            'positions': self._positions.copy(),
            'strategies': list(self.strategies.keys())
        }


# ==================== 交易调度器 ====================

class TradeScheduler:
    """
    交易调度器
    管理交易任务的定时执行
    """
    
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._running = False
    
    def add_task(
        self,
        name: str,
        callback: Callable,
        schedule_time: dtime = None,
        interval_seconds: int = None,
        enabled: bool = True
    ):
        """添加定时任务"""
        self._tasks[name] = {
            'callback': callback,
            'schedule_time': schedule_time,
            'interval': interval_seconds,
            'enabled': enabled,
            'last_run': None
        }
    
    def remove_task(self, name: str):
        """移除任务"""
        if name in self._tasks:
            del self._tasks[name]
    
    def enable_task(self, name: str):
        """启用任务"""
        if name in self._tasks:
            self._tasks[name]['enabled'] = True
    
    def disable_task(self, name: str):
        """禁用任务"""
        if name in self._tasks:
            self._tasks[name]['enabled'] = False
    
    async def start(self):
        """启动调度器"""
        self._running = True
        asyncio.create_task(self._scheduler_loop())
        logger.info("Trade scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        logger.info("Trade scheduler stopped")
    
    async def _scheduler_loop(self):
        """调度循环"""
        while self._running:
            now = datetime.now()
            
            for name, task in self._tasks.items():
                if not task['enabled']:
                    continue
                
                should_run = False
                
                # 定时任务
                if task['schedule_time']:
                    if now.time().hour == task['schedule_time'].hour and \
                       now.time().minute == task['schedule_time'].minute:
                        if task['last_run'] is None or \
                           task['last_run'].date() != now.date():
                            should_run = True
                
                # 间隔任务
                elif task['interval']:
                    if task['last_run'] is None or \
                       (now - task['last_run']).total_seconds() >= task['interval']:
                        should_run = True
                
                if should_run:
                    try:
                        if asyncio.iscoroutinefunction(task['callback']):
                            await task['callback']()
                        else:
                            task['callback']()
                        task['last_run'] = now
                        logger.info(f"Task '{name}' executed")
                    except Exception as e:
                        logger.error(f"Task '{name}' failed: {e}")
            
            await asyncio.sleep(1)
