"""
回测引擎 - 超高速事件驱动回测系统
支持Tick级/Bar级回测，多策略并行，精确滑点模拟
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import uuid
import time
from abc import ABC, abstractmethod

from ..data.data_structures import (
    BarData, TickData, OrderData, TradeData, PositionData, AccountData,
    Direction, Offset, OrderType, OrderStatus, PerformanceMetrics,
    SignalData, DataBuffer
)

logger = logging.getLogger(__name__)


# ==================== 回测配置 ====================

@dataclass
class BacktestConfig:
    """回测配置"""
    # 基本配置
    initial_capital: float = 1000000.0    # 初始资金
    start_date: datetime = None
    end_date: datetime = None
    
    # 手续费配置 - 更真实的费率
    commission_rate: float = 0.00025      # 手续费率（万分之2.5）
    commission_fixed: float = 0.0         # 固定手续费（每手）
    commission_min: float = 0.0           # 最低手续费
    
    # 滑点配置 - 更真实的滑点
    slippage_mode: str = "adaptive"       # fixed / adaptive / model
    slippage_rate: float = 0.0002         # 固定滑点率
    slippage_ticks: int = 2               # 固定滑点跳数(更保守)
    slippage_volatility_factor: float = 0.5  # 波动率滑点因子
    
    # 撮合配置
    match_mode: str = "next_bar"          # next_bar / current_bar / tick
    volume_limit: float = 0.05            # 成交量限制（更保守5%）
    partial_fill_enabled: bool = True     # 启用部分成交
    
    # 保证金配置
    margin_ratio: float = 0.1             # 默认保证金比例
    leverage: float = 1.0                 # 杠杆倍数
    
    # 合约配置
    contract_size: float = 10.0           # 合约乘数
    price_tick: float = 1.0               # 最小价格变动
    
    # 市场冲击配置
    market_impact_enabled: bool = True    # 启用市场冲击模型
    market_impact_factor: float = 0.1     # 市场冲击因子


# ==================== 订单撮合引擎 ====================

class MatchEngine:
    """
    订单撮合引擎
    模拟真实交易所的订单撮合逻辑
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._pending_orders: List[OrderData] = []
        self._slippage_model = None
    
    def submit_order(self, order: OrderData):
        """提交订单"""
        order.status = OrderStatus.SUBMITTED
        order.create_time = datetime.now()
        self._pending_orders.append(order)
        logger.debug(f"Order submitted: {order.order_id}")
    
    def match_orders(self, bar: BarData) -> List[TradeData]:
        """
        撮合订单（Bar级别）
        
        Args:
            bar: 当前K线数据
        
        Returns:
            成交列表
        """
        trades = []
        remaining_orders = []
        
        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue
            
            trade = self._try_match(order, bar)
            if trade:
                trades.append(trade)
                if order.remaining > 0:
                    remaining_orders.append(order)
            else:
                remaining_orders.append(order)
        
        self._pending_orders = remaining_orders
        return trades
    
    def _try_match(self, order: OrderData, bar: BarData) -> Optional[TradeData]:
        """
        尝试撮合单个订单
        """
        # 检查是否可以成交
        can_fill, fill_price = self._check_fill(order, bar)
        
        if not can_fill:
            return None
        
        # 计算滑点
        slippage = self._calculate_slippage(order, bar)
        
        # 调整成交价格
        if order.direction == Direction.LONG:
            fill_price += slippage
            fill_price = min(fill_price, bar.high)  # 不能超过最高价
        else:
            fill_price -= slippage
            fill_price = max(fill_price, bar.low)   # 不能低于最低价
        
        # 计算成交数量（考虑成交量限制）
        max_volume = int(bar.volume * self.config.volume_limit)
        fill_volume = min(order.remaining, max_volume) if max_volume > 0 else order.remaining
        
        if fill_volume <= 0:
            return None
        
        # 更新订单状态
        order.traded += fill_volume
        order.remaining = order.volume - order.traded
        order.avg_price = (
            (order.avg_price * (order.traded - fill_volume) + fill_price * fill_volume) 
            / order.traded if order.traded > 0 else fill_price
        )
        
        if order.remaining == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILLED
        
        order.update_time = bar.datetime
        
        # 计算手续费
        commission = self._calculate_commission(fill_price, fill_volume)
        
        # 创建成交记录
        trade = TradeData(
            trade_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            exchange=order.exchange,
            direction=order.direction,
            offset=order.offset,
            price=fill_price,
            volume=fill_volume,
            datetime=bar.datetime,
            commission=commission,
            strategy_name=order.strategy_name
        )
        
        logger.debug(f"Trade executed: {trade.trade_id}, price={fill_price}, volume={fill_volume}")
        return trade
    
    def _check_fill(self, order: OrderData, bar: BarData) -> Tuple[bool, float]:
        """
        检查订单是否可以成交
        
        Returns:
            (是否可以成交, 成交价格)
        """
        if order.order_type == OrderType.MARKET:
            # 市价单：使用开盘价成交
            return True, bar.open
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单：检查价格是否在范围内
            if order.direction == Direction.LONG:
                # 买入：委托价 >= 最低价
                if order.price >= bar.low:
                    fill_price = min(order.price, bar.open)  # 使用较低价格成交
                    return True, fill_price
            else:
                # 卖出：委托价 <= 最高价
                if order.price <= bar.high:
                    fill_price = max(order.price, bar.open)  # 使用较高价格成交
                    return True, fill_price
        
        elif order.order_type == OrderType.STOP:
            # 止损单
            if order.direction == Direction.LONG:
                if bar.high >= order.price:
                    return True, max(order.price, bar.open)
            else:
                if bar.low <= order.price:
                    return True, min(order.price, bar.open)
        
        return False, 0.0
    
    def _calculate_slippage(self, order: OrderData, bar: BarData) -> float:
        """
        计算滑点 - 增强版
        支持固定滑点、自适应滑点、模型滑点
        
        更真实的滑点模拟考虑:
        1. 市场波动率
        2. 订单大小相对于成交量
        3. 市场冲击成本
        4. 价格跳动最小单位
        """
        base_slippage = 0.0
        
        if self.config.slippage_mode == "fixed":
            # 固定滑点
            base_slippage = self.config.price_tick * self.config.slippage_ticks
        
        elif self.config.slippage_mode == "adaptive":
            # 自适应滑点（根据波动率和成交量）
            # 1. 波动率成分
            volatility = bar.range / bar.close if bar.close > 0 else 0
            volatility_slippage = bar.close * volatility * self.config.slippage_volatility_factor
            
            # 2. 成交量成分 - 订单越大相对于成交量，滑点越大
            volume_ratio = order.volume / bar.volume if bar.volume > 0 else 1
            volume_slippage = bar.close * volume_ratio * 0.001
            
            # 3. 市场冲击成本
            market_impact = 0.0
            if self.config.market_impact_enabled:
                # 平方根市场冲击模型
                import math
                participation_rate = min(volume_ratio, 1.0)
                market_impact = bar.close * self.config.market_impact_factor * math.sqrt(participation_rate)
            
            base_slippage = volatility_slippage + volume_slippage + market_impact
            
            # 确保滑点至少为1个tick
            base_slippage = max(base_slippage, self.config.price_tick)
        
        elif self.config.slippage_mode == "model":
            # 模型滑点（预留接口）
            if self._slippage_model:
                base_slippage = self._slippage_model.predict(order, bar)
            else:
                base_slippage = self.config.price_tick * self.config.slippage_ticks
        
        else:
            base_slippage = self.config.price_tick
        
        # 滑点取整到最小价格变动单位
        ticks = int(base_slippage / self.config.price_tick + 0.5)
        slippage = ticks * self.config.price_tick
        
        # 设置最大滑点限制（不超过K线振幅的30%）
        max_slippage = bar.range * 0.3
        slippage = min(slippage, max_slippage)
        
        return slippage
    
    def _calculate_commission(self, price: float, volume: int) -> float:
        """计算手续费"""
        # 按比例计算
        commission = price * volume * self.config.contract_size * self.config.commission_rate
        
        # 加上固定费用
        commission += self.config.commission_fixed * volume
        
        # 不低于最低手续费
        commission = max(commission, self.config.commission_min)
        
        return commission
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        for order in self._pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self._pending_orders.remove(order)
                logger.debug(f"Order cancelled: {order_id}")
                return True
        return False
    
    def get_pending_orders(self, symbol: str = None) -> List[OrderData]:
        """获取未成交订单"""
        if symbol:
            return [o for o in self._pending_orders if o.symbol == symbol]
        return self._pending_orders.copy()


# ==================== 账户管理器 ====================

class AccountManager:
    """
    账户管理器
    管理资金、持仓、盈亏计算
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._account = AccountData(
            account_id="backtest",
            balance=config.initial_capital,
            available=config.initial_capital,
            pre_balance=config.initial_capital
        )
        self._positions: Dict[str, Dict[Direction, PositionData]] = defaultdict(dict)
        self._trades: List[TradeData] = []
        
        # 每日权益记录
        self._daily_balance: List[Tuple[datetime, float]] = []
    
    def process_trade(self, trade: TradeData, current_price: float):
        """
        处理成交
        
        Args:
            trade: 成交数据
            current_price: 当前价格（用于计算盈亏）
        """
        self._trades.append(trade)
        
        symbol = trade.symbol
        direction = trade.direction
        
        # 获取或创建持仓
        if direction not in self._positions[symbol]:
            self._positions[symbol][direction] = PositionData(
                symbol=symbol,
                exchange=trade.exchange,
                direction=direction
            )
        
        position = self._positions[symbol][direction]
        
        if trade.offset == Offset.OPEN:
            # 开仓
            total_cost = position.price * position.volume + trade.price * trade.volume
            position.volume += trade.volume
            position.td_volume += trade.volume
            position.price = total_cost / position.volume if position.volume > 0 else 0
            position.available = position.volume - position.frozen
            
            # 冻结保证金
            margin = trade.price * trade.volume * self.config.contract_size * self.config.margin_ratio
            self._account.margin += margin
            self._account.available -= margin
        
        else:
            # 平仓
            if position.volume >= trade.volume:
                # 计算平仓盈亏
                if direction == Direction.LONG:
                    pnl = (trade.price - position.price) * trade.volume * self.config.contract_size
                else:
                    pnl = (position.price - trade.price) * trade.volume * self.config.contract_size
                
                position.realized_pnl += pnl
                self._account.close_profit += pnl
                
                # 释放保证金
                margin = position.price * trade.volume * self.config.contract_size * self.config.margin_ratio
                self._account.margin -= margin
                self._account.available += margin + pnl
                
                # 更新持仓
                position.volume -= trade.volume
                if trade.offset == Offset.CLOSE_TODAY:
                    position.td_volume -= trade.volume
                else:
                    position.yd_volume -= trade.volume
                position.available = position.volume - position.frozen
        
        # 扣除手续费
        self._account.commission += trade.commission
        self._account.available -= trade.commission
        self._account.balance -= trade.commission
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """
        更新持仓浮动盈亏
        
        Args:
            symbol: 合约代码
            current_price: 当前价格
        """
        if symbol not in self._positions:
            return
        
        for direction, position in self._positions[symbol].items():
            if position.volume == 0:
                continue
            
            if direction == Direction.LONG:
                position.unrealized_pnl = (
                    (current_price - position.price) * 
                    position.volume * 
                    self.config.contract_size
                )
            else:
                position.unrealized_pnl = (
                    (position.price - current_price) * 
                    position.volume * 
                    self.config.contract_size
                )
            
            position.pnl = position.realized_pnl + position.unrealized_pnl
    
    def update_account(self):
        """更新账户总权益"""
        # 计算持仓盈亏
        position_profit = sum(
            pos.unrealized_pnl
            for positions in self._positions.values()
            for pos in positions.values()
        )
        
        self._account.position_profit = position_profit
        self._account.balance = (
            self._account.pre_balance + 
            self._account.close_profit + 
            self._account.position_profit - 
            self._account.commission
        )
    
    def record_daily_balance(self, date: datetime):
        """记录每日权益"""
        self._daily_balance.append((date, self._account.balance))
    
    def settle_day(self, date: datetime):
        """日结算"""
        # 更新账户
        self.update_account()
        
        # 记录权益
        self.record_daily_balance(date)
        
        # 今仓转昨仓
        for positions in self._positions.values():
            for position in positions.values():
                position.yd_volume = position.volume
                position.td_volume = 0
        
        # 更新昨日权益
        self._account.pre_balance = self._account.balance
        self._account.close_profit = 0
        self._account.commission = 0
    
    def get_position(self, symbol: str, direction: Direction = None) -> Optional[PositionData]:
        """获取持仓"""
        if symbol not in self._positions:
            return None
        
        if direction:
            return self._positions[symbol].get(direction)
        
        # 返回净持仓
        long_pos = self._positions[symbol].get(Direction.LONG)
        short_pos = self._positions[symbol].get(Direction.SHORT)
        
        long_vol = long_pos.volume if long_pos else 0
        short_vol = short_pos.volume if short_pos else 0
        
        if long_vol > short_vol:
            return long_pos
        elif short_vol > long_vol:
            return short_pos
        return None
    
    def get_all_positions(self) -> Dict[str, Dict[Direction, PositionData]]:
        """获取所有持仓"""
        return dict(self._positions)
    
    def get_account(self) -> AccountData:
        """获取账户信息"""
        return self._account
    
    def get_daily_balance(self) -> pd.DataFrame:
        """获取每日权益DataFrame"""
        if not self._daily_balance:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._daily_balance, columns=['datetime', 'balance'])
        df['returns'] = df['balance'].pct_change()
        return df
    
    def get_trades(self) -> List[TradeData]:
        """获取所有成交记录"""
        return self._trades.copy()


# ==================== 回测引擎核心 ====================

class BacktestEngine:
    """
    回测引擎核心
    事件驱动，支持多策略并行回测
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self._match_engine = MatchEngine(self.config)
        self._account_manager = AccountManager(self.config)
        
        self._strategies: Dict[str, Any] = {}  # 策略实例
        self._data: Dict[str, pd.DataFrame] = {}  # 回测数据
        self._current_bars: Dict[str, BarData] = {}  # 当前K线
        
        self._is_running = False
        self._current_datetime: Optional[datetime] = None
        
        # 事件回调
        self._callbacks: Dict[str, List[Callable]] = {
            "on_bar": [],
            "on_trade": [],
            "on_order": [],
            "on_finish": []
        }
        
        # 性能统计
        self._start_time: Optional[float] = None
        self._bar_count = 0
    
    def add_strategy(self, strategy, name: str = None):
        """
        添加策略
        
        Args:
            strategy: 策略实例（需要实现on_bar方法）
            name: 策略名称
        """
        name = name or strategy.__class__.__name__
        self._strategies[name] = strategy
        strategy.engine = self
        logger.info(f"Strategy added: {name}")
    
    def add_data(self, symbol: str, data: pd.DataFrame):
        """
        添加回测数据
        
        Args:
            symbol: 合约代码
            data: DataFrame with columns: datetime, open, high, low, close, volume
        """
        # 确保数据格式正确
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 排序
        data = data.sort_values('datetime').reset_index(drop=True)
        
        self._data[symbol] = data
        logger.info(f"Data added: {symbol}, {len(data)} bars")
    
    def run(self) -> PerformanceMetrics:
        """
        运行回测
        
        Returns:
            绩效指标
        """
        if not self._strategies:
            raise ValueError("No strategy added")
        
        if not self._data:
            raise ValueError("No data added")
        
        logger.info("="*50)
        logger.info("Starting backtest...")
        logger.info(f"Initial capital: {self.config.initial_capital:,.2f}")
        logger.info(f"Strategies: {list(self._strategies.keys())}")
        logger.info(f"Symbols: {list(self._data.keys())}")
        logger.info("="*50)
        
        self._is_running = True
        self._start_time = time.time()
        self._bar_count = 0
        
        # 初始化策略
        for name, strategy in self._strategies.items():
            if hasattr(strategy, 'on_init'):
                strategy.on_init()
        
        # 合并所有品种的时间序列
        all_datetimes = set()
        for df in self._data.values():
            all_datetimes.update(df['datetime'].tolist())
        
        sorted_datetimes = sorted(all_datetimes)
        
        # 时间过滤
        if self.config.start_date:
            sorted_datetimes = [dt for dt in sorted_datetimes if dt >= self.config.start_date]
        if self.config.end_date:
            sorted_datetimes = [dt for dt in sorted_datetimes if dt <= self.config.end_date]
        
        logger.info(f"Backtest period: {sorted_datetimes[0]} to {sorted_datetimes[-1]}")
        logger.info(f"Total bars: {len(sorted_datetimes)}")
        
        # 按时间推进
        current_date = None
        for dt in sorted_datetimes:
            self._current_datetime = dt
            
            # 日期变化时进行日结算
            if current_date and dt.date() != current_date:
                self._account_manager.settle_day(datetime.combine(current_date, datetime.min.time()))
            current_date = dt.date()
            
            # 更新各品种当前K线
            bars = {}
            for symbol, df in self._data.items():
                bar_data = df[df['datetime'] == dt]
                if not bar_data.empty:
                    row = bar_data.iloc[0]
                    bar = BarData(
                        symbol=symbol,
                        exchange="",
                        datetime=dt,
                        interval="1m",
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=int(row.get('volume', 0)),
                        turnover=row.get('turnover', 0),
                        open_interest=int(row.get('open_interest', 0))
                    )
                    bars[symbol] = bar
                    self._current_bars[symbol] = bar
            
            if not bars:
                continue
            
            # 撮合订单
            for symbol, bar in bars.items():
                trades = self._match_engine.match_orders(bar)
                for trade in trades:
                    self._account_manager.process_trade(trade, bar.close)
                    self._emit("on_trade", trade)
            
            # 更新持仓盈亏
            for symbol, bar in bars.items():
                self._account_manager.update_position_pnl(symbol, bar.close)
            
            # 更新账户
            self._account_manager.update_account()
            
            # 调用策略
            for name, strategy in self._strategies.items():
                try:
                    if hasattr(strategy, 'on_bar'):
                        strategy.on_bar(bars)
                except Exception as e:
                    logger.error(f"Strategy {name} error: {e}")
            
            # 触发回调
            self._emit("on_bar", bars)
            self._bar_count += 1
        
        # 最后一天日结算
        if current_date:
            self._account_manager.settle_day(datetime.combine(current_date, datetime.min.time()))
        
        self._is_running = False
        
        # 计算绩效
        metrics = self._calculate_performance()
        
        # 打印结果
        elapsed = time.time() - self._start_time
        logger.info("="*50)
        logger.info("Backtest completed!")
        logger.info(f"Time elapsed: {elapsed:.2f}s")
        logger.info(f"Bars processed: {self._bar_count}")
        logger.info(f"Speed: {self._bar_count/elapsed:.0f} bars/s")
        logger.info("="*50)
        self._print_performance(metrics)
        
        # 触发完成回调
        self._emit("on_finish", metrics)
        
        return metrics
    
    def send_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT,
        strategy_name: str = ""
    ) -> str:
        """
        发送订单
        
        Args:
            symbol: 合约代码
            direction: 方向
            offset: 开平
            price: 价格
            volume: 数量
            order_type: 订单类型
            strategy_name: 策略名称
        
        Returns:
            订单ID
        """
        order = OrderData(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            exchange="",
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            order_type=order_type,
            strategy_name=strategy_name,
            create_time=self._current_datetime
        )
        
        self._match_engine.submit_order(order)
        self._emit("on_order", order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        return self._match_engine.cancel_order(order_id)
    
    def get_position(self, symbol: str, direction: Direction = None) -> Optional[PositionData]:
        """获取持仓"""
        return self._account_manager.get_position(symbol, direction)
    
    def get_account(self) -> AccountData:
        """获取账户信息"""
        return self._account_manager.get_account()
    
    def get_current_bar(self, symbol: str) -> Optional[BarData]:
        """获取当前K线"""
        return self._current_bars.get(symbol)
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def _emit(self, event_type: str, data: Any):
        """触发事件"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _calculate_performance(self) -> PerformanceMetrics:
        """计算绩效指标"""
        metrics = PerformanceMetrics()
        
        # 获取每日权益
        df = self._account_manager.get_daily_balance()
        if df.empty:
            return metrics
        
        # 获取交易记录
        trades = self._account_manager.get_trades()
        
        # 收益计算
        initial = self.config.initial_capital
        final = df['balance'].iloc[-1]
        metrics.total_return = (final - initial) / initial
        
        # 年化收益（假设252个交易日）
        days = len(df)
        if days > 0:
            metrics.annual_return = (1 + metrics.total_return) ** (252 / days) - 1
        
        # 日收益率统计
        returns = df['returns'].dropna()
        if len(returns) > 0:
            metrics.daily_return_mean = returns.mean()
            metrics.daily_return_std = returns.std()
        
        # 夏普比率（无风险利率假设3%）
        risk_free_daily = 0.03 / 252
        if metrics.daily_return_std > 0:
            metrics.sharpe_ratio = (
                (metrics.daily_return_mean - risk_free_daily) / 
                metrics.daily_return_std * np.sqrt(252)
            )
        
        # 最大回撤
        cummax = df['balance'].cummax()
        drawdown = (cummax - df['balance']) / cummax
        metrics.max_drawdown = drawdown.max()
        
        # 最大回撤持续时间
        in_drawdown = drawdown > 0
        if in_drawdown.any():
            drawdown_groups = (~in_drawdown).cumsum()
            drawdown_durations = in_drawdown.groupby(drawdown_groups).sum()
            metrics.max_drawdown_duration = int(drawdown_durations.max())
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                metrics.sortino_ratio = (
                    (metrics.daily_return_mean - risk_free_daily) / 
                    downside_std * np.sqrt(252)
                )
        
        # 卡尔马比率
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown
        
        # 交易统计
        metrics.total_trades = len(trades)
        
        # 计算每笔交易盈亏
        trade_pnls = []
        for i, trade in enumerate(trades):
            if trade.offset != Offset.OPEN:
                # 平仓交易的盈亏（简化计算）
                trade_pnls.append(trade.price * trade.volume * self.config.contract_size * 
                                  (1 if trade.direction == Direction.LONG else -1))
        
        if trade_pnls:
            winning = [p for p in trade_pnls if p > 0]
            losing = [p for p in trade_pnls if p < 0]
            
            metrics.winning_trades = len(winning)
            metrics.losing_trades = len(losing)
            metrics.win_rate = metrics.winning_trades / len(trade_pnls) if trade_pnls else 0
            
            metrics.avg_profit = np.mean(winning) if winning else 0
            metrics.avg_loss = np.mean(losing) if losing else 0
            
            if metrics.avg_loss != 0:
                metrics.profit_factor = abs(metrics.avg_profit / metrics.avg_loss)
        
        # VaR计算
        if len(returns) > 0:
            metrics.var_95 = np.percentile(returns, 5)
            metrics.var_99 = np.percentile(returns, 1)
        
        return metrics
    
    def _print_performance(self, metrics: PerformanceMetrics):
        """打印绩效报告"""
        account = self._account_manager.get_account()
        
        print("\n" + "="*60)
        print("                    回测绩效报告")
        print("="*60)
        
        print(f"\n【资金概况】")
        print(f"  初始资金:     {self.config.initial_capital:>15,.2f}")
        print(f"  最终权益:     {account.balance:>15,.2f}")
        print(f"  总收益:       {account.balance - self.config.initial_capital:>15,.2f}")
        
        print(f"\n【收益指标】")
        print(f"  总收益率:     {metrics.total_return:>14.2%}")
        print(f"  年化收益率:   {metrics.annual_return:>14.2%}")
        print(f"  日均收益率:   {metrics.daily_return_mean:>14.4%}")
        
        print(f"\n【风险指标】")
        print(f"  最大回撤:     {metrics.max_drawdown:>14.2%}")
        print(f"  回撤天数:     {metrics.max_drawdown_duration:>14}天")
        print(f"  日波动率:     {metrics.daily_return_std:>14.4%}")
        print(f"  95% VaR:      {metrics.var_95:>14.4%}")
        
        print(f"\n【风险调整收益】")
        print(f"  夏普比率:     {metrics.sharpe_ratio:>14.2f}")
        print(f"  索提诺比率:   {metrics.sortino_ratio:>14.2f}")
        print(f"  卡尔马比率:   {metrics.calmar_ratio:>14.2f}")
        
        print(f"\n【交易统计】")
        print(f"  总交易次数:   {metrics.total_trades:>14}")
        print(f"  盈利次数:     {metrics.winning_trades:>14}")
        print(f"  亏损次数:     {metrics.losing_trades:>14}")
        print(f"  胜率:         {metrics.win_rate:>14.2%}")
        print(f"  盈亏比:       {metrics.profit_factor:>14.2f}")
        
        print("\n" + "="*60)
        
        # 评估是否达标
        if metrics.is_valid():
            print("✅ 策略绩效达到基本要求!")
        else:
            print("⚠️ 策略绩效未达标，建议优化:")
            if metrics.sharpe_ratio < 1.5:
                print(f"   - 夏普比率偏低 ({metrics.sharpe_ratio:.2f} < 1.5)")
            if metrics.max_drawdown > 0.20:
                print(f"   - 最大回撤过大 ({metrics.max_drawdown:.2%} > 20%)")
            if metrics.win_rate < 0.45:
                print(f"   - 胜率偏低 ({metrics.win_rate:.2%} < 45%)")
    
    def get_result_dataframe(self) -> pd.DataFrame:
        """获取回测结果DataFrame"""
        return self._account_manager.get_daily_balance()
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        trades = self._account_manager.get_trades()
        if not trades:
            return pd.DataFrame()
        
        data = []
        for trade in trades:
            data.append({
                'datetime': trade.datetime,
                'symbol': trade.symbol,
                'direction': trade.direction.value,
                'offset': trade.offset.value,
                'price': trade.price,
                'volume': trade.volume,
                'commission': trade.commission,
                'strategy': trade.strategy_name
            })
        
        return pd.DataFrame(data)
