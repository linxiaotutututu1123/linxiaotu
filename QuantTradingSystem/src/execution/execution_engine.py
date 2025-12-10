"""
交易执行引擎
负责订单管理、执行和仓位控制
"""

import asyncio
import threading
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import time
from abc import ABC, abstractmethod

import sys
sys.path.append('..')

from data.data_structures import (
    OrderData, TradeData, PositionData, AccountData,
    OrderStatus, Direction, OrderType
)

logger = logging.getLogger(__name__)


# ==================== 执行配置 ====================

@dataclass
class ExecutionConfig:
    """执行配置"""
    # 订单配置
    default_slippage: float = 1.0     # 默认滑点(跳)
    max_order_value: float = 1000000  # 单笔最大下单金额
    
    # 执行算法配置
    twap_interval: int = 60           # TWAP间隔(秒)
    vwap_participation: float = 0.1   # VWAP参与率
    iceberg_show_size: float = 0.1    # 冰山单显示比例
    
    # 重试配置
    max_retries: int = 3
    retry_interval: float = 0.5       # 秒
    
    # 超时配置
    order_timeout: int = 60           # 订单超时(秒)
    
    # 风控
    max_open_orders: int = 100        # 最大挂单数
    position_check_interval: int = 5  # 仓位检查间隔(秒)


# ==================== 订单事件 ====================

class OrderEvent(Enum):
    """订单事件类型"""
    SUBMITTED = "submitted"       # 已提交
    ACCEPTED = "accepted"         # 已接受
    REJECTED = "rejected"         # 已拒绝
    PARTIALLY_FILLED = "partial"  # 部分成交
    FILLED = "filled"            # 全部成交
    CANCELLED = "cancelled"      # 已取消
    EXPIRED = "expired"          # 已过期
    ERROR = "error"              # 错误


@dataclass
class OrderEventData:
    """订单事件数据"""
    order_id: str
    event: OrderEvent
    timestamp: datetime
    order: OrderData
    message: str = ""
    trade: Optional[TradeData] = None


# ==================== 执行算法 ====================

class ExecutionAlgorithm(ABC):
    """执行算法基类"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    @abstractmethod
    def split_order(self, order: OrderData) -> List[OrderData]:
        """拆分订单"""
        pass
    
    @abstractmethod
    def get_next_slice(self) -> Optional[OrderData]:
        """获取下一个子订单"""
        pass


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    TWAP算法 (Time Weighted Average Price)
    在指定时间内均匀执行订单
    """
    
    def __init__(
        self, 
        config: ExecutionConfig,
        total_time: int = 3600,    # 总执行时间(秒)
        num_slices: int = 10      # 拆分数量
    ):
        super().__init__(config)
        self.total_time = total_time
        self.num_slices = num_slices
        self.slices: List[OrderData] = []
        self.current_index = 0
        self.start_time = None
    
    def split_order(self, order: OrderData) -> List[OrderData]:
        """将订单拆分为均匀的子订单"""
        slice_volume = order.volume / self.num_slices
        self.slices = []
        
        for i in range(self.num_slices):
            slice_order = OrderData(
                symbol=order.symbol,
                direction=order.direction,
                order_type=order.order_type,
                volume=slice_volume,
                price=order.price,
                order_id=f"{order.order_id}_slice_{i}",
                timestamp=datetime.now()
            )
            self.slices.append(slice_order)
        
        self.current_index = 0
        self.start_time = datetime.now()
        
        return self.slices
    
    def get_next_slice(self) -> Optional[OrderData]:
        """获取下一个需要执行的子订单"""
        if self.current_index >= len(self.slices):
            return None
        
        # 计算当前应执行的切片
        elapsed = (datetime.now() - self.start_time).total_seconds()
        target_index = int(elapsed / self.total_time * self.num_slices)
        
        if target_index > self.current_index:
            slice_order = self.slices[self.current_index]
            self.current_index += 1
            return slice_order
        
        return None


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    VWAP算法 (Volume Weighted Average Price)
    根据历史成交量分布执行订单
    """
    
    def __init__(
        self, 
        config: ExecutionConfig,
        volume_profile: Dict[int, float] = None  # 分钟 -> 成交量比例
    ):
        super().__init__(config)
        # 默认成交量分布（按小时）
        self.volume_profile = volume_profile or {
            9: 0.15,    # 开盘
            10: 0.12,
            11: 0.10,
            13: 0.08,
            14: 0.20,   # 午后
            15: 0.35,   # 尾盘
        }
        self.slices: List[OrderData] = []
        self.current_index = 0
    
    def split_order(self, order: OrderData) -> List[OrderData]:
        """根据成交量分布拆分订单"""
        total_ratio = sum(self.volume_profile.values())
        self.slices = []
        
        for hour, ratio in self.volume_profile.items():
            volume = order.volume * ratio / total_ratio
            slice_order = OrderData(
                symbol=order.symbol,
                direction=order.direction,
                order_type=order.order_type,
                volume=volume,
                price=order.price,
                order_id=f"{order.order_id}_vwap_{hour}",
                timestamp=datetime.now()
            )
            self.slices.append(slice_order)
        
        return self.slices
    
    def get_next_slice(self) -> Optional[OrderData]:
        """获取当前时段的子订单"""
        current_hour = datetime.now().hour
        
        for i, (hour, _) in enumerate(self.volume_profile.items()):
            if hour == current_hour and i >= self.current_index:
                self.current_index = i + 1
                return self.slices[i] if i < len(self.slices) else None
        
        return None


class IcebergAlgorithm(ExecutionAlgorithm):
    """
    冰山算法
    只显示部分订单量，隐藏真实下单量
    """
    
    def __init__(
        self, 
        config: ExecutionConfig,
        show_ratio: float = 0.1    # 显示比例
    ):
        super().__init__(config)
        self.show_ratio = show_ratio
        self.remaining_volume = 0
        self.original_order: Optional[OrderData] = None
    
    def split_order(self, order: OrderData) -> List[OrderData]:
        """拆分为冰山订单"""
        self.original_order = order
        self.remaining_volume = order.volume
        return []  # 冰山单动态生成
    
    def get_next_slice(self) -> Optional[OrderData]:
        """获取下一个显示订单"""
        if self.remaining_volume <= 0 or self.original_order is None:
            return None
        
        show_volume = min(
            self.original_order.volume * self.show_ratio,
            self.remaining_volume
        )
        
        slice_order = OrderData(
            symbol=self.original_order.symbol,
            direction=self.original_order.direction,
            order_type=self.original_order.order_type,
            volume=show_volume,
            price=self.original_order.price,
            order_id=f"{self.original_order.order_id}_ice_{int(time.time()*1000)}",
            timestamp=datetime.now()
        )
        
        self.remaining_volume -= show_volume
        return slice_order


# ==================== 交易接口 ====================

class BrokerGateway(ABC):
    """
    交易接口基类
    对接不同的交易通道
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接交易服务器"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def send_order(self, order: OrderData) -> str:
        """发送订单，返回订单ID"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    async def query_order(self, order_id: str) -> Optional[OrderData]:
        """查询订单"""
        pass
    
    @abstractmethod
    async def query_position(self, symbol: str = None) -> List[PositionData]:
        """查询持仓"""
        pass
    
    @abstractmethod
    async def query_account(self) -> AccountData:
        """查询账户"""
        pass


class CTPGateway(BrokerGateway):
    """
    CTP交易接口
    对接期货CTP系统
    """
    
    def __init__(
        self,
        broker_id: str,
        user_id: str,
        password: str,
        td_address: str,
        md_address: str
    ):
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.td_address = td_address
        self.md_address = md_address
        
        self.connected = False
        self._order_ref = 0
        self._orders: Dict[str, OrderData] = {}
    
    async def connect(self) -> bool:
        """连接CTP"""
        try:
            # 实际实现需要导入CTP库
            # from vnpy_ctp import CtpGateway
            logger.info("Connecting to CTP...")
            
            # 模拟连接
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info("CTP connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"CTP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """断开CTP"""
        self.connected = False
        logger.info("CTP disconnected")
    
    async def send_order(self, order: OrderData) -> str:
        """发送订单到CTP"""
        if not self.connected:
            raise RuntimeError("CTP not connected")
        
        self._order_ref += 1
        order_id = f"CTP_{self._order_ref}_{int(time.time()*1000)}"
        
        # 实际实现调用CTP API
        # self.td_api.ReqOrderInsert(...)
        
        order.order_id = order_id
        order.status = OrderStatus.SUBMITTED
        self._orders[order_id] = order
        
        logger.info(f"Order submitted: {order_id}, {order.symbol} {order.direction.value} {order.volume}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False
    
    async def query_order(self, order_id: str) -> Optional[OrderData]:
        """查询订单"""
        return self._orders.get(order_id)
    
    async def query_position(self, symbol: str = None) -> List[PositionData]:
        """查询持仓"""
        # 实际实现调用CTP查询
        return []
    
    async def query_account(self) -> AccountData:
        """查询账户"""
        # 实际实现调用CTP查询
        return AccountData(
            account_id=self.user_id,
            balance=1000000,
            available=800000,
            frozen=200000,
            timestamp=datetime.now()
        )


class TqsdkGateway(BrokerGateway):
    """
    天勤量化交易接口
    """
    
    def __init__(self, account: str = None, password: str = None):
        self.account = account
        self.password = password
        self.connected = False
        self._api = None
    
    async def connect(self) -> bool:
        """连接天勤"""
        try:
            # from tqsdk import TqApi, TqAccount
            logger.info("Connecting to Tqsdk...")
            
            # 模拟连接
            self.connected = True
            logger.info("Tqsdk connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Tqsdk connection failed: {e}")
            return False
    
    async def disconnect(self):
        self.connected = False
    
    async def send_order(self, order: OrderData) -> str:
        if not self.connected:
            raise RuntimeError("Tqsdk not connected")
        
        order_id = f"TQ_{int(time.time()*1000)}"
        order.order_id = order_id
        order.status = OrderStatus.SUBMITTED
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        return True
    
    async def query_order(self, order_id: str) -> Optional[OrderData]:
        return None
    
    async def query_position(self, symbol: str = None) -> List[PositionData]:
        return []
    
    async def query_account(self) -> AccountData:
        return AccountData(
            account_id=self.account or "demo",
            balance=1000000,
            available=1000000,
            frozen=0,
            timestamp=datetime.now()
        )


# ==================== 订单管理器 ====================

class OrderManager:
    """
    订单管理器
    管理订单生命周期
    """
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        
        # 订单存储
        self._orders: Dict[str, OrderData] = {}
        self._pending_orders: Dict[str, OrderData] = {}  # 待执行
        self._active_orders: Dict[str, OrderData] = {}   # 活跃
        
        # 事件回调
        self._callbacks: List[Callable[[OrderEventData], None]] = []
        
        # 订单队列
        self._order_queue: queue.Queue = queue.Queue()
    
    def add_order(self, order: OrderData):
        """添加订单"""
        self._orders[order.order_id] = order
        self._pending_orders[order.order_id] = order
        
        self._notify(OrderEventData(
            order_id=order.order_id,
            event=OrderEvent.SUBMITTED,
            timestamp=datetime.now(),
            order=order
        ))
    
    def update_order(self, order_id: str, status: OrderStatus, trade: TradeData = None):
        """更新订单状态"""
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        old_status = order.status
        order.status = status
        
        # 状态转换
        if status == OrderStatus.ACTIVE:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            self._active_orders[order_id] = order
            event = OrderEvent.ACCEPTED
            
        elif status == OrderStatus.FILLED:
            if order_id in self._active_orders:
                del self._active_orders[order_id]
            event = OrderEvent.FILLED
            
        elif status == OrderStatus.PARTIALLY_FILLED:
            event = OrderEvent.PARTIALLY_FILLED
            
        elif status == OrderStatus.CANCELLED:
            if order_id in self._active_orders:
                del self._active_orders[order_id]
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            event = OrderEvent.CANCELLED
            
        elif status == OrderStatus.REJECTED:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            event = OrderEvent.REJECTED
            
        else:
            event = OrderEvent.ERROR
        
        self._notify(OrderEventData(
            order_id=order_id,
            event=event,
            timestamp=datetime.now(),
            order=order,
            trade=trade
        ))
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self._active_orders:
            self.update_order(order_id, OrderStatus.CANCELLED)
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[OrderData]:
        """获取订单"""
        return self._orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[OrderData]:
        """获取活跃订单"""
        orders = list(self._active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def register_callback(self, callback: Callable[[OrderEventData], None]):
        """注册事件回调"""
        self._callbacks.append(callback)
    
    def _notify(self, event: OrderEventData):
        """通知所有回调"""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# ==================== 执行引擎 ====================

class ExecutionEngine:
    """
    交易执行引擎
    核心执行逻辑
    """
    
    def __init__(
        self,
        gateway: BrokerGateway,
        config: ExecutionConfig = None
    ):
        self.gateway = gateway
        self.config = config or ExecutionConfig()
        
        self.order_manager = OrderManager(config)
        self._running = False
        self._event_loop = None
        
        # 算法执行器
        self._algo_executions: Dict[str, ExecutionAlgorithm] = {}
        
        # 仓位缓存
        self._positions: Dict[str, PositionData] = {}
        self._account: Optional[AccountData] = None
    
    async def start(self):
        """启动引擎"""
        logger.info("Starting execution engine...")
        
        # 连接交易接口
        connected = await self.gateway.connect()
        if not connected:
            raise RuntimeError("Failed to connect to broker")
        
        self._running = True
        
        # 启动后台任务
        asyncio.create_task(self._position_sync_loop())
        
        logger.info("Execution engine started")
    
    async def stop(self):
        """停止引擎"""
        logger.info("Stopping execution engine...")
        self._running = False
        
        # 取消所有活跃订单
        for order in self.order_manager.get_active_orders():
            await self.cancel_order(order.order_id)
        
        await self.gateway.disconnect()
        logger.info("Execution engine stopped")
    
    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        volume: float,
        price: float = None,
        order_type: OrderType = OrderType.LIMIT,
        algo: str = None
    ) -> str:
        """
        提交订单
        
        Args:
            symbol: 合约代码
            direction: 方向
            volume: 数量
            price: 价格(市价单可为None)
            order_type: 订单类型
            algo: 执行算法 (twap/vwap/iceberg/None)
        
        Returns:
            订单ID
        """
        # 创建订单
        order = OrderData(
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            volume=volume,
            price=price or 0,
            order_id=f"ORD_{int(time.time()*1000)}",
            timestamp=datetime.now()
        )
        
        # 风控检查
        if not self._pre_trade_check(order):
            logger.warning(f"Order rejected by risk check: {order.order_id}")
            return ""
        
        # 使用执行算法
        if algo:
            return await self._execute_with_algo(order, algo)
        
        # 直接执行
        return await self._execute_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        success = await self.gateway.cancel_order(order_id)
        if success:
            self.order_manager.update_order(order_id, OrderStatus.CANCELLED)
        return success
    
    async def modify_order(
        self, 
        order_id: str, 
        price: float = None, 
        volume: float = None
    ) -> bool:
        """修改订单（撤单重发）"""
        order = self.order_manager.get_order(order_id)
        if not order:
            return False
        
        # 先取消
        await self.cancel_order(order_id)
        
        # 重新下单
        new_order = OrderData(
            symbol=order.symbol,
            direction=order.direction,
            order_type=order.order_type,
            volume=volume or order.volume,
            price=price or order.price,
            order_id=f"ORD_{int(time.time()*1000)}",
            timestamp=datetime.now()
        )
        
        await self._execute_order(new_order)
        return True
    
    def _pre_trade_check(self, order: OrderData) -> bool:
        """交易前风控检查"""
        # 检查挂单数
        active_orders = self.order_manager.get_active_orders()
        if len(active_orders) >= self.config.max_open_orders:
            logger.warning("Too many open orders")
            return False
        
        # 检查订单金额
        order_value = order.volume * order.price
        if order_value > self.config.max_order_value:
            logger.warning(f"Order value {order_value} exceeds max {self.config.max_order_value}")
            return False
        
        return True
    
    async def _execute_order(self, order: OrderData) -> str:
        """执行单个订单"""
        self.order_manager.add_order(order)
        
        try:
            order_id = await self.gateway.send_order(order)
            order.order_id = order_id
            self.order_manager.update_order(order_id, OrderStatus.ACTIVE)
            return order_id
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            self.order_manager.update_order(order.order_id, OrderStatus.REJECTED)
            return ""
    
    async def _execute_with_algo(self, order: OrderData, algo_name: str) -> str:
        """使用算法执行"""
        # 创建算法
        if algo_name == "twap":
            algo = TWAPAlgorithm(self.config)
        elif algo_name == "vwap":
            algo = VWAPAlgorithm(self.config)
        elif algo_name == "iceberg":
            algo = IcebergAlgorithm(self.config)
        else:
            logger.warning(f"Unknown algorithm: {algo_name}")
            return await self._execute_order(order)
        
        # 拆分订单
        algo.split_order(order)
        self._algo_executions[order.order_id] = algo
        
        # 启动算法执行
        asyncio.create_task(self._algo_execution_loop(order.order_id))
        
        return order.order_id
    
    async def _algo_execution_loop(self, parent_order_id: str):
        """算法执行循环"""
        algo = self._algo_executions.get(parent_order_id)
        if not algo:
            return
        
        while self._running:
            slice_order = algo.get_next_slice()
            if slice_order is None:
                # 检查是否完成
                await asyncio.sleep(1)
                continue
            
            await self._execute_order(slice_order)
            await asyncio.sleep(self.config.twap_interval)
    
    async def _position_sync_loop(self):
        """仓位同步循环"""
        while self._running:
            try:
                # 查询持仓
                positions = await self.gateway.query_position()
                self._positions = {p.symbol: p for p in positions}
                
                # 查询账户
                self._account = await self.gateway.query_account()
                
            except Exception as e:
                logger.error(f"Position sync failed: {e}")
            
            await asyncio.sleep(self.config.position_check_interval)
    
    def get_position(self, symbol: str) -> Optional[PositionData]:
        """获取持仓"""
        return self._positions.get(symbol)
    
    def get_account(self) -> Optional[AccountData]:
        """获取账户"""
        return self._account
    
    def get_all_positions(self) -> Dict[str, PositionData]:
        """获取所有持仓"""
        return self._positions.copy()
