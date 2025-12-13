# 行情网关接口契约

## 1. 异常类型定义

所有网关异常继承自 `GatewayException`，便于统一捕获与处理。

| 异常类 | 场景 | HTTP等价 |
|--------|------|----------|
| `GatewayException` | 网关基础异常 | 500 |
| `ConnectionError` | 连接失败（网络不可达） | 503 |
| `AuthenticationError` | 认证失败（账户/密码错误） | 401 |
| `SubscriptionError` | 订阅失败（合约不存在/超限） | 400 |
| `ValidationError` | 数据校验失败 | 422 |
| `TimeoutError` | 操作超时 | 504 |
| `ReconnectExhaustedError` | 重连次数耗尽 | 503 |

## 2. 抽象基类签名

```python
from abc import ABC, abstractmethod
from typing import Callable, Set, Optional
from datetime import datetime

class AbstractMarketGateway(ABC):
    """
    行情网关抽象基类。
    
    定义所有行情网关必须实现的接口契约。
    子类需实现具体柜台协议（CTP、恒生、IB等）。
    
    Attributes:
        name: 网关标识名称
        connected: 当前连接状态
        subscribed_symbols: 已订阅合约集合
    
    Example:
        >>> gateway = CtpMarketGateway(config)
        >>> await gateway.connect()
        >>> await gateway.subscribe(["IF2401", "IC2401"])
        >>> async for tick in gateway.tick_stream():
        ...     process(tick)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """网关唯一标识名称。"""
        ...
    
    @property
    @abstractmethod
    def connected(self) -> bool:
        """当前是否已连接并登录成功。"""
        ...
    
    @property
    @abstractmethod
    def subscribed_symbols(self) -> Set[str]:
        """当前已订阅的合约代码集合（只读副本）。"""
        ...
    
    @abstractmethod
    async def connect(
        self,
        timeout: float = 10.0,
        retry: int = 3
    ) -> None:
        """
        建立与柜台的连接并完成登录。
        
        Args:
            timeout: 单次连接超时秒数，默认10秒
            retry: 连接失败重试次数，默认3次
        
        Raises:
            ConnectionError: 网络不可达或服务器拒绝连接
            AuthenticationError: 用户名/密码/BrokerID错误
            TimeoutError: 连接超时且重试耗尽
        
        Example:
            >>> await gateway.connect(timeout=5.0, retry=5)  # 正确
            >>> await gateway.connect(timeout=-1)  # ValueError
            >>> await gateway.connect()  # 使用默认参数
        """
        ...
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        断开连接并释放资源。
        
        幂等操作：多次调用不会抛出异常。
        断开前会自动取消所有订阅。
        
        Raises:
            GatewayException: 断开过程中发生内部错误
        """
        ...
    
    @abstractmethod
    async def subscribe(
        self,
        symbols: list[str],
        *,
        ignore_duplicates: bool = True
    ) -> Set[str]:
        """
        订阅行情数据。
        
        Args:
            symbols: 合约代码列表，如 ["IF2401", "IC2401"]
            ignore_duplicates: 是否忽略重复订阅，默认True
        
        Returns:
            实际新增订阅的合约集合
        
        Raises:
            SubscriptionError: 合约代码无效或订阅数超限
            ConnectionError: 未连接状态下调用
        
        Note:
            - 单连接订阅上限1000个合约
            - 幂等：重复订阅同一合约不会报错（ignore_duplicates=True）
        
        Example:
            >>> added = await gateway.subscribe(["IF2401"])
            >>> added = await gateway.subscribe(["IF2401"])  # 返回空集合
        """
        ...
    
    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> Set[str]:
        """
        取消订阅。
        
        Args:
            symbols: 需取消订阅的合约代码列表
        
        Returns:
            实际取消的合约集合
        
        Raises:
            ConnectionError: 未连接状态下调用
        """
        ...
    
    @abstractmethod
    def set_tick_callback(
        self,
        callback: Callable[["TickData"], None]
    ) -> None:
        """
        设置Tick数据回调函数。
        
        Args:
            callback: 接收TickData的同步回调函数
        
        Note:
            回调函数应尽快返回，耗时操作请提交到队列异步处理。
        """
        ...
    
    @abstractmethod
    async def tick_stream(self) -> "AsyncIterator[TickData]":
        """
        异步迭代器方式获取Tick流。
        
        Yields:
            TickData: 标准化行情数据
        
        Example:
            >>> async for tick in gateway.tick_stream():
            ...     await strategy.on_tick(tick)
        """
        ...
```

## 3. 方法调用状态约束

```
                    ┌─────────────────┐
                    │   Created       │
                    │ (初始化完成)     │
                    └────────┬────────┘
                             │ connect()
                             ▼
                    ┌─────────────────┐
                    │   Connected     │◄───┐
                    │ (已连接未订阅)   │    │
                    └────────┬────────┘    │
                             │ subscribe()  │ unsubscribe_all()
                             ▼              │
                    ┌─────────────────┐    │
                    │   Subscribed    │────┘
                    │ (已订阅行情)     │
                    └────────┬────────┘
                             │ disconnect()
                             ▼
                    ┌─────────────────┐
                    │   Disconnected  │
                    │ (已断开)         │
                    └─────────────────┘
```

## 4. 线程安全与并发约束

| 方法 | 线程安全 | 并发限制 |
|------|----------|----------|
| `connect` | 是 | 同一时刻仅允许一个调用 |
| `disconnect` | 是 | 幂等，可多次调用 |
| `subscribe` | 是 | 可并发调用，内部加锁 |
| `unsubscribe` | 是 | 可并发调用，内部加锁 |
| `tick_stream` | 是 | 可多消费者并发读取 |
