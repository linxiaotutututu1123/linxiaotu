# 行情网关模块架构设计

## 1. 系统架构图

```mermaid
graph TB
    subgraph 外部系统
        CTP[CTP柜台]
        SimNow[SimNow仿真]
        Exchange[交易所接口]
    end

    subgraph 网关层 Gateway Layer
        AGW[AbstractMarketGateway<br/>抽象基类]
        CGW[CtpMarketGateway<br/>CTP实现]
        SGW[SimNowGateway<br/>仿真实现]
        
        AGW --> CGW
        AGW --> SGW
    end

    subgraph 数据转换层 Transform Layer
        RawParser[RawDataParser<br/>原始数据解析]
        Validator[DataValidator<br/>数据校验器]
        Cleaner[DataCleaner<br/>数据清洗器]
        Converter[TickConverter<br/>格式转换器]
        
        RawParser --> Validator
        Validator --> Cleaner
        Cleaner --> Converter
    end

    subgraph 存储层 Storage Layer
        MemCache[MemoryCache<br/>内存缓存60s]
        AsyncWriter[AsyncDBWriter<br/>异步写入器]
        AuditLog[AuditLogger<br/>审计日志]
    end

    subgraph 发布订阅层 PubSub Layer
        EventBus[EventBus<br/>事件总线]
        Queue[asyncio.Queue<br/>内存队列]
        RedisPub[RedisPubSub<br/>可选分布式]
    end

    subgraph 消费者 Consumers
        Strategy[策略引擎]
        RiskCtrl[风控引擎]
        Monitor[监控系统]
        Recorder[行情录制]
    end

    CTP --> CGW
    SimNow --> SGW
    Exchange -.-> AGW
    
    CGW --> RawParser
    SGW --> RawParser
    
    Converter --> MemCache
    Converter --> EventBus
    Cleaner --> AuditLog
    
    MemCache --> AsyncWriter
    EventBus --> Queue
    Queue --> RedisPub
    
    Queue --> Strategy
    Queue --> RiskCtrl
    RedisPub --> Monitor
    AsyncWriter --> Recorder
```

## 2. 数据流时序图

```mermaid
sequenceDiagram
    participant CTP as CTP柜台
    participant GW as CtpMarketGateway
    participant VAL as DataValidator
    participant CONV as TickConverter
    participant CACHE as MemoryCache
    participant BUS as EventBus
    participant STRAT as 策略引擎

    CTP->>GW: OnRtnDepthMarketData(raw)
    GW->>VAL: validate(raw)
    
    alt 数据有效
        VAL->>CONV: convert(raw)
        CONV->>CACHE: cache(tick)
        CONV->>BUS: publish(MarketEvent)
        BUS->>STRAT: on_tick(tick)
    else 数据无效
        VAL->>GW: log_audit(raw, reason)
    end
```

## 3. 断线重连状态机

```mermaid
stateDiagram-v2
    [*] --> Disconnected
    Disconnected --> Connecting: connect()
    Connecting --> Connected: 登录成功
    Connecting --> WaitRetry: 登录失败
    Connected --> Subscribed: subscribe()
    Subscribed --> Connected: unsubscribe()
    Connected --> Disconnected: 网络断开
    Subscribed --> Disconnected: 网络断开
    WaitRetry --> Connecting: 退避时间到
    WaitRetry --> Failed: 重试次数>=10
    Failed --> Disconnected: reset()
    Failed --> [*]: 人工介入
```

## 4. 核心类职责（单一职责原则）

| 类名 | 职责 | 依赖 |
|------|------|------|
| `AbstractMarketGateway` | 定义网关抽象接口，生命周期管理 | - |
| `CtpMarketGateway` | CTP协议具体实现，回调处理 | pyctp, AbstractMarketGateway |
| `GatewayConfig` | 网关配置模型（服务器、账户、超时） | pydantic |
| `TickData` | 标准化行情数据模型 | pydantic |
| `DepthData` | 深度行情数据模型（5档） | pydantic |
| `DataValidator` | 数据有效性校验（价格、量、时间） | - |
| `DataCleaner` | 数据清洗与标准化 | DataValidator |
| `SubscriptionManager` | 订阅列表管理（幂等、上限控制） | - |
| `ReconnectPolicy` | 重连策略（指数退避、次数限制） | - |
| `EventBus` | 事件发布订阅（asyncio.Queue） | asyncio |
| `MemoryCache` | 内存缓存（LRU, 60s TTL） | - |
| `AsyncDBWriter` | 异步数据库写入（批量、重试） | asyncio, asyncpg |
| `AuditLogger` | 审计日志（脏数据、异常事件） | structlog |

## 5. 模块依赖关系

```mermaid
graph LR
    subgraph core[核心层]
        models[models.py<br/>数据模型]
        exceptions[exceptions.py<br/>异常定义]
        constants[constants.py<br/>常量配置]
    end

    subgraph gateway[网关层]
        base[base.py<br/>抽象基类]
        ctp[ctp_gateway.py<br/>CTP实现]
        subscription[subscription.py<br/>订阅管理]
        reconnect[reconnect.py<br/>重连策略]
    end

    subgraph transform[转换层]
        validator[validator.py<br/>数据校验]
        cleaner[cleaner.py<br/>数据清洗]
        converter[converter.py<br/>格式转换]
    end

    subgraph storage[存储层]
        cache[cache.py<br/>内存缓存]
        writer[writer.py<br/>异步写入]
        audit[audit.py<br/>审计日志]
    end

    subgraph pubsub[发布订阅]
        eventbus[eventbus.py<br/>事件总线]
        events[events.py<br/>事件类型]
    end

    base --> models
    base --> exceptions
    ctp --> base
    ctp --> subscription
    ctp --> reconnect
    validator --> models
    cleaner --> validator
    converter --> cleaner
    cache --> models
    writer --> cache
    eventbus --> events
    events --> models
```
