# 量化交易系统 (Quantitative Trading System)

## 🚀 项目简介

这是一个面向中国期货市场的**全自动量化交易系统**，采用模块化设计，集成了数据管理、策略开发、回测引擎、风险控制、AI预测和交易执行等完整功能。

### 核心特性

- 📊 **多数据源支持**: CTP、天勤量化(TqSdk)、Tushare等
- 🤖 **AI增强**: LSTM/Transformer预测、情感分析、强化学习
- ⚡ **高性能回测**: 事件驱动架构，支持参数优化
- 🛡️ **五层风控**: 订单级 → 账户级 → 策略级 → 组合级 → 系统级
- 📈 **丰富策略**: 趋势跟踪、均值回归、套利、自适应等
- 🌐 **Web监控**: 实时仪表盘，WebSocket推送

### 性能目标

| 指标 | 目标值 |
|------|--------|
| 年化收益率 | >30% |
| 夏普比率 | >2.0 |
| 最大回撤 | <15% |
| 胜率 | >55% |
| 盈亏比 | >1.5 |

---

## 📁 项目结构

```
QuantTradingSystem/
├── config/                     # 配置文件
│   └── settings.py            # 系统配置
├── src/                       # 核心源码
│   ├── data/                  # 数据层
│   │   ├── data_structures.py # 数据结构定义
│   │   └── data_manager.py    # 数据管理器
│   ├── backtest/              # 回测引擎
│   │   ├── backtest_engine.py # 回测核心
│   │   └── optimizer.py       # 参数优化器
│   ├── strategy/              # 策略框架
│   │   ├── strategy_base.py   # 策略基类
│   │   └── advanced_strategies.py # 高级策略
│   ├── risk/                  # 风险控制
│   │   └── risk_manager.py    # 风险管理器
│   ├── ai/                    # AI模块
│   │   ├── predictors.py      # LSTM/Transformer预测
│   │   ├── sentiment.py       # 情感分析
│   │   └── reinforcement.py   # 强化学习
│   └── execution/             # 交易执行
│       ├── execution_engine.py # 执行引擎
│       └── auto_trader.py     # 自动交易器
├── frontend/                  # Web前端
│   ├── api_server.py          # FastAPI后端
│   └── index.html             # 监控界面
├── tests/                     # 测试用例
├── data/                      # 数据存储
├── logs/                      # 日志文件
├── main.py                    # 主入口
├── requirements.txt           # 依赖列表
└── README.md                  # 说明文档
```

---

## 🔧 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置系统

编辑 `config/settings.py` 或创建 `config/settings.yaml`:

```yaml
ctp:
  broker_id: "9999"
  user_id: "your_user_id"
  password: "your_password"
  td_address: "tcp://180.168.146.187:10101"
  md_address: "tcp://180.168.146.187:10111"

risk:
  max_position_per_symbol: 100
  max_daily_loss: 50000
  max_drawdown: 0.15

trading:
  symbols:
    - rb2501
    - i2501
    - hc2501
```

### 3. 运行系统

```bash
# 查看帮助
python main.py --help

# 运行回测
python main.py backtest -s dual_ma --symbol rb2501 --start 2023-01-01 --end 2024-01-01

# 模拟交易
python main.py paper -s rb2501,i2501

# 启动监控
python main.py monitor

# 实盘交易 (谨慎!)
python main.py live -b ctp
```

---

## 📊 策略说明

### 内置策略

| 策略 | 类型 | 说明 |
|------|------|------|
| DualMAStrategy | 趋势跟踪 | 双均线交叉策略 |
| TurtleStrategy | 趋势跟踪 | 海龟交易法则 |
| GridStrategy | 震荡 | 网格交易策略 |
| MomentumStrategy | 动量 | 动量突破策略 |
| MeanReversionStrategy | 均值回归 | 布林带回归策略 |
| SpreadArbitrageStrategy | 套利 | 跨期套利策略 |
| MultiStrategyPortfolio | 组合 | 多策略组合 |
| AdaptiveStrategy | 自适应 | 根据市场状态切换策略 |

### 自定义策略

继承 `BaseStrategy` 类：

```python
from src.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy", symbols=["rb2501"])
    
    def on_bar(self, bar):
        # 获取技术指标
        sma_fast = self.sma(bar.close, 5)
        sma_slow = self.sma(bar.close, 20)
        
        # 生成信号
        if sma_fast > sma_slow and self.position <= 0:
            return self.buy(bar.symbol, bar.close, 1)
        elif sma_fast < sma_slow and self.position >= 0:
            return self.sell(bar.symbol, bar.close, 1)
        
        return None
```

---

## 🛡️ 风险控制

### 五层风控体系

1. **订单级**: 单笔订单金额/数量限制
2. **账户级**: 最大持仓、保证金占用
3. **策略级**: 策略最大亏损、连续亏损次数
4. **组合级**: 组合相关性、集中度
5. **系统级**: 熔断机制、紧急平仓

### 仓位管理

- **凯利公式**: 基于胜率和盈亏比计算最优仓位
- **VaR计算**: 历史模拟法、参数法、蒙特卡洛
- **动态止损**: ATR追踪止损

---

## 🤖 AI模块

### 价格预测

```python
from src.ai import LSTMPredictor, ModelConfig

config = ModelConfig(sequence_length=60, hidden_size=128)
predictor = LSTMPredictor(config)
predictor.train(X_train, y_train)
predictions = predictor.predict(X_test)
```

### 情感分析

```python
from src.ai import NewsSentimentAnalyzer

analyzer = NewsSentimentAnalyzer()
result = analyzer.analyze_text("螺纹钢期货大涨，市场情绪乐观")
print(result['sentiment'])  # 正面情感得分
```

### 强化学习

```python
from src.ai import DQNAgent, RLTrainer

trainer = RLTrainer(df, agent_type="dqn")
trainer.train(num_episodes=1000)
result = trainer.evaluate()
```

---

## 🌐 Web监控

启动监控服务后访问 `http://localhost:8000`:

- 实时账户信息
- 持仓监控
- 订单管理
- 策略状态
- 快速下单

### WebSocket订阅

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({ action: 'subscribe', channel: 'positions' }));
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

---

## 📝 开发计划

- [x] 核心框架
- [x] 数据管理
- [x] 回测引擎
- [x] 风险控制
- [x] 策略框架
- [x] AI预测
- [x] 交易执行
- [x] Web监控
- [ ] 实盘对接完善
- [ ] 更多数据源
- [ ] 策略市场
- [ ] 移动端APP

---

## ⚠️ 风险提示

**本系统仅供学习研究使用，实盘交易风险自负！**

期货交易具有高杠杆、高风险特性，可能导致本金亏损。请在充分了解风险的前提下谨慎使用。

---

## 📄 许可证

MIT License

---

## 🤝 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
