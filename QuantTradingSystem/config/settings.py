"""
顶级量化交易系统 - 全局配置文件
作者: QuantMaster
版本: 1.0.0
更新: 2025-12-11
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# ==================== 基础路径配置 ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# ==================== 交易所配置 ====================
class Exchange(Enum):
    """支持的交易所"""
    SHFE = "上海期货交易所"      # 铜、铝、锌、铅、镍、锡、黄金、白银、螺纹钢、线材、热轧卷板、燃料油、石油沥青、天然橡胶
    DCE = "大连商品交易所"       # 玉米、玉米淀粉、黄大豆、豆粕、豆油、棕榈油、鸡蛋、纤维板、胶合板、聚乙烯、聚氯乙烯、聚丙烯、焦炭、焦煤、铁矿石
    CZCE = "郑州商品交易所"      # 强麦、普麦、棉花、白糖、PTA、菜籽油、早籼稻、甲醇、玻璃、油菜籽、菜籽粕、动力煤、粳稻、晚籼稻、硅铁、锰硅、棉纱
    CFFEX = "中国金融期货交易所"  # 沪深300股指期货、上证50股指期货、中证500股指期货、2年期国债期货、5年期国债期货、10年期国债期货
    INE = "上海国际能源交易中心"   # 原油期货、20号胶、低硫燃料油
    GFEX = "广州期货交易所"       # 工业硅、碳酸锂

# ==================== CTP接口配置 ====================
@dataclass
class CTPConfig:
    """CTP接口配置"""
    # 行情前置地址
    md_front: str = "tcp://180.168.146.187:10131"  # simnow模拟环境
    # 交易前置地址
    td_front: str = "tcp://180.168.146.187:10130"  # simnow模拟环境
    # Broker ID
    broker_id: str = "9999"
    # 用户ID (需替换为真实账号)
    user_id: str = ""
    # 密码 (需替换为真实密码)
    password: str = ""
    # 认证码
    auth_code: str = ""
    # AppID
    app_id: str = ""
    # 是否使用模拟盘
    is_simulate: bool = True

# ==================== 数据库配置 ====================
@dataclass
class DatabaseConfig:
    """数据库配置"""
    # PostgreSQL配置 (存储历史数据、用户数据)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "quant_trading"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # InfluxDB配置 (存储实时时序数据)
    influxdb_host: str = "localhost"
    influxdb_port: int = 8086
    influxdb_db: str = "market_data"
    influxdb_token: str = ""
    influxdb_org: str = "quant"
    
    # Redis配置 (缓存)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

# ==================== 风险控制配置 ====================
@dataclass
class RiskConfig:
    """风险控制参数配置"""
    # 第一层：单笔交易风险
    max_loss_per_trade: float = 0.02          # 单笔最大亏损比例 2%
    max_profit_per_trade: float = 0.10        # 单笔最大盈利比例 10%（止盈）
    
    # 第二层：日内风险
    max_daily_loss: float = 0.05              # 日内最大亏损比例 5%
    max_daily_profit: float = 0.15            # 日内最大盈利比例 15%（止盈保护）
    max_daily_trades: int = 50                # 日内最大交易次数
    
    # 第三层：品种风险
    max_position_per_symbol: float = 0.15     # 单品种最大仓位比例 15%
    max_correlated_exposure: float = 0.30     # 相关品种组合最大敞口 30%
    
    # 第四层：组合风险
    max_portfolio_var: float = 0.03           # 组合VaR限制 3%
    max_total_position: float = 0.80          # 最大总仓位 80%
    max_drawdown_threshold: float = 0.10      # 最大回撤阈值 10%
    
    # 第五层：极端风险
    black_swan_circuit_breaker: bool = True   # 黑天鹅熔断机制
    volatility_scaling: bool = True           # 波动率自适应仓位
    max_leverage: float = 3.0                 # 最大杠杆倍数
    
    # 流动性风险
    min_volume_threshold: int = 2000          # 最小成交量阈值（手/分钟）
    max_slippage: float = 0.002               # 最大滑点容忍度 0.2%

# ==================== 策略配置 ====================
@dataclass
class StrategyConfig:
    """策略配置"""
    # 策略资金分配比例
    allocation: Dict[str, float] = field(default_factory=lambda: {
        "trend_following": 0.40,      # 趋势跟踪策略 40%
        "mean_reversion": 0.25,       # 均值回归策略 25%
        "arbitrage": 0.20,            # 套利策略 20%
        "high_frequency": 0.10,       # 高频策略 10%
        "ml_strategy": 0.05           # 机器学习策略 5%
    })
    
    # 回测配置
    backtest_start_date: str = "2015-01-01"
    backtest_end_date: str = "2024-12-31"
    initial_capital: float = 1000000.0        # 初始资金100万
    commission_rate: float = 0.0001           # 手续费率万分之一
    slippage_rate: float = 0.0001             # 滑点率万分之一
    
    # 参数优化配置
    optimization_method: str = "bayesian"      # bayesian / grid / random
    max_optimization_iterations: int = 1000
    cross_validation_folds: int = 5
    
    # 过拟合检测
    min_trades_for_validation: int = 200       # 最小交易次数
    parameter_sensitivity_threshold: float = 0.30  # 参数敏感性阈值

# ==================== AI模型配置 ====================
@dataclass
class AIConfig:
    """AI模型配置"""
    # LSTM配置
    lstm_sequence_length: int = 60            # 序列长度
    lstm_hidden_size: int = 128               # 隐藏层大小
    lstm_num_layers: int = 2                  # LSTM层数
    lstm_dropout: float = 0.2                 # Dropout比例
    
    # Transformer配置
    transformer_d_model: int = 256            # 模型维度
    transformer_nhead: int = 8                # 注意力头数
    transformer_num_layers: int = 6           # 编码器层数
    transformer_dropout: float = 0.1          # Dropout比例
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # 模型路径
    model_save_path: str = str(BASE_DIR / "models")

# ==================== 执行配置 ====================
@dataclass
class ExecutionConfig:
    """交易执行配置"""
    # 订单执行策略
    default_order_type: str = "LIMIT"          # LIMIT / MARKET
    large_order_threshold: int = 100           # 大单阈值（手）
    twap_time_slices: int = 10                 # TWAP拆分数量
    vwap_participation_rate: float = 0.10      # VWAP参与率
    
    # 滑点控制
    aggressive_price_offset: float = 0.0001    # 激进执行价格偏移
    passive_price_offset: float = -0.0001      # 被动执行价格偏移
    
    # 延迟监控
    max_order_latency_ms: int = 50             # 最大订单延迟（毫秒）
    max_data_latency_ms: int = 100             # 最大数据延迟（毫秒）

# ==================== 绩效目标配置 ====================
@dataclass
class PerformanceTarget:
    """绩效目标配置"""
    # 保守目标
    conservative_annual_return: float = 0.20   # 年化收益 20%
    conservative_max_drawdown: float = 0.10    # 最大回撤 10%
    conservative_sharpe_ratio: float = 1.5     # 夏普比率 1.5
    
    # 中等目标
    moderate_annual_return: float = 0.35       # 年化收益 35%
    moderate_max_drawdown: float = 0.15        # 最大回撤 15%
    moderate_sharpe_ratio: float = 2.0         # 夏普比率 2.0
    
    # 激进目标
    aggressive_annual_return: float = 0.50     # 年化收益 50%+
    aggressive_max_drawdown: float = 0.20      # 最大回撤 20%
    aggressive_sharpe_ratio: float = 2.5       # 夏普比率 2.5+
    
    # 通用指标要求
    min_monthly_win_rate: float = 0.65         # 月度胜率 65%
    max_drawdown_duration_days: int = 60       # 最长回撤周期 60天
    min_profit_factor: float = 2.0             # 盈亏比 2.0

# ==================== 日志配置 ====================
@dataclass
class LogConfig:
    """日志配置"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = str(LOG_DIR / "quant_system.log")
    max_log_size: int = 10 * 1024 * 1024       # 10MB
    backup_count: int = 5

# ==================== 全局配置实例 ====================
class Settings:
    """全局配置管理"""
    def __init__(self):
        self.ctp = CTPConfig()
        self.database = DatabaseConfig()
        self.risk = RiskConfig()
        self.strategy = StrategyConfig()
        self.ai = AIConfig()
        self.execution = ExecutionConfig()
        self.performance = PerformanceTarget()
        self.log = LogConfig()
    
    def load_from_env(self):
        """从环境变量加载敏感配置"""
        self.ctp.user_id = os.getenv("CTP_USER_ID", "")
        self.ctp.password = os.getenv("CTP_PASSWORD", "")
        self.ctp.auth_code = os.getenv("CTP_AUTH_CODE", "")
        self.database.postgres_password = os.getenv("POSTGRES_PASSWORD", "password")
        self.database.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.database.influxdb_token = os.getenv("INFLUXDB_TOKEN", "")

# 创建全局配置实例
settings = Settings()
