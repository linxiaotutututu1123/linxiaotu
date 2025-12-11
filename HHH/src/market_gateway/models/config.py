"""
行情网关配置模型定义。

本模块定义网关配置相关的数据模型：
- GatewayConfig: 网关主配置
- ReconnectConfig: 重连策略配置
- ServerConfig: 服务器连接配置

使用 Pydantic v2 进行配置校验与环境变量加载。
"""

from __future__ import annotations

__all__ = [
    "GatewayConfig",
    "ReconnectConfig",
    "ServerConfig",
]

from typing import Optional
import os

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    SecretStr,
    ConfigDict,
)
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """
    服务器连接配置。
    
    Attributes:
        host: 服务器地址
        port: 端口号
        
    Example:
        >>> server = ServerConfig(host="180.168.146.187", port=10211)
        >>> repr(server)
        "ServerConfig(host=180.168.146.187, port=10211)"
    """
    
    model_config = ConfigDict(frozen=True)
    
    host: str = Field(..., min_length=1, description="服务器地址")
    port: int = Field(..., ge=1, le=65535, description="端口号")
    
    def __repr__(self) -> str:
        """调试友好的字符串表示。"""
        return f"ServerConfig(host={self.host}, port={self.port})"
    
    @property
    def address(self) -> str:
        """
        返回完整连接地址。
        
        # WHY: CTP 使用 tcp:// 协议前缀
        """
        return f"tcp://{self.host}:{self.port}"


class ReconnectConfig(BaseModel):
    """
    重连策略配置。
    
    使用指数退避算法：wait = min(base * 2^attempt, max_interval)
    
    Attributes:
        max_attempts: 最大重连次数，超过后触发告警
        base_interval: 基础等待时间（秒）
        max_interval: 最大等待时间（秒）
        auto_resubscribe: 重连后是否自动恢复订阅
        
    Example:
        >>> config = ReconnectConfig(max_attempts=10)
        >>> # 重连间隔序列: 1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s, 60s, 60s
        
    # RISK: 无限重试可能导致日志膨胀
    # 缓解措施: max_attempts=10 后发送告警并停止
    """
    
    model_config = ConfigDict(frozen=True)
    
    # WHY: 10次重试覆盖约3分钟，足够应对临时网络抖动
    max_attempts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="最大重连次数"
    )
    
    # WHY: 1秒起步避免瞬间重试风暴
    base_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="基础等待时间（秒）"
    )
    
    # WHY: 60秒上限避免等待过久错过交易时段
    max_interval: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="最大等待时间（秒）"
    )
    
    auto_resubscribe: bool = Field(
        default=True,
        description="重连后是否自动恢复订阅"
    )
    
    def get_wait_time(self, attempt: int) -> float:
        """
        计算第 attempt 次重连的等待时间。
        
        Args:
            attempt: 当前重试次数（从0开始）
            
        Returns:
            等待时间（秒）
            
        # WHY: 指数退避减少服务器压力，同时有上限避免等待过久
        """
        wait = self.base_interval * (2 ** attempt)
        return min(wait, self.max_interval)
    
    def __repr__(self) -> str:
        """调试友好的字符串表示。"""
        return (
            f"ReconnectConfig(max={self.max_attempts}, "
            f"base={self.base_interval}s, cap={self.max_interval}s)"
        )


class GatewayConfig(BaseSettings):
    """
    行情网关主配置。
    
    支持从环境变量加载敏感信息（账户、密码）。
    
    Attributes:
        name: 网关唯一标识
        broker_id: 期货公司代码
        user_id: 用户账号
        password: 用户密码（SecretStr 类型）
        md_server: 行情服务器配置
        td_server: 交易服务器配置（可选）
        reconnect: 重连策略配置
        connect_timeout: 连接超时（秒）
        subscribe_limit: 单连接订阅上限
        
    Example:
        >>> # 从环境变量加载
        >>> os.environ["CTP_USER_ID"] = "123456"
        >>> os.environ["CTP_PASSWORD"] = "secret"
        >>> config = GatewayConfig(
        ...     name="ctp_main",
        ...     broker_id="9999",
        ...     md_server=ServerConfig(host="180.168.146.187", port=10211)
        ... )
        
    # RISK: 密码明文日志泄露
    # 缓解措施: 使用 SecretStr，repr/日志自动脱敏
    """
    
    model_config = ConfigDict(
        env_prefix="CTP_",  # 环境变量前缀
        env_file=".env",    # 支持 .env 文件
        extra="forbid",     # 禁止未知字段
    )
    
    # === 网关标识 ===
    name: str = Field(
        default="ctp_gateway",
        min_length=1,
        max_length=50,
        description="网关唯一标识"
    )
    
    # === 认证信息 ===
    broker_id: str = Field(..., min_length=1, description="期货公司代码")
    
    # WHY: 从环境变量读取，避免硬编码
    user_id: str = Field(
        ...,
        min_length=1,
        description="用户账号",
        json_schema_extra={"env": "CTP_USER_ID"}
    )
    
    # WHY: SecretStr 自动脱敏，防止日志泄露
    password: SecretStr = Field(
        ...,
        description="用户密码",
        json_schema_extra={"env": "CTP_PASSWORD"}
    )
    
    # === 服务器配置 ===
    md_server: ServerConfig = Field(..., description="行情服务器")
    td_server: Optional[ServerConfig] = Field(
        None, 
        description="交易服务器（可选）"
    )
    
    # === 重连配置 ===
    reconnect: ReconnectConfig = Field(
        default_factory=ReconnectConfig,
        description="重连策略"
    )
    
    # === 超时与限制 ===
    # WHY: 10秒超时适合国内网络环境
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="连接超时（秒）"
    )
    
    # WHY: CTP单连接上限约1000合约
    subscribe_limit: int = Field(
        default=1000,
        ge=1,
        le=2000,
        description="单连接订阅上限"
    )
    
    @field_validator("broker_id")
    @classmethod
    def validate_broker_id(cls, v: str) -> str:
        """
        校验期货公司代码格式。
        
        # WHY: 标准 broker_id 为4位数字
        """
        cleaned = v.strip()
        if not cleaned.isdigit():
            raise ValueError("broker_id 必须为纯数字")
        return cleaned
    
    def __repr__(self) -> str:
        """
        调试友好的字符串表示。
        
        # WHY: 密码脱敏显示，安全合规
        """
        return (
            f"GatewayConfig(name={self.name}, "
            f"broker={self.broker_id}, user={self.user_id}, "
            f"server={self.md_server.address})"
        )
