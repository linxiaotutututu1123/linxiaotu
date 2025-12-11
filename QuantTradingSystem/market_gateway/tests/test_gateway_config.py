"""
GatewayConfig配置模型单元测试。

测试覆盖：
- 必填字段校验
- 默认值验证
- 超时参数范围校验
- 环境变量加载
- 密码安全性（SecretStr）
"""

import pytest
from pydantic import ValidationError


class TestGatewayConfigValidation:
    """GatewayConfig字段验证测试组。"""

    def test_create_minimal_config(self) -> None:
        """测试：最小必填参数应成功创建配置。"""
        from market_gateway.core.models import GatewayConfig
        
        config = GatewayConfig(
            broker_id="9999",
            user_id="test_user",
            password="test_password",
            front_addr="tcp://180.168.146.187:10131",
        )
        
        assert config.broker_id == "9999"
        assert config.front_addr.startswith("tcp://")

    def test_password_is_secret(self) -> None:
        """测试：密码字段应被SecretStr保护。"""
        from market_gateway.core.models import GatewayConfig
        
        config = GatewayConfig(
            broker_id="9999",
            user_id="test_user",
            password="my_secret_123",
            front_addr="tcp://127.0.0.1:10131",
        )
        
        # 为什么这样测试：确保序列化时密码不会泄露
        dumped = config.model_dump()
        assert "my_secret_123" not in str(dumped)
        
        # 为什么需要get_secret_value：显式获取才能拿到真实密码
        assert config.password.get_secret_value() == "my_secret_123"

    def test_password_hidden_in_repr(self) -> None:
        """测试：__repr__中密码应被隐藏。"""
        from market_gateway.core.models import GatewayConfig
        
        config = GatewayConfig(
            broker_id="9999",
            user_id="test_user",
            password="super_secret",
            front_addr="tcp://127.0.0.1:10131",
        )
        
        repr_str = repr(config)
        
        # 为什么检查：防止日志泄露密码
        assert "super_secret" not in repr_str
        assert "9999" in repr_str  # broker_id应可见

    def test_empty_broker_id_rejected(self) -> None:
        """测试：空broker_id应抛出ValidationError。"""
        from market_gateway.core.models import GatewayConfig
        
        with pytest.raises(ValidationError):
            GatewayConfig(
                broker_id="",  # 空字符串非法
                user_id="test_user",
                password="test_password",
                front_addr="tcp://127.0.0.1:10131",
            )

    def test_invalid_front_addr_rejected(self) -> None:
        """测试：非法前置地址格式应抛出ValidationError。"""
        from market_gateway.core.models import GatewayConfig
        
        # 为什么校验地址格式：防止运行时连接失败
        with pytest.raises(ValidationError):
            GatewayConfig(
                broker_id="9999",
                user_id="test_user",
                password="test_password",
                front_addr="invalid_address",  # 缺少tcp://前缀
            )


class TestReconnectConfigValidation:
    """ReconnectConfig重连配置验证测试组。"""

    def test_default_reconnect_values(self) -> None:
        """测试：默认重连参数应合理。"""
        from market_gateway.core.models import ReconnectConfig
        
        config = ReconnectConfig()
        
        # 为什么检查默认值：确保合理的开箱即用体验
        assert config.immediate_retry_count >= 1
        assert config.max_backoff_seconds > 0
        assert config.scheduled_check_interval_seconds > 0

    def test_negative_retry_count_rejected(self) -> None:
        """测试：负重试次数应抛出ValidationError。"""
        from market_gateway.core.models import ReconnectConfig
        
        with pytest.raises(ValidationError):
            ReconnectConfig(immediate_retry_count=-1)

    def test_zero_backoff_rejected(self) -> None:
        """测试：零退避时间应抛出ValidationError。"""
        from market_gateway.core.models import ReconnectConfig
        
        # 为什么禁止零退避：防止重连风暴压垮服务器
        with pytest.raises(ValidationError):
            ReconnectConfig(max_backoff_seconds=0)
