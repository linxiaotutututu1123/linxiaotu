"""
TickData数据模型单元测试。

测试覆盖：
- 字段类型验证
- 必填字段校验
- 价格/数量非负校验
- 时间戳合法性校验
- 不可变性验证
- __repr__方法验证
"""

import pytest
from datetime import datetime
from decimal import Decimal

# 为什么使用pytest.raises：确保非法输入抛出明确异常而非静默失败
from pydantic import ValidationError


class TestTickDataValidation:
    """TickData字段验证测试组。"""

    def test_create_valid_tick_data(self) -> None:
        """测试：合法参数应成功创建TickData实例。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=12345,
            open_interest=98765,
            timestamp=datetime(2024, 12, 11, 10, 30, 0, 123456),
            bid_price_1=Decimal("3850.00"),
            ask_price_1=Decimal("3850.40"),
            bid_volume_1=100,
            ask_volume_1=150,
        )
        
        assert tick.symbol == "IF2312"
        assert tick.last_price == Decimal("3850.20")

    def test_symbol_cannot_be_empty(self) -> None:
        """测试：空合约代码应抛出ValidationError。"""
        from market_gateway.core.models import TickData
        
        with pytest.raises(ValidationError) as exc_info:
            TickData(
                symbol="",  # 空字符串非法
                exchange="CFFEX",
                last_price=Decimal("3850.20"),
                volume=100,
                timestamp=datetime.now(),
            )
        
        # 为什么检查错误信息：确保用户能理解验证失败原因
        assert "symbol" in str(exc_info.value)

    def test_negative_price_rejected(self) -> None:
        """测试：负价格应抛出ValidationError。"""
        from market_gateway.core.models import TickData
        
        with pytest.raises(ValidationError) as exc_info:
            TickData(
                symbol="IF2312",
                exchange="CFFEX",
                last_price=Decimal("-100"),  # 负价格非法
                volume=100,
                timestamp=datetime.now(),
            )
        
        assert "last_price" in str(exc_info.value)

    def test_negative_volume_rejected(self) -> None:
        """测试：负成交量应抛出ValidationError。"""
        from market_gateway.core.models import TickData
        
        with pytest.raises(ValidationError) as exc_info:
            TickData(
                symbol="IF2312",
                exchange="CFFEX",
                last_price=Decimal("3850.20"),
                volume=-1,  # 负数量非法
                timestamp=datetime.now(),
            )
        
        assert "volume" in str(exc_info.value)


class TestTickDataImmutability:
    """TickData不可变性测试组。"""

    def test_tick_data_is_frozen(self) -> None:
        """测试：TickData实例创建后不可修改。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        # 为什么要冻结：防止行情数据被意外篡改导致策略逻辑错误
        with pytest.raises(Exception):  # pydantic frozen model raises
            tick.last_price = Decimal("9999")  # type: ignore


class TestTickDataRepr:
    """TickData调试输出测试组。"""

    def test_repr_contains_key_fields(self) -> None:
        """测试：__repr__应包含关键字段便于调试。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime(2024, 12, 11, 10, 30, 0),
        )
        
        repr_str = repr(tick)
        
        # 为什么检查这些字段：这是调试时最需要快速查看的信息
        assert "IF2312" in repr_str
        assert "3850.20" in repr_str or "3850.2" in repr_str


class TestTickDataRawTimestamp:
    """TickData原始时间戳测试组。"""

    def test_raw_timestamp_preserved(self) -> None:
        """测试：原始时间戳应被完整保留。"""
        from market_gateway.core.models import TickData
        
        # 为什么测试这个：确保高精度时间戳不丢失
        raw_ts = "1702267800123456"  # 飞马微秒级时间戳
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime(2024, 12, 11, 10, 30, 0),
            raw_timestamp=raw_ts,
            timestamp_source="femas",  # 标识来源
        )
        
        assert tick.raw_timestamp == raw_ts
        assert tick.timestamp_source == "femas"

    def test_raw_timestamp_optional(self) -> None:
        """测试：raw_timestamp是可选字段。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        # 为什么允许为空：兼容不提供原始时间戳的场景
        assert tick.raw_timestamp is None
        assert tick.timestamp_source is None

    def test_timestamp_source_validation(self) -> None:
        """测试：timestamp_source应只接受预定义的柜台类型。"""
        from market_gateway.core.models import TickData
        
        # 为什么限制来源：确保解析时能正确处理
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            raw_timestamp="10:30:00|500",
            timestamp_source="ctp",
        )
        
        assert tick.timestamp_source == "ctp"


class TestTickDataSchemaVersion:
    """TickData数据版本测试组。"""

    def test_schema_version_default(self) -> None:
        """测试：schema_version应有默认值。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        # 为什么需要版本号：便于数据迁移和兼容性检查
        assert tick.schema_version == 1

    def test_schema_version_in_serialization(self) -> None:
        """测试：序列化后应包含版本号。"""
        from market_gateway.core.models import TickData
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        dumped = tick.model_dump()
        
        # 为什么检查这个：确保持久化后能识别版本
        assert "schema_version" in dumped
        assert dumped["schema_version"] == 1


class TestTickDataTimestampConsistency:
    """TickData时间戳一致性测试组。"""

    def test_source_without_raw_timestamp_rejected(self) -> None:
        """测试：有timestamp_source但无raw_timestamp应报错。"""
        from market_gateway.core.models import TickData
        
        # 为什么禁止：逻辑矛盾，有来源标签却没有原始数据
        with pytest.raises(ValidationError) as exc_info:
            TickData(
                symbol="IF2312",
                exchange="CFFEX",
                last_price=Decimal("3850.20"),
                volume=100,
                timestamp=datetime.now(),
                timestamp_source="ctp",  # 有来源
                # 但没有 raw_timestamp！
            )
        
        assert "timestamp" in str(exc_info.value).lower()

    def test_raw_timestamp_without_source_allowed(self) -> None:
        """测试：有raw_timestamp但无source应允许（兼容旧数据）。"""
        from market_gateway.core.models import TickData
        
        # 为什么允许：兼容未标记来源的历史数据
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            raw_timestamp="10:30:00|500",  # 有原始数据
            # 但没有 timestamp_source
        )
        
        assert tick.raw_timestamp == "10:30:00|500"
        assert tick.timestamp_source is None
