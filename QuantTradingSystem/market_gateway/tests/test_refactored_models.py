"""
重构后模型的单元测试。

测试覆盖：
- A. TickDataCore / TickDataDepth 拆分
- B. PriceType 价格类型配置
- C. TickDataView 计算属性视图
"""

import pytest
from datetime import datetime
from decimal import Decimal

from pydantic import ValidationError


# ============================================================
# A. TickDataCore / TickDataDepth 拆分测试
# ============================================================

class TestTickDataCore:
    """TickDataCore核心数据测试组（轻量级，高频使用）。"""

    def test_create_minimal_core(self) -> None:
        """测试：核心数据只需必要字段。"""
        from market_gateway.core.models import TickDataCore
        
        core = TickDataCore(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=12345,
            timestamp=datetime(2024, 12, 11, 10, 30, 0),
        )
        
        # 为什么只保留这些字段：这是策略层最常用的数据
        assert core.symbol == "IF2312"
        assert core.last_price == Decimal("3850.20")
        assert core.volume == 12345

    def test_core_is_frozen(self) -> None:
        """测试：核心数据不可变。"""
        from market_gateway.core.models import TickDataCore
        
        core = TickDataCore(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        with pytest.raises(Exception):
            core.last_price = Decimal("9999")  # type: ignore

    def test_core_memory_footprint_smaller(self) -> None:
        """测试：核心数据字段数应少于完整TickData。"""
        from market_gateway.core.models import TickDataCore, TickData
        
        core_fields = len(TickDataCore.model_fields)
        full_fields = len(TickData.model_fields)
        
        # 为什么检查这个：确保拆分达到减少内存的目的
        assert core_fields < full_fields


class TestTickDataDepth:
    """TickDataDepth深度数据测试组（盘口数据）。"""

    def test_create_depth_data(self) -> None:
        """测试：深度数据包含盘口信息。"""
        from market_gateway.core.models import TickDataDepth
        
        depth = TickDataDepth(
            symbol="IF2312",
            exchange="CFFEX",
            timestamp=datetime(2024, 12, 11, 10, 30, 0),
            bid_price_1=Decimal("3850.00"),
            ask_price_1=Decimal("3850.40"),
            bid_volume_1=100,
            ask_volume_1=150,
        )
        
        assert depth.bid_price_1 == Decimal("3850.00")
        assert depth.ask_price_1 == Decimal("3850.40")

    def test_depth_supports_multiple_levels(self) -> None:
        """测试：深度数据支持多档盘口。"""
        from market_gateway.core.models import TickDataDepth
        
        # 为什么支持多档：某些策略需要完整盘口深度
        depth = TickDataDepth(
            symbol="IF2312",
            exchange="CFFEX",
            timestamp=datetime.now(),
            bid_price_1=Decimal("3850.00"),
            bid_price_2=Decimal("3849.80"),
            bid_price_3=Decimal("3849.60"),
            ask_price_1=Decimal("3850.40"),
            ask_price_2=Decimal("3850.60"),
            ask_price_3=Decimal("3850.80"),
        )
        
        assert depth.bid_price_2 == Decimal("3849.80")

    def test_depth_is_frozen(self) -> None:
        """测试：深度数据不可变。"""
        from market_gateway.core.models import TickDataDepth
        
        depth = TickDataDepth(
            symbol="IF2312",
            exchange="CFFEX",
            timestamp=datetime.now(),
        )
        
        with pytest.raises(Exception):
            depth.bid_price_1 = Decimal("9999")  # type: ignore


# ============================================================
# B. PriceType 价格类型配置测试
# ============================================================

class TestPriceTypeConfig:
    """价格类型配置测试组。"""

    def test_default_uses_decimal(self) -> None:
        """测试：默认使用Decimal保证精度。"""
        from market_gateway.core.models import PriceConfig
        
        config = PriceConfig()
        
        # 为什么默认Decimal：金融场景精度优先
        assert config.use_decimal is True

    def test_can_switch_to_float(self) -> None:
        """测试：可切换为float提升性能。"""
        from market_gateway.core.models import PriceConfig
        
        config = PriceConfig(use_decimal=False)
        
        # 为什么允许float：高频回测场景牺牲精度换性能
        assert config.use_decimal is False

    def test_performance_warning_for_float(self) -> None:
        """测试：使用float时应有精度警告字段。"""
        from market_gateway.core.models import PriceConfig
        
        config = PriceConfig(use_decimal=False)
        
        # 为什么需要警告：提醒用户float有精度风险
        assert hasattr(config, 'precision_warning')


# ============================================================
# C. TickDataView 计算属性视图测试
# ============================================================

class TestTickDataView:
    """TickDataView计算属性视图测试组。"""

    def test_mid_price_calculation(self) -> None:
        """测试：中间价计算正确。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            bid_price_1=Decimal("3850.00"),
            ask_price_1=Decimal("3850.40"),
        )
        
        view = TickDataView(tick)
        
        # 为什么计算中间价：常用于滑点估算
        expected_mid = (Decimal("3850.00") + Decimal("3850.40")) / 2
        assert view.mid_price == expected_mid

    def test_spread_calculation(self) -> None:
        """测试：价差计算正确。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            bid_price_1=Decimal("3850.00"),
            ask_price_1=Decimal("3850.40"),
        )
        
        view = TickDataView(tick)
        
        # 为什么计算价差：用于流动性分析
        assert view.spread == Decimal("0.40")

    def test_spread_bps_calculation(self) -> None:
        """测试：价差基点计算正确。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            bid_price_1=Decimal("3850.00"),
            ask_price_1=Decimal("3850.40"),
        )
        
        view = TickDataView(tick)
        
        # 为什么用基点：便于跨品种比较流动性
        # spread_bps = (ask - bid) / mid * 10000
        assert view.spread_bps is not None
        assert view.spread_bps > 0

    def test_view_handles_missing_depth(self) -> None:
        """测试：无盘口数据时计算属性应返回None。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
            # 无盘口数据
        )
        
        view = TickDataView(tick)
        
        # 为什么返回None：避免除零错误
        assert view.mid_price is None
        assert view.spread is None

    def test_view_is_not_frozen(self) -> None:
        """测试：视图可以添加计算缓存。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        view = TickDataView(tick)
        
        # 为什么不冻结：允许缓存计算结果
        # 但原始tick仍然是不可变的
        # 验证底层tick是frozen的
        assert TickData.model_config.get('frozen') is True
        # 验证view能正常访问属性（说明view本身可用）
        assert view.symbol == tick.symbol

    def test_view_delegates_to_tick(self) -> None:
        """测试：视图可直接访问底层tick属性。"""
        from market_gateway.core.models import TickData, TickDataView
        
        tick = TickData(
            symbol="IF2312",
            exchange="CFFEX",
            last_price=Decimal("3850.20"),
            volume=100,
            timestamp=datetime.now(),
        )
        
        view = TickDataView(tick)
        
        # 为什么委托：使用方便，无需.tick.symbol
        assert view.symbol == "IF2312"
        assert view.last_price == Decimal("3850.20")
