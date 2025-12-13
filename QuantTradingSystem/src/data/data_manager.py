"""
数据层 - 多源数据管理器
支持CTP、天勤量化、Tushare、本地文件等多种数据源
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
import threading
from queue import Queue
import time

from .data_structures import (
    TickData, BarData, ContractData, 
    Direction, DataBuffer
)

logger = logging.getLogger(__name__)


# ==================== 数据源基类 ====================

class BaseDataSource(ABC):
    """数据源基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            "tick": [],
            "bar": [],
            "status": []
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """订阅行情"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    @abstractmethod
    async def get_historical_bars(
        self, 
        symbol: str, 
        interval: str,
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """获取历史K线数据"""
        pass
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def _emit(self, event_type: str, data: Any):
        """触发回调"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# ==================== 本地数据源（CSV/Parquet） ====================

class LocalDataSource(BaseDataSource):
    """
    本地数据源
    支持CSV和Parquet格式的历史数据
    """
    
    def __init__(self, data_dir: str):
        super().__init__("local")
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    async def connect(self) -> bool:
        """连接（检查目录是否存在）"""
        if self.data_dir.exists():
            self.is_connected = True
            logger.info(f"Local data source connected: {self.data_dir}")
            return True
        logger.error(f"Data directory not found: {self.data_dir}")
        return False
    
    async def disconnect(self):
        """断开（清理缓存）"""
        self._cache.clear()
        self.is_connected = False
    
    async def subscribe(self, symbols: List[str]):
        """订阅（本地数据不需要订阅）"""
        pass
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    async def get_historical_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            symbol: 合约代码
            interval: K线周期 (1m, 5m, 15m, 30m, 1h, 1d)
            start: 开始时间
            end: 结束时间
        
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{interval}"
        
        # 尝试从缓存获取
        if cache_key in self._cache:
            df = self._cache[cache_key]
            mask = (df['datetime'] >= start) & (df['datetime'] <= end)
            return df.loc[mask].copy()
        
        # 尝试加载文件
        file_patterns = [
            f"{symbol}_{interval}.csv",
            f"{symbol}_{interval}.parquet",
            f"{symbol.lower()}_{interval}.csv",
            f"{symbol.upper()}_{interval}.csv",
        ]
        
        df = None
        for pattern in file_patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path, parse_dates=['datetime'])
                elif file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                break
        
        if df is None:
            logger.warning(f"Data file not found for {symbol}_{interval}")
            return pd.DataFrame()
        
        # 确保datetime列格式正确
        if 'datetime' not in df.columns and 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 排序并缓存
        df = df.sort_values('datetime').reset_index(drop=True)
        self._cache[cache_key] = df
        
        # 筛选时间范围
        mask = (df['datetime'] >= start) & (df['datetime'] <= end)
        return df.loc[mask].copy()
    
    def save_bars(self, symbol: str, interval: str, df: pd.DataFrame, format: str = "parquet"):
        """
        保存K线数据到本地
        
        Args:
            symbol: 合约代码
            interval: K线周期
            df: 数据DataFrame
            format: 保存格式 (csv/parquet)
        """
        file_path = self.data_dir / f"{symbol}_{interval}.{format}"
        
        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "parquet":
            df.to_parquet(file_path, index=False)
        
        logger.info(f"Saved {len(df)} bars to {file_path}")


# ==================== 天勤量化数据源 ====================

class TqDataSource(BaseDataSource):
    """
    天勤量化数据源
    使用tqsdk获取实时和历史数据
    """
    
    def __init__(self, user: str = "", password: str = ""):
        super().__init__("tq")
        self.user = user
        self.password = password
        self._api = None
        self._subscribed: Dict[str, Any] = {}
    
    async def connect(self) -> bool:
        """连接天勤量化"""
        try:
            from tqsdk import TqApi, TqAuth
            
            if self.user and self.password:
                self._api = TqApi(auth=TqAuth(self.user, self.password))
            else:
                self._api = TqApi()
            
            self.is_connected = True
            logger.info("TqSdk connected successfully")
            return True
        except ImportError:
            logger.error("tqsdk not installed. Run: pip install tqsdk")
            return False
        except Exception as e:
            logger.error(f"TqSdk connection failed: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self._api:
            self._api.close()
            self._api = None
        self.is_connected = False
        self._subscribed.clear()
    
    async def subscribe(self, symbols: List[str]):
        """订阅行情"""
        if not self._api:
            return
        
        for symbol in symbols:
            if symbol not in self._subscribed:
                quote = self._api.get_quote(symbol)
                self._subscribed[symbol] = quote
                logger.info(f"Subscribed: {symbol}")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self._subscribed:
                del self._subscribed[symbol]
    
    async def get_historical_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """获取历史K线数据"""
        if not self._api:
            return pd.DataFrame()
        
        try:
            # 转换周期
            duration_map = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "30m": 1800,
                "1h": 3600,
                "1d": 86400
            }
            duration = duration_map.get(interval, 60)
            
            # 计算需要的K线数量
            days = (end - start).days + 1
            data_length = days * 24 * 3600 // duration
            data_length = min(data_length, 8000)  # 天勤限制
            
            klines = self._api.get_kline_serial(symbol, duration, data_length)
            
            # 转换为DataFrame
            df = pd.DataFrame({
                'datetime': pd.to_datetime(klines.datetime, unit='ns'),
                'open': klines.open,
                'high': klines.high,
                'low': klines.low,
                'close': klines.close,
                'volume': klines.volume,
                'open_interest': klines.open_oi
            })
            
            # 筛选时间范围
            mask = (df['datetime'] >= start) & (df['datetime'] <= end)
            return df.loc[mask].reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> Optional[TickData]:
        """获取实时行情"""
        if symbol not in self._subscribed:
            return None
        
        quote = self._subscribed[symbol]
        
        return TickData(
            symbol=symbol,
            exchange=symbol.split('.')[0] if '.' in symbol else "",
            datetime=datetime.fromtimestamp(quote.datetime / 1e9),
            last_price=quote.last_price,
            open_price=quote.open,
            high_price=quote.highest,
            low_price=quote.lowest,
            pre_close=quote.pre_close,
            pre_settlement=quote.pre_settlement,
            volume=quote.volume,
            turnover=quote.amount,
            open_interest=quote.open_interest,
            upper_limit=quote.upper_limit,
            lower_limit=quote.lower_limit,
            bid_price_1=quote.bid_price1,
            bid_volume_1=quote.bid_volume1,
            ask_price_1=quote.ask_price1,
            ask_volume_1=quote.ask_volume1
        )


# ==================== Tushare数据源 ====================

class TushareDataSource(BaseDataSource):
    """
    Tushare数据源
    获取股票、期货历史数据
    """
    
    def __init__(self, token: str = ""):
        super().__init__("tushare")
        self.token = token
        self._pro = None
    
    async def connect(self) -> bool:
        """连接Tushare"""
        try:
            import tushare as ts
            
            if self.token:
                ts.set_token(self.token)
            self._pro = ts.pro_api()
            
            self.is_connected = True
            logger.info("Tushare connected successfully")
            return True
        except ImportError:
            logger.error("tushare not installed. Run: pip install tushare")
            return False
        except Exception as e:
            logger.error(f"Tushare connection failed: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        self._pro = None
        self.is_connected = False
    
    async def subscribe(self, symbols: List[str]):
        """订阅（Tushare不支持实时订阅）"""
        logger.warning("Tushare doesn't support realtime subscription")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    async def get_historical_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """获取历史K线数据"""
        if not self._pro:
            return pd.DataFrame()
        
        try:
            start_str = start.strftime("%Y%m%d")
            end_str = end.strftime("%Y%m%d")
            
            # 期货日线数据
            if interval == "1d":
                df = self._pro.fut_daily(
                    ts_code=symbol,
                    start_date=start_str,
                    end_date=end_str
                )
            else:
                # 分钟数据需要使用其他接口
                logger.warning("Tushare free version doesn't support minute data")
                return pd.DataFrame()
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                'trade_date': 'datetime',
                'vol': 'volume',
                'oi': 'open_interest'
            })
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            
        except Exception as e:
            logger.error(f"Failed to get Tushare data: {e}")
            return pd.DataFrame()


# ==================== 数据管理器 ====================

class DataManager:
    """
    数据管理器 - 统一管理多个数据源
    支持数据缓存、清洗、同步
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._sources: Dict[str, BaseDataSource] = {}
        self._primary_source: Optional[str] = None
        self._bar_buffers: Dict[str, DataBuffer] = {}
        self._tick_buffers: Dict[str, DataBuffer] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "tick": [],
            "bar": [],
            "error": []
        }
        self._running = False
        
    def add_source(self, source: BaseDataSource, is_primary: bool = False):
        """添加数据源"""
        self._sources[source.name] = source
        if is_primary:
            self._primary_source = source.name
        logger.info(f"Added data source: {source.name}, primary={is_primary}")
    
    async def connect_all(self) -> bool:
        """连接所有数据源"""
        results = []
        for name, source in self._sources.items():
            try:
                result = await source.connect()
                results.append(result)
                if result:
                    logger.info(f"Data source {name} connected")
                else:
                    logger.warning(f"Data source {name} connection failed")
            except Exception as e:
                logger.error(f"Error connecting {name}: {e}")
                results.append(False)
        
        return any(results)
    
    async def disconnect_all(self):
        """断开所有数据源"""
        for source in self._sources.values():
            await source.disconnect()
        self._running = False
    
    async def subscribe(self, symbols: List[str], source_name: str = None):
        """订阅行情"""
        source_name = source_name or self._primary_source
        if source_name and source_name in self._sources:
            await self._sources[source_name].subscribe(symbols)
            
            # 初始化数据缓冲区
            for symbol in symbols:
                if symbol not in self._bar_buffers:
                    self._bar_buffers[symbol] = DataBuffer(max_size=5000)
                if symbol not in self._tick_buffers:
                    self._tick_buffers[symbol] = DataBuffer(max_size=10000)
    
    async def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        source_name: str = None
    ) -> pd.DataFrame:
        """
        获取K线数据（自动选择最优数据源）
        
        Args:
            symbol: 合约代码
            interval: K线周期
            start: 开始时间
            end: 结束时间
            source_name: 指定数据源名称
        
        Returns:
            DataFrame with OHLCV data
        """
        # 优先使用指定数据源
        if source_name and source_name in self._sources:
            return await self._sources[source_name].get_historical_bars(
                symbol, interval, start, end
            )
        
        # 按优先级尝试各数据源
        source_priority = [
            self._primary_source,
            "local",
            "tq",
            "tushare"
        ]
        
        for src_name in source_priority:
            if src_name and src_name in self._sources:
                source = self._sources[src_name]
                if source.is_connected:
                    df = await source.get_historical_bars(symbol, interval, start, end)
                    if df is not None and not df.empty:
                        return self._clean_data(df)
        
        logger.warning(f"No data available for {symbol}")
        return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        - 去除重复数据
        - 处理缺失值
        - 异常值检测
        """
        if df.empty:
            return df
        
        # 去除重复
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 处理缺失值（使用前值填充）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # 处理volume缺失值（填充0）
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # 异常值检测（价格变动超过20%的数据标记）
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change()
            anomaly_mask = returns.abs() > 0.20
            if anomaly_mask.any():
                logger.warning(f"Found {anomaly_mask.sum()} potential anomalies in data")
        
        return df
    
    def get_buffer(self, symbol: str, data_type: str = "bar") -> Optional[DataBuffer]:
        """获取数据缓冲区"""
        if data_type == "bar":
            return self._bar_buffers.get(symbol)
        elif data_type == "tick":
            return self._tick_buffers.get(symbol)
        return None
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册数据回调"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    async def download_historical_data(
        self,
        symbols: List[str],
        interval: str,
        start: datetime,
        end: datetime,
        save_local: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下载历史数据
        
        Args:
            symbols: 合约代码列表
            interval: K线周期
            start: 开始时间
            end: 结束时间
            save_local: 是否保存到本地
        
        Returns:
            Dict[symbol, DataFrame]
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Downloading {symbol} {interval} data...")
            df = await self.get_bars(symbol, interval, start, end)
            
            if not df.empty:
                results[symbol] = df
                
                # 保存到本地
                if save_local and "local" in self._sources:
                    local_source: LocalDataSource = self._sources["local"]
                    local_source.save_bars(symbol, interval, df)
                
                logger.info(f"Downloaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"No data downloaded for {symbol}")
            
            # 避免请求过快
            await asyncio.sleep(0.5)
        
        return results


# ==================== 数据质量监控 ====================

class DataQualityMonitor:
    """
    数据质量监控
    实时监控数据延迟、完整性、异常
    """
    
    def __init__(self):
        self._last_tick_time: Dict[str, datetime] = {}
        self._latency_stats: Dict[str, List[float]] = {}
        self._missing_count: Dict[str, int] = {}
        self._anomaly_count: Dict[str, int] = {}
    
    def record_tick(self, symbol: str, tick: TickData):
        """记录Tick数据"""
        now = datetime.now()
        
        # 计算延迟
        data_time = tick.datetime
        latency = (now - data_time).total_seconds() * 1000  # 毫秒
        
        if symbol not in self._latency_stats:
            self._latency_stats[symbol] = []
        self._latency_stats[symbol].append(latency)
        
        # 保留最近1000条记录
        if len(self._latency_stats[symbol]) > 1000:
            self._latency_stats[symbol].pop(0)
        
        # 检查数据间隔
        if symbol in self._last_tick_time:
            gap = (data_time - self._last_tick_time[symbol]).total_seconds()
            if gap > 5:  # 超过5秒没有数据
                self._missing_count[symbol] = self._missing_count.get(symbol, 0) + 1
                logger.warning(f"Data gap detected for {symbol}: {gap:.1f}s")
        
        self._last_tick_time[symbol] = data_time
    
    def get_latency_stats(self, symbol: str) -> Dict[str, float]:
        """获取延迟统计"""
        if symbol not in self._latency_stats:
            return {}
        
        latencies = self._latency_stats[symbol]
        if not latencies:
            return {}
        
        return {
            "mean": np.mean(latencies),
            "max": max(latencies),
            "min": min(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        report = {}
        
        for symbol in self._last_tick_time.keys():
            latency_stats = self.get_latency_stats(symbol)
            report[symbol] = {
                "latency": latency_stats,
                "missing_count": self._missing_count.get(symbol, 0),
                "anomaly_count": self._anomaly_count.get(symbol, 0),
                "last_update": self._last_tick_time.get(symbol)
            }
        
        return report
