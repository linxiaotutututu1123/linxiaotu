"""
Web监控界面 - FastAPI后端
提供REST API和WebSocket实时数据推送
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ==================== API数据模型 ====================

class OrderRequest(BaseModel):
    """下单请求"""
    symbol: str
    direction: str  # "long" / "short"
    volume: float
    price: float = None
    order_type: str = "limit"  # "limit" / "market"
    algo: str = None  # "twap" / "vwap" / "iceberg" / None


class OrderResponse(BaseModel):
    """订单响应"""
    success: bool
    order_id: str = ""
    message: str = ""


class PositionInfo(BaseModel):
    """持仓信息"""
    symbol: str
    direction: str
    volume: float
    avg_price: float
    current_price: float
    pnl: float
    pnl_ratio: float


class AccountInfo(BaseModel):
    """账户信息"""
    balance: float
    available: float
    frozen: float
    total_pnl: float
    total_pnl_ratio: float


class StrategyInfo(BaseModel):
    """策略信息"""
    name: str
    enabled: bool
    pnl: float
    win_rate: float
    trade_count: int


class SystemStatus(BaseModel):
    """系统状态"""
    running: bool
    paper_trading: bool
    connected: bool
    strategies_count: int
    active_orders: int
    open_positions: int
    uptime: str


# ==================== WebSocket连接管理 ====================

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._subscriptions: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected, total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # 移除订阅
        for channel in self._subscriptions.values():
            if websocket in channel:
                channel.remove(websocket)
        logger.info(f"WebSocket disconnected, total: {len(self.active_connections)}")
    
    def subscribe(self, websocket: WebSocket, channel: str):
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        if websocket not in self._subscriptions[channel]:
            self._subscriptions[channel].append(websocket)
    
    def unsubscribe(self, websocket: WebSocket, channel: str):
        if channel in self._subscriptions:
            if websocket in self._subscriptions[channel]:
                self._subscriptions[channel].remove(websocket)
    
    async def broadcast(self, message: dict, channel: str = None):
        """广播消息"""
        targets = self._subscriptions.get(channel, []) if channel else self.active_connections
        
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """发送给特定连接"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Send error: {e}")


# ==================== 监控服务 ====================

class MonitorService:
    """监控服务 - 模拟数据"""
    
    def __init__(self):
        self._start_time = datetime.now()
        self._positions: Dict[str, Dict] = {}
        self._orders: List[Dict] = []
        self._strategies: Dict[str, Dict] = {}
        self._account: Dict = {
            'balance': 1000000,
            'available': 800000,
            'frozen': 200000
        }
    
    def get_system_status(self) -> SystemStatus:
        uptime = datetime.now() - self._start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return SystemStatus(
            running=True,
            paper_trading=True,
            connected=True,
            strategies_count=len(self._strategies),
            active_orders=len([o for o in self._orders if o.get('status') == 'active']),
            open_positions=len(self._positions),
            uptime=f"{hours}h {minutes}m {seconds}s"
        )
    
    def get_account_info(self) -> AccountInfo:
        total_pnl = sum(p.get('pnl', 0) for p in self._positions.values())
        return AccountInfo(
            balance=self._account['balance'],
            available=self._account['available'],
            frozen=self._account['frozen'],
            total_pnl=total_pnl,
            total_pnl_ratio=total_pnl / self._account['balance'] * 100
        )
    
    def get_positions(self) -> List[PositionInfo]:
        return [
            PositionInfo(
                symbol=symbol,
                direction=pos['direction'],
                volume=pos['volume'],
                avg_price=pos['avg_price'],
                current_price=pos.get('current_price', pos['avg_price']),
                pnl=pos.get('pnl', 0),
                pnl_ratio=pos.get('pnl_ratio', 0)
            )
            for symbol, pos in self._positions.items()
        ]
    
    def get_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                name=name,
                enabled=info.get('enabled', True),
                pnl=info.get('pnl', 0),
                win_rate=info.get('win_rate', 0),
                trade_count=info.get('trade_count', 0)
            )
            for name, info in self._strategies.items()
        ]
    
    def submit_order(self, request: OrderRequest) -> OrderResponse:
        order_id = f"ORD_{int(datetime.now().timestamp() * 1000)}"
        self._orders.append({
            'order_id': order_id,
            'symbol': request.symbol,
            'direction': request.direction,
            'volume': request.volume,
            'price': request.price,
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        })
        return OrderResponse(success=True, order_id=order_id, message="Order submitted")
    
    def cancel_order(self, order_id: str) -> bool:
        for order in self._orders:
            if order['order_id'] == order_id:
                order['status'] = 'cancelled'
                return True
        return False


# ==================== FastAPI应用 ====================

def create_app(monitor_service: MonitorService = None) -> FastAPI:
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="量化交易监控系统",
        description="期货量化交易系统监控API",
        version="1.0.0"
    )
    
    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 服务实例
    service = monitor_service or MonitorService()
    manager = ConnectionManager()
    
    # ==================== REST API ====================
    
    @app.get("/api/status", response_model=SystemStatus)
    async def get_status():
        """获取系统状态"""
        return service.get_system_status()
    
    @app.get("/api/account", response_model=AccountInfo)
    async def get_account():
        """获取账户信息"""
        return service.get_account_info()
    
    @app.get("/api/positions", response_model=List[PositionInfo])
    async def get_positions():
        """获取持仓列表"""
        return service.get_positions()
    
    @app.get("/api/strategies", response_model=List[StrategyInfo])
    async def get_strategies():
        """获取策略列表"""
        return service.get_strategies()
    
    @app.get("/api/orders")
    async def get_orders(status: str = Query(None, description="active/filled/cancelled")):
        """获取订单列表"""
        orders = service._orders
        if status:
            orders = [o for o in orders if o['status'] == status]
        return orders
    
    @app.post("/api/orders", response_model=OrderResponse)
    async def submit_order(request: OrderRequest):
        """提交订单"""
        return service.submit_order(request)
    
    @app.delete("/api/orders/{order_id}")
    async def cancel_order(order_id: str):
        """取消订单"""
        success = service.cancel_order(order_id)
        if success:
            return {"success": True, "message": "Order cancelled"}
        raise HTTPException(status_code=404, detail="Order not found")
    
    @app.post("/api/strategies/{name}/enable")
    async def enable_strategy(name: str):
        """启用策略"""
        if name in service._strategies:
            service._strategies[name]['enabled'] = True
            return {"success": True}
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    @app.post("/api/strategies/{name}/disable")
    async def disable_strategy(name: str):
        """禁用策略"""
        if name in service._strategies:
            service._strategies[name]['enabled'] = False
            return {"success": True}
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # ==================== WebSocket ====================
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket实时数据"""
        await manager.connect(websocket)
        
        try:
            while True:
                data = await websocket.receive_json()
                
                # 处理订阅请求
                if data.get('action') == 'subscribe':
                    channel = data.get('channel')
                    if channel:
                        manager.subscribe(websocket, channel)
                        await manager.send_personal(websocket, {
                            'type': 'subscribed',
                            'channel': channel
                        })
                
                elif data.get('action') == 'unsubscribe':
                    channel = data.get('channel')
                    if channel:
                        manager.unsubscribe(websocket, channel)
                
                elif data.get('action') == 'ping':
                    await manager.send_personal(websocket, {'type': 'pong'})
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
    
    # ==================== 定时推送任务 ====================
    
    async def push_updates():
        """定时推送更新"""
        while True:
            try:
                # 推送系统状态
                status = service.get_system_status()
                await manager.broadcast({
                    'type': 'status',
                    'data': status.dict()
                }, channel='status')
                
                # 推送账户信息
                account = service.get_account_info()
                await manager.broadcast({
                    'type': 'account',
                    'data': account.dict()
                }, channel='account')
                
                # 推送持仓信息
                positions = service.get_positions()
                await manager.broadcast({
                    'type': 'positions',
                    'data': [p.dict() for p in positions]
                }, channel='positions')
                
            except Exception as e:
                logger.error(f"Push update error: {e}")
            
            await asyncio.sleep(1)
    
    @app.on_event("startup")
    async def startup():
        asyncio.create_task(push_updates())
    
    return app


# ==================== 主程序 ====================

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
