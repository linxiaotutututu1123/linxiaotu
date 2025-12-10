import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.strategy import DualMAStrategy
from src.risk.risk_manager import RiskManager, RiskManagerConfig
from src.execution.auto_trader import AutoTrader, AutoTraderConfig
from src.data.data_structures import BarData, OrderData, Direction, Offset, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PaperTradingSim")

class MockExecutionEngine:
    """Mock execution engine for paper trading simulation"""
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.balance = 1_000_000.0
        
    def send_order(self, symbol, direction, offset, price, volume, order_type, strategy_name):
        order_id = f"ORD_{len(self.orders)+1}"
        logger.info(f"📝 [MOCK EXEC] Order Received: {direction.value} {symbol} {volume}@{price:.2f} ({offset.value})")
        
        # Simulate immediate fill
        self.orders.append({
            'id': order_id,
            'symbol': symbol,
            'direction': direction,
            'price': price,
            'volume': volume,
            'status': 'FILLED'
        })
        
        # Update mock position
        pos_key = f"{symbol}_{direction.value}"
        if offset == Offset.OPEN:
            self.positions[pos_key] = self.positions.get(pos_key, 0) + volume
            self.balance -= price * volume * 10 * 0.1 # Margin
        else:
            # Simplified close
            pass
            
        return order_id

    def get_position(self, symbol, direction):
        # Return mock position object if needed
        return None

    def get_account(self):
        # Return mock account object
        from src.data.data_structures import AccountData
        return AccountData(
            account_id="MOCK_ACC",
            balance=self.balance,
            frozen=0,
            available=self.balance
        )
    
    def cancel_order(self, order_id):
        logger.info(f"❌ [MOCK EXEC] Cancel Order: {order_id}")
        return True

async def run_simulation():
    print('='*60)
    print('       量化交易系统 - 模拟交易流程 (Paper Trading)')
    print('='*60)

    # 1. Setup components
    risk_config = RiskManagerConfig(max_loss_per_trade=0.05)
    risk_manager = RiskManager(risk_config)
    
    exec_engine = MockExecutionEngine()
    
    trader_config = AutoTraderConfig(
        enabled=True,
        paper_trading=True,
        symbols=['rb9999']
    )
    
    # 2. Setup Strategy
    strategy = DualMAStrategy(fast_period=5, slow_period=10, symbols=['rb9999'])
    strategy.engine = exec_engine # Bind engine to strategy
    strategy.on_init()
    
    # 3. Setup AutoTrader
    auto_trader = AutoTrader(
        strategies=[strategy],
        risk_manager=risk_manager,
        execution_engine=exec_engine,
        config=trader_config
    )
    
    await auto_trader.start()
    
    # 4. Simulate Data Feed
    print("\n[模拟] 开始推送实时行情数据...")
    
    # Generate some data that triggers signals
    # Create a trend to trigger DualMA
    prices = [4000.0]
    for i in range(50):
        if i < 20:
            prices.append(prices[-1] * 0.995) # Down trend
        elif i < 40:
            prices.append(prices[-1] * 1.01)  # Up trend (Golden Cross)
        else:
            prices.append(prices[-1] * 0.99)  # Down trend (Death Cross)
            
    start_time = datetime.now()
    
    for i, price in enumerate(prices):
        # Construct BarData
        bar = BarData(
            symbol='rb9999',
            exchange='SHFE',
            datetime=start_time + timedelta(minutes=i),
            interval='1m',
            open=price,
            high=price+5,
            low=price-5,
            close=price,
            volume=100,
            turnover=price*100*10,
            open_interest=5000
        )
        
        # Push to strategy directly (simulating AutoTrader loop)
        # In real AutoTrader, it might subscribe to event bus. 
        # Here we manually call on_bar for demonstration.
        
        print(f"⏳ Tick {i}: Price={price:.2f}")
        
        # AutoTrader.on_bar checks trading time, we might need to mock that or bypass
        # For simplicity, we call strategy.on_bar directly to show logic, 
        # or we can try auto_trader.on_bar if we mock time.
        
        # Let's call strategy directly to ensure we see signals in this short script
        strategy.on_bar({'rb9999': bar})
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.05)

    await auto_trader.stop()
    print('\n' + '='*60)
    print('模拟交易结束')
    print(f"总挂单数: {len(exec_engine.orders)}")
    print('='*60)

if __name__ == "__main__":
    # Run async main
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_simulation())
