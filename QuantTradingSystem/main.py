"""
量化交易系统主入口
支持多种运行模式：回测、模拟交易、实盘交易
"""

import asyncio
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, load_settings
from src.data import DataManager
from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import (
    DualMAStrategy, TurtleStrategy, GridStrategy,
    MomentumStrategy, MultiStrategyPortfolio
)
from src.risk import RiskManager
from src.execution import ExecutionEngine, CTPGateway, TqsdkGateway, AutoTrader, AutoTraderConfig


# ==================== 日志配置 ====================

def setup_logging(level: str = "INFO", log_file: str = None):
    """配置日志"""
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 文件输出
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )


# ==================== 回测模式 ====================

@click.command()
@click.option('--config', '-c', default='config/backtest.yaml', help='配置文件路径')
@click.option('--strategy', '-s', default='dual_ma', help='策略名称')
@click.option('--symbol', default='rb2501', help='合约代码')
@click.option('--start', default='2023-01-01', help='开始日期')
@click.option('--end', default='2024-01-01', help='结束日期')
@click.option('--capital', default=1000000, help='初始资金')
def backtest(config, strategy, symbol, start, end, capital):
    """运行策略回测"""
    setup_logging("INFO")
    logger.info(f"Starting backtest: {strategy} on {symbol}")
    
    try:
        # 加载配置
        settings = load_settings(config) if Path(config).exists() else Settings()
        
        # 初始化数据管理器
        data_manager = DataManager(settings.data)
        
        # 加载数据
        logger.info(f"Loading data from {start} to {end}")
        df = data_manager.get_historical_data(
            symbol=symbol,
            start_date=start,
            end_date=end,
            frequency="1m"
        )
        
        if df is None or df.empty:
            logger.error("Failed to load data")
            return
        
        logger.info(f"Loaded {len(df)} bars")
        
        # 创建策略
        strategy_map = {
            'dual_ma': DualMAStrategy(name="DualMA", symbols=[symbol]),
            'turtle': TurtleStrategy(name="Turtle", symbols=[symbol]),
            'grid': GridStrategy(name="Grid", symbols=[symbol]),
            'momentum': MomentumStrategy(name="Momentum", symbols=[symbol]),
        }
        
        if strategy not in strategy_map:
            logger.error(f"Unknown strategy: {strategy}")
            return
        
        selected_strategy = strategy_map[strategy]
        
        # 创建回测引擎
        backtest_config = BacktestConfig(
            initial_capital=capital,
            commission_rate=0.0003,
            slippage=1.0,
            start_date=start,
            end_date=end
        )
        
        engine = BacktestEngine(backtest_config)
        
        # 运行回测
        logger.info("Running backtest...")
        result = engine.run(selected_strategy, df)
        
        # 输出结果
        logger.info("=" * 50)
        logger.info("回测结果 Backtest Results")
        logger.info("=" * 50)
        logger.info(f"初始资金: ¥{capital:,.2f}")
        logger.info(f"最终权益: ¥{result.final_equity:,.2f}")
        logger.info(f"总收益率: {result.total_return*100:.2f}%")
        logger.info(f"年化收益: {result.annual_return*100:.2f}%")
        logger.info(f"夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {result.max_drawdown*100:.2f}%")
        logger.info(f"胜率: {result.win_rate*100:.2f}%")
        logger.info(f"盈亏比: {result.profit_factor:.2f}")
        logger.info(f"总交易次数: {result.total_trades}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


# ==================== 模拟交易模式 ====================

@click.command()
@click.option('--config', '-c', default='config/settings.yaml', help='配置文件路径')
@click.option('--symbols', '-s', default='rb2501,i2501', help='交易品种(逗号分隔)')
def paper_trade(config, symbols):
    """运行模拟交易"""
    setup_logging("INFO", "logs/paper_trade.log")
    logger.info("Starting paper trading...")
    
    symbols_list = [s.strip() for s in symbols.split(',')]
    
    async def run():
        try:
            # 加载配置
            settings = load_settings(config) if Path(config).exists() else Settings()
            
            # 初始化组件
            data_manager = DataManager(settings.data)
            risk_manager = RiskManager(settings.risk)
            
            # 创建策略组合
            strategies = [
                DualMAStrategy(name="DualMA", symbols=symbols_list),
                MomentumStrategy(name="Momentum", symbols=symbols_list),
            ]
            
            portfolio = MultiStrategyPortfolio(
                name="Portfolio",
                symbols=symbols_list,
                strategies=strategies
            )
            
            # 创建模拟交易网关
            gateway = TqsdkGateway()
            
            # 创建执行引擎
            execution_engine = ExecutionEngine(gateway, settings.execution)
            
            # 创建自动交易器
            trader_config = AutoTraderConfig(
                enabled=True,
                paper_trading=True,
                symbols=symbols_list
            )
            
            auto_trader = AutoTrader(
                strategies=[portfolio],
                risk_manager=risk_manager,
                execution_engine=execution_engine,
                config=trader_config
            )
            
            # 启动
            await execution_engine.start()
            await auto_trader.start()
            
            logger.info(f"Paper trading started for: {symbols_list}")
            
            # 主循环
            while True:
                # 获取行情数据
                for symbol in symbols_list:
                    bar = data_manager.get_latest_bar(symbol)
                    if bar:
                        await auto_trader.on_bar(bar)
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
        finally:
            await auto_trader.stop()
            await execution_engine.stop()
    
    # 运行
    asyncio.run(run())


# ==================== 实盘交易模式 ====================

@click.command()
@click.option('--config', '-c', default='config/live.yaml', help='配置文件路径')
@click.option('--broker', '-b', default='ctp', help='交易通道(ctp/tqsdk)')
def live_trade(config, broker):
    """运行实盘交易"""
    setup_logging("INFO", "logs/live_trade.log")
    logger.warning("=" * 50)
    logger.warning("警告: 即将启动实盘交易!")
    logger.warning("Warning: Starting LIVE trading!")
    logger.warning("=" * 50)
    
    # 确认
    confirm = input("确认启动实盘交易? (输入 'YES' 确认): ")
    if confirm != 'YES':
        logger.info("已取消")
        return
    
    async def run():
        try:
            # 加载配置
            settings = load_settings(config)
            
            # 创建交易网关
            if broker == 'ctp':
                gateway = CTPGateway(
                    broker_id=settings.ctp.broker_id,
                    user_id=settings.ctp.user_id,
                    password=settings.ctp.password,
                    td_address=settings.ctp.td_address,
                    md_address=settings.ctp.md_address
                )
            else:
                gateway = TqsdkGateway(
                    account=settings.tqsdk.account,
                    password=settings.tqsdk.password
                )
            
            # 初始化组件
            data_manager = DataManager(settings.data)
            risk_manager = RiskManager(settings.risk)
            
            # 创建策略
            strategies = [
                DualMAStrategy(name="DualMA", symbols=settings.trading.symbols),
            ]
            
            # 创建执行引擎
            execution_engine = ExecutionEngine(gateway, settings.execution)
            
            # 创建自动交易器
            trader_config = AutoTraderConfig(
                enabled=True,
                paper_trading=False,  # 实盘!
                symbols=settings.trading.symbols
            )
            
            auto_trader = AutoTrader(
                strategies=strategies,
                risk_manager=risk_manager,
                execution_engine=execution_engine,
                config=trader_config
            )
            
            # 启动
            await execution_engine.start()
            await auto_trader.start()
            
            logger.info("Live trading started")
            
            # 优雅关闭
            def signal_handler(sig, frame):
                logger.info("Received shutdown signal")
                asyncio.create_task(auto_trader.stop())
                asyncio.create_task(execution_engine.stop())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # 主循环
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            raise
        finally:
            await auto_trader.stop()
            await execution_engine.stop()
    
    asyncio.run(run())


# ==================== 监控服务 ====================

@click.command()
@click.option('--host', default='0.0.0.0', help='监听地址')
@click.option('--port', default=8000, help='监听端口')
def monitor(host, port):
    """启动监控Web服务"""
    setup_logging("INFO")
    logger.info(f"Starting monitor server on {host}:{port}")
    
    import uvicorn
    from frontend.api_server import app
    
    uvicorn.run(app, host=host, port=port)


# ==================== CLI入口 ====================

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    量化交易系统 v1.0.0
    
    Quantitative Trading System
    
    Commands:
    
      backtest     运行策略回测
      
      paper        运行模拟交易
      
      live         运行实盘交易
      
      monitor      启动监控服务
    """
    pass


cli.add_command(backtest)
cli.add_command(paper_trade, name='paper')
cli.add_command(live_trade, name='live')
cli.add_command(monitor)


if __name__ == '__main__':
    cli()
