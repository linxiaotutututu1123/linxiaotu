"""Main entry point for the quantitative trading system.

Provides command-line interface for backtesting, simulation, and live trading.
"""

import argparse
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_handler import DataHandler
from core.backtest_engine import BacktestEngine
from core.risk_manager import RiskManager
from strategies.factor_strategy import FactorStrategy
from strategies.grid_strategy import GridStrategy
from strategies.combined_strategy import CombinedStrategy
from utils.logger import setup_logger, get_logger


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Quantitative Futures Trading System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'simulate', 'live'],
        default='backtest',
        help='Trading mode: backtest, simulate, or live'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['factor', 'grid', 'combined'],
        default='combined',
        help='Trading strategy: factor, grid, or combined'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Trading symbols (e.g., RB IF IO)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def run_backtest(args, logger):
    """Run backtesting mode.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("Starting Backtest Mode")
    logger.info("=" * 60)
    
    # Initialize components
    data_handler = DataHandler(args.config)
    backtest_engine = BacktestEngine(args.config)
    risk_manager = RiskManager(args.config)
    
    # Select strategy
    if args.strategy == 'factor':
        strategy = FactorStrategy(args.config)
    elif args.strategy == 'grid':
        strategy = GridStrategy(args.config)
    else:
        strategy = CombinedStrategy(args.config)
    
    logger.info(f"Strategy: {strategy.name}")
    
    # Get symbols
    symbols = args.symbols or data_handler.get_symbols_list()
    logger.info(f"Trading symbols: {symbols}")
    
    # Fetch data
    start_date = args.start_date or "2023-01-01"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        df = data_handler.fetch_data(symbol, start_date, end_date, frequency='1d')
        data[symbol] = df
        logger.info(f"  Loaded {len(df)} bars")
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtest_engine.run_backtest(strategy, data, start_date, end_date)
    
    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Initial Capital:  {results['initial_capital']:,.0f}")
    logger.info(f"Final Capital:    {results['final_capital']:,.0f}")
    logger.info(f"Total Return:     {results['total_return']:.2%}")
    logger.info(f"Annual Return:    {results['annual_return']:.2%}")
    logger.info(f"Max Drawdown:     {results['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    logger.info(f"Total Trades:     {results['total_trades']}")
    logger.info(f"Win Rate:         {results['win_rate']:.2%}")
    logger.info("=" * 60)
    
    # Check if targets are met
    logger.info("")
    logger.info("Target Achievement:")
    
    target_return = 0.25  # 25% annual return target for Stage 1
    target_drawdown = 0.20  # 20% max drawdown target for Stage 1
    
    if results['annual_return'] >= target_return:
        logger.info(f"✓ Annual return target MET ({results['annual_return']:.2%} >= {target_return:.2%})")
    else:
        logger.warning(f"✗ Annual return target NOT MET ({results['annual_return']:.2%} < {target_return:.2%})")
    
    if results['max_drawdown'] <= target_drawdown:
        logger.info(f"✓ Max drawdown target MET ({results['max_drawdown']:.2%} <= {target_drawdown:.2%})")
    else:
        logger.warning(f"✗ Max drawdown target NOT MET ({results['max_drawdown']:.2%} > {target_drawdown:.2%})")
    
    logger.info("")


def run_simulation(args, logger):
    """Run simulation mode.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("Simulation Mode - NOT YET IMPLEMENTED")
    logger.info("=" * 60)
    logger.info("This mode will simulate live trading with paper trading account.")
    logger.info("Implementation pending in Stage 1 completion.")


def run_live(args, logger):
    """Run live trading mode.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("Live Trading Mode - NOT YET IMPLEMENTED")
    logger.info("=" * 60)
    logger.warning("⚠️  WARNING: Live trading involves real money!")
    logger.info("This mode will execute real trades on your brokerage account.")
    logger.info("Implementation pending in Stage 1 completion.")
    logger.info("")
    logger.info("Before enabling live trading:")
    logger.info("1. Complete extensive backtesting")
    logger.info("2. Run paper trading for at least 2-4 weeks")
    logger.info("3. Test with small capital for 1-2 weeks")
    logger.info("4. Ensure all risk controls are working properly")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(
        'QuantTradingSystem',
        log_level=args.log_level,
        log_to_console=True,
        log_to_file=True,
        log_dir='./logs'
    )
    
    logger.info(f"Quantitative Futures Trading System")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Strategy: {args.strategy}")
    
    try:
        if args.mode == 'backtest':
            run_backtest(args, logger)
        elif args.mode == 'simulate':
            run_simulation(args, logger)
        elif args.mode == 'live':
            run_live(args, logger)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return 1
    
    logger.info("Program completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
