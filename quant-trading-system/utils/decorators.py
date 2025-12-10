"""Decorator utilities for the quantitative trading system.

Provides timing, exception handling, and other utility decorators.
"""

import time
import functools
from typing import Callable, Any
from .logger import get_logger


logger = get_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper


def exception_handler(func: Callable) -> Callable:
    """Decorator to handle exceptions and log errors.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper
