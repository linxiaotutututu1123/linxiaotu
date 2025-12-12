# Implementation Phase

## 🎯 Purpose

The Implementation Phase is where code changes are made based on the design and with tests already in place. The goal is to make minimal, surgical changes that pass all tests and meet requirements.

## 📋 Objectives

- Implement the designed solution
- Make minimal necessary changes
- Follow existing code patterns
- Ensure all tests pass
- Maintain code quality

## ✅ Checklist

### Pre-Implementation
- [ ] Design is complete and approved
- [ ] Tests are written and failing (Red)
- [ ] Environment is set up correctly
- [ ] Dependencies are installed
- [ ] Current tests pass (baseline)

### During Implementation
- [ ] Follow the design document
- [ ] Make changes in planned order
- [ ] Write minimal code to pass tests
- [ ] Run tests frequently
- [ ] Commit small, logical changes

### Code Quality
- [ ] Follow coding standards (PEP 8)
- [ ] Use type hints
- [ ] Add docstrings
- [ ] Handle errors appropriately
- [ ] Add logging where needed

### Post-Implementation
- [ ] All tests pass (Green)
- [ ] Code is linted
- [ ] No debug code remains
- [ ] No commented-out code
- [ ] Documentation is updated

## 🏗️ Implementation Principles

### Minimal Change Principle
- Change only what's necessary
- Don't refactor unrelated code
- Keep diffs small and focused
- One concern per commit

### Test-Driven Principle
- Run tests after each change
- Keep tests passing (Green)
- If tests break, fix immediately
- Don't skip failing tests

### Consistency Principle
- Match existing code style
- Use existing patterns
- Follow naming conventions
- Maintain code structure

### Quality Principle
- Write clean, readable code
- Handle errors gracefully
- Add appropriate logging
- Document complex logic

## 📝 Implementation Workflow

### 1. Start with Simplest Case
```python
# Start with the happy path
def validate_email(email: str) -> bool:
    """Validate email format"""
    if '@' not in email:
        return False
    return True
```

### 2. Run Tests (Should Fail)
```bash
pytest tests/test_validators.py::test_validate_email -v
```

### 3. Implement to Pass Tests
```python
import re

def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid")
        False
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

### 4. Run Tests (Should Pass)
```bash
pytest tests/test_validators.py -v
```

### 5. Refactor if Needed
```python
# Extract regex pattern as constant
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

def validate_email(email: str) -> bool:
    """Validate email format using RFC-compliant regex"""
    return bool(email and re.match(EMAIL_PATTERN, email))
```

### 6. Run Tests Again
```bash
pytest tests/test_validators.py -v
```

## 🎯 Code Quality Standards

### Python Code Style (PEP 8)
```python
# Good: Clear naming, type hints, docstring
def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio for returns
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default: 0.02)
        
    Returns:
        Sharpe ratio
        
    Raises:
        ValueError: If returns list is empty
    """
    if not returns:
        raise ValueError("Returns list cannot be empty")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return
```

```python
# Bad: Poor naming, no types, no docstring
def calc(r, rfr=0.02):
    if not r:
        raise ValueError("empty")
    m = np.mean(r)
    s = np.std(r)
    if s == 0:
        return 0
    return (m - rfr) / s
```

### Error Handling
```python
# Good: Specific exceptions, clear messages
def get_user(user_id: int) -> User:
    """Get user by ID"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching user {user_id}: {e}")
        raise DatabaseError("Failed to fetch user") from e
```

```python
# Bad: Generic exceptions, unclear errors
def get_user(user_id):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except:
        raise Exception("Error")
```

### Logging
```python
# Good: Structured logging with context
import logging

logger = logging.getLogger(__name__)

def process_trade(trade: Trade) -> None:
    """Process a trade order"""
    logger.info(
        "Processing trade",
        extra={
            "symbol": trade.symbol,
            "quantity": trade.quantity,
            "price": trade.price
        }
    )
    
    try:
        result = execute_trade(trade)
        logger.info(f"Trade executed successfully: {result.trade_id}")
    except TradeError as e:
        logger.error(
            f"Trade execution failed: {e}",
            extra={"trade": trade.dict()},
            exc_info=True
        )
        raise
```

```python
# Bad: Unclear logging, no context
def process_trade(trade):
    print("Processing...")
    try:
        result = execute_trade(trade)
        print("Done")
    except:
        print("Error!")
```

## 🎯 Project-Specific Guidelines

### For Quantitative Trading System

#### Strategy Implementation
```python
from strategies.base_strategy import BaseStrategy
from typing import Dict

class MyStrategy(BaseStrategy):
    """
    Custom trading strategy
    
    Attributes:
        config: Strategy configuration
        logger: Logger instance
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.6)
    
    def generate_signals(self) -> Dict[str, int]:
        """
        Generate trading signals
        
        Returns:
            Dict mapping symbol to signal (-1, 0, 1)
        """
        signals = {}
        
        for symbol in self.symbols:
            score = self._calculate_score(symbol)
            
            if score > self.threshold:
                signals[symbol] = 1  # Buy
            elif score < (1 - self.threshold):
                signals[symbol] = -1  # Sell
            else:
                signals[symbol] = 0  # Hold
        
        return signals
    
    def _calculate_score(self, symbol: str) -> float:
        """Calculate signal score for symbol"""
        # Implementation here
        pass
```

#### Risk Management Integration
```python
from core.risk_manager import RiskManager

def execute_trade_with_risk_check(trade: Trade, risk_manager: RiskManager) -> None:
    """
    Execute trade with risk management checks
    
    Args:
        trade: Trade to execute
        risk_manager: Risk manager instance
        
    Raises:
        RiskLimitError: If trade violates risk limits
    """
    # Check position limits
    if not risk_manager.can_open_position(trade.quantity):
        raise RiskLimitError("Position limit exceeded")
    
    # Check risk limits
    if not risk_manager.check_risk_limits(trade):
        raise RiskLimitError("Risk limits violated")
    
    # Execute trade
    result = trade_executor.execute(trade)
    
    # Update risk manager
    risk_manager.update_position(trade.symbol, trade.quantity)
    
    logger.info(f"Trade executed: {result}")
```

## 🚨 Common Implementation Mistakes

Avoid these pitfalls:

### Don't Skip Tests
```python
# Bad: Implementing without running tests
def new_feature():
    # Code...
    pass
# TODO: Write tests later (never happens)
```

### Don't Over-Implement
```python
# Bad: Implementing features not in requirements
def validate_email(email: str) -> bool:
    # Validate format
    # Also check if domain exists (not required!)
    # Also check if mailbox exists (not required!)
    # Also integrate with spam checker (not required!)
    pass
```

### Don't Leave Debug Code
```python
# Bad: Debug code in production
def calculate(x):
    print(f"DEBUG: x = {x}")  # Remove this!
    # import pdb; pdb.set_trace()  # Remove this!
    result = x * 2
    print(f"DEBUG: result = {result}")  # Remove this!
    return result
```

### Don't Ignore Errors
```python
# Bad: Catching and ignoring errors
try:
    critical_operation()
except:
    pass  # Silently failing!
```

## 📊 Code Review Checklist

Before committing, verify:

- [ ] All tests pass
- [ ] Code is linted (no PEP 8 violations)
- [ ] Type hints are used
- [ ] Docstrings are complete
- [ ] Error handling is appropriate
- [ ] Logging is adequate
- [ ] No debug code remains
- [ ] No TODOs remain
- [ ] No commented-out code
- [ ] Configuration is externalized
- [ ] Security best practices followed

## ✨ Best Practices

### Do's
- ✅ Make small, incremental changes
- ✅ Run tests after each change
- ✅ Commit working code frequently
- ✅ Follow existing patterns
- ✅ Write self-documenting code
- ✅ Handle errors explicitly
- ✅ Add logging for important events
- ✅ Use type hints
- ✅ Keep functions small and focused

### Don'ts
- ❌ Make large, sweeping changes
- ❌ Skip running tests
- ❌ Commit broken code
- ❌ Introduce new patterns unnecessarily
- ❌ Write overly clever code
- ❌ Ignore error cases
- ❌ Skip logging
- ❌ Use magic numbers
- ❌ Create god functions

## 🎓 Examples

### Good Implementation
```python
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Execute trades with proper error handling and logging
    """
    
    def __init__(self, broker_api, risk_manager):
        self.broker = broker_api
        self.risk_manager = risk_manager
        self.logger = logger
    
    def execute_trade(self, trade: Trade) -> Optional[str]:
        """
        Execute a trade order
        
        Args:
            trade: Trade object to execute
            
        Returns:
            Trade ID if successful, None otherwise
            
        Raises:
            RiskLimitError: If risk limits violated
            BrokerError: If broker API fails
        """
        # Validate risk limits
        if not self.risk_manager.validate_trade(trade):
            raise RiskLimitError(f"Trade violates risk limits: {trade}")
        
        # Execute via broker
        try:
            trade_id = self.broker.submit_order(
                symbol=trade.symbol,
                quantity=trade.quantity,
                price=trade.price,
                direction=trade.direction
            )
            
            self.logger.info(
                f"Trade executed successfully: {trade_id}",
                extra={"trade": trade.dict()}
            )
            
            return trade_id
            
        except BrokerAPIError as e:
            self.logger.error(
                f"Broker API error: {e}",
                extra={"trade": trade.dict()},
                exc_info=True
            )
            raise BrokerError("Failed to execute trade") from e
```

### Poor Implementation
```python
# Bad: No types, error handling, logging, or documentation
class TradeExecutor:
    def execute(self, t):
        try:
            result = self.broker.order(t.s, t.q, t.p, t.d)
            return result
        except:
            return None
```

## 🔄 Transition to Verification Phase

Exit Criteria:
- ✅ All planned code changes are complete
- ✅ All tests pass
- ✅ Code is linted and follows standards
- ✅ Error handling is in place
- ✅ Logging is adequate
- ✅ Documentation is updated
- ✅ No debug code remains

Once all exit criteria are met, proceed to: [06-verification-phase.md](./06-verification-phase.md)

---

**Remember**: Good implementation is not just about making tests pass—it's about writing maintainable, high-quality code that others can understand and extend.
