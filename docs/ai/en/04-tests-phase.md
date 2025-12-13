# Tests Phase

## 🎯 Purpose

The Tests Phase focuses on creating comprehensive tests BEFORE or alongside implementation. This ensures code quality, prevents regressions, and validates that requirements are met.

## 📋 Objectives

- Write tests before implementation (TDD)
- Ensure comprehensive test coverage
- Test edge cases and error conditions
- Create maintainable and clear tests
- Validate tests fail before implementation

## ✅ Checklist

### Test Planning
- [ ] Identify what needs to be tested
- [ ] List test cases (happy path)
- [ ] List test cases (error conditions)
- [ ] List test cases (edge cases)
- [ ] Determine test data needs

### Test Structure
- [ ] Identify existing test files to update
- [ ] Identify new test files needed
- [ ] Follow existing test patterns
- [ ] Plan test fixtures/mocks
- [ ] Plan test data setup

### Coverage Planning
- [ ] Ensure unit tests for all functions
- [ ] Plan integration tests if needed
- [ ] Plan end-to-end tests if applicable
- [ ] Target minimum coverage (aim for >80%)
- [ ] Identify untestable code (minimize)

### Test Quality
- [ ] Tests are independent
- [ ] Tests are repeatable
- [ ] Tests are fast
- [ ] Tests are clear and readable
- [ ] Tests follow AAA pattern (Arrange-Act-Assert)

## 🧪 Test Types

### Unit Tests
Test individual functions/methods in isolation:
```python
def test_validate_email():
    """Test email validation with valid email"""
    assert validate_email("user@example.com") == True

def test_validate_email_invalid():
    """Test email validation with invalid email"""
    with pytest.raises(ValueError):
        validate_email("invalid-email")
```

### Integration Tests
Test interaction between components:
```python
def test_user_registration_flow():
    """Test complete user registration process"""
    user_data = {"email": "test@example.com", "password": "secure123"}
    user = create_user(user_data)
    assert user.email == "test@example.com"
    assert user.is_active == True
```

### Functional Tests
Test business logic and workflows:
```python
def test_trading_strategy_generates_signals():
    """Test that strategy generates correct signals"""
    strategy = FactorStrategy(config)
    signals = strategy.generate_signals()
    assert signals["RB"] in [-1, 0, 1]
```

### Edge Case Tests
Test boundary conditions:
```python
def test_position_limit_at_maximum():
    """Test behavior when at position limit"""
    manager = RiskManager(max_position=100)
    assert manager.can_open_position(100) == False
    assert manager.can_open_position(99) == True
```

## 📝 Test Template

```python
"""
Test module for user authentication
"""
import pytest
from unittest.mock import Mock, patch
from myapp.auth import authenticate_user, UserNotFoundError

class TestUserAuthentication:
    """Test cases for user authentication"""
    
    @pytest.fixture
    def sample_user(self):
        """Fixture providing sample user data"""
        return {
            "email": "test@example.com",
            "password": "secure123",
            "id": 1
        }
    
    def test_authenticate_valid_user(self, sample_user):
        """
        Test authentication with valid credentials
        
        Given: A registered user with valid credentials
        When: authenticate_user is called
        Then: User object is returned
        """
        # Arrange
        email = sample_user["email"]
        password = sample_user["password"]
        
        # Act
        result = authenticate_user(email, password)
        
        # Assert
        assert result is not None
        assert result.email == email
    
    def test_authenticate_invalid_password(self, sample_user):
        """
        Test authentication with invalid password
        
        Given: A registered user
        When: authenticate_user is called with wrong password
        Then: Returns None
        """
        # Arrange
        email = sample_user["email"]
        wrong_password = "wrongpass"
        
        # Act
        result = authenticate_user(email, wrong_password)
        
        # Assert
        assert result is None
    
    def test_authenticate_nonexistent_user(self):
        """
        Test authentication with non-existent user
        
        Given: An email not in the system
        When: authenticate_user is called
        Then: Raises UserNotFoundError
        """
        # Arrange
        email = "nonexistent@example.com"
        password = "anypassword"
        
        # Act & Assert
        with pytest.raises(UserNotFoundError):
            authenticate_user(email, password)
    
    def test_authenticate_empty_email(self):
        """Test authentication with empty email"""
        with pytest.raises(ValueError, match="Email is required"):
            authenticate_user("", "password123")
    
    @patch('myapp.auth.database')
    def test_authenticate_database_error(self, mock_db, sample_user):
        """Test authentication when database fails"""
        # Arrange
        mock_db.query.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(Exception):
            authenticate_user(sample_user["email"], sample_user["password"])
```

## 🎯 Test-Driven Development (TDD)

### TDD Workflow
1. **Write Test**: Write a test for the new functionality
2. **Run Test**: Verify the test fails (Red)
3. **Write Code**: Write minimal code to pass the test
4. **Run Test**: Verify the test passes (Green)
5. **Refactor**: Improve code while keeping tests green
6. **Repeat**: Continue for next functionality

### TDD Benefits
- ✅ Forces clear requirements
- ✅ Ensures code is testable
- ✅ Provides immediate feedback
- ✅ Creates safety net for refactoring
- ✅ Documents expected behavior

## 🏗️ Test Organization

### File Structure
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_auth.py             # Authentication tests
├── test_models.py           # Model tests
├── test_strategies.py       # Strategy tests
└── integration/
    ├── __init__.py
    └── test_backtest.py     # Integration tests
```

### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Be descriptive: `test_user_registration_with_invalid_email`

## 🎯 Project-Specific Guidelines

### For Quantitative Trading System

#### Strategy Tests
```python
def test_factor_strategy_generates_signals():
    """Test that FactorStrategy generates valid signals"""
    strategy = FactorStrategy(config={})
    signals = strategy.generate_signals()
    
    assert isinstance(signals, dict)
    for symbol, signal in signals.items():
        assert signal in [-1, 0, 1]
        assert isinstance(symbol, str)
```

#### Backtest Tests
```python
def test_backtest_engine_calculates_metrics():
    """Test that BacktestEngine calculates performance metrics"""
    engine = BacktestEngine(strategy, data)
    results = engine.run()
    
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert results["sharpe_ratio"] > 0
```

#### Risk Management Tests
```python
def test_risk_manager_enforces_position_limit():
    """Test that RiskManager enforces maximum position"""
    manager = RiskManager(max_position=100)
    
    assert manager.can_open_position(50) == True
    manager.open_position(50)
    assert manager.can_open_position(51) == False
    assert manager.can_open_position(50) == True
```

### Using pytest Fixtures
```python
@pytest.fixture
def sample_strategy_config():
    """Provide sample strategy configuration"""
    return {
        "multi_factor": {
            "weights": {
                "technical": 0.30,
                "fund_flow": 0.25,
                "sentiment": 0.25,
                "correlation": 0.15,
                "anomaly": 0.05
            }
        }
    }

@pytest.fixture
def mock_market_data():
    """Provide mock market data for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100),
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
```

## 🚨 Common Testing Mistakes

Avoid these pitfalls:

- ❌ Writing tests after implementation
- ❌ Not testing error conditions
- ❌ Tests that depend on each other
- ❌ Tests that depend on external state
- ❌ Tests that are too complex
- ❌ Not using fixtures for common setup
- ❌ Testing implementation details instead of behavior
- ❌ Not mocking external dependencies
- ❌ Ignoring test failures
- ❌ Tests without clear assertions

## 📊 Test Coverage

### Measuring Coverage
```bash
# Run tests with coverage
pytest --cov=myapp tests/

# Generate HTML report
pytest --cov=myapp --cov-report=html tests/
```

### Coverage Goals
- **Minimum**: 80% coverage
- **Target**: 90%+ coverage
- **Critical Code**: 100% coverage (auth, risk management, etc.)

### What to Cover
- ✅ All public functions/methods
- ✅ All error handling paths
- ✅ All edge cases
- ✅ All business logic
- ✅ Critical workflows

### What Not to Cover
- Configuration files
- Third-party library code
- Generated code
- Trivial getters/setters

## ✨ Best Practices

### Do's
- ✅ Write tests first (TDD)
- ✅ Keep tests simple and focused
- ✅ Use descriptive test names
- ✅ Test one thing per test
- ✅ Use fixtures for common setup
- ✅ Mock external dependencies
- ✅ Test edge cases
- ✅ Test error conditions
- ✅ Run tests frequently

### Don'ts
- ❌ Skip writing tests
- ❌ Write tests after implementation
- ❌ Test implementation details
- ❌ Write dependent tests
- ❌ Ignore failing tests
- ❌ Write overly complex tests
- ❌ Test third-party code
- ❌ Hardcode test data
- ❌ Skip edge cases

## 📊 Output

At the end of the Tests Phase, you should have:

1. **Test Files**
   - All test files created
   - Tests written for new functionality
   - Tests updated for modified functionality

2. **Test Coverage**
   - Coverage report generated
   - Coverage meets minimum threshold
   - Critical paths fully covered

3. **Test Documentation**
   - Clear test names
   - Docstrings explaining what is tested
   - Test data documented

4. **Passing Tests**
   - All tests run successfully
   - No flaky tests
   - Tests are fast and reliable

## 🎓 Examples

### Good Test
```python
def test_validate_email_with_valid_format():
    """
    Test email validation accepts valid email format
    
    Given: A valid email address
    When: validate_email is called
    Then: Returns True without raising exceptions
    """
    valid_emails = [
        "user@example.com",
        "test.user@example.co.uk",
        "user+tag@example.com"
    ]
    
    for email in valid_emails:
        assert validate_email(email) == True
```

### Poor Test
```python
def test_stuff():
    """Test things"""
    x = do_something()
    assert x  # What are we testing? Why?
```

## 🔄 Transition to Implementation Phase

Exit Criteria:
- ✅ All test cases are written
- ✅ Tests follow project conventions
- ✅ Tests currently fail (Red in TDD)
- ✅ Test coverage plan is complete
- ✅ Fixtures and mocks are ready

Once all exit criteria are met, proceed to: [05-implementation-phase.md](./05-implementation-phase.md)

---

**Remember**: Good tests are your safety net. They give you confidence to refactor and prevent regressions.
