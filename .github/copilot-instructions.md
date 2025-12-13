# Copilot Instructions for linxiaotu Repository

## Repository Overview

This repository is a personal collection of various projects and resources, primarily focused on:

1. **Quantitative Trading System** (`quant-trading-system/`) - A professional futures quantitative trading system built with Python
2. **AI Engineering Hub** (`ai-engineering-hub-main/`) - AI-related projects including podcast generation and paralegal agent crews
3. **Reference Resources** - Collections of useful tools, prompts, and models

## Repository Structure

```
linxiaotu/
├── quant-trading-system/        # Python-based futures trading system
│   ├── core/                    # Core modules (data, backtest, trade execution, risk)
│   ├── models/                  # Trading models (multi-factor, grid, arbitrage, DQN)
│   ├── strategies/              # Trading strategies
│   ├── utils/                   # Utility functions and helpers
│   ├── tests/                   # Test suite
│   └── config/                  # Configuration files
├── ai-engineering-hub-main/     # AI projects and experiments
├── awesome-mac-master/          # Mac productivity resources
├── northstar-master/            # Reference materials
└── system-prompts-and-models-of-ai-tools-main/  # AI tools and prompts collection
```

## Coding Conventions

### General Guidelines

- **Language Priority**: Python is the primary language for quantitative trading projects; JavaScript/TypeScript for any web components
- **Code Style**: Follow PEP 8 for Python code; use descriptive variable names
- **Comments**: Add docstrings for all functions and classes; inline comments for complex logic
- **File Organization**: Keep related functionality together; use `__init__.py` for module organization

### Python Specific

- Use Python 3.8+ features
- Prefer type hints for function signatures
- Use f-strings for string formatting
- Follow PEP 8 naming conventions:
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- Use list comprehensions and generator expressions where appropriate
- Handle exceptions explicitly; avoid bare `except:` clauses

### Documentation

- All public APIs should have docstrings following Google style
- Update README.md files when adding new features or changing functionality
- Document complex algorithms and trading strategies with clear explanations
- Include usage examples in docstrings

## Build and Test Instructions

### For Quantitative Trading System

**Setup:**
```bash
cd quant-trading-system
pip install -r requirements.txt
```

**Testing:**
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_strategies.py

# Run with coverage
python -m pytest --cov=core --cov=models --cov=strategies tests/
```

**Running the System:**
```bash
# Backtest mode
python main.py --mode backtest --strategy factor_grid

# Simulate mode (paper trading)
python main.py --mode simulate --strategy factor_grid

# Live trading (use with caution)
python main.py --mode live --strategy factor_grid
```

### For JavaScript/Node.js Projects

**Testing:**
```bash
npm test
```

**Installing Dependencies:**
```bash
npm install
```

## Contribution Guidelines

### When Making Changes

1. **Minimal Changes**: Make the smallest possible changes to accomplish the task
2. **Test Coverage**: Add or update tests for any new functionality
3. **Documentation**: Update relevant README files and docstrings
4. **Configuration**: Don't commit sensitive data (API keys, credentials) - use environment variables or config files
5. **Dependencies**: Only add new dependencies if absolutely necessary; document the reason

### Code Review Checklist

- [ ] Code follows existing style and conventions
- [ ] All tests pass
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] No sensitive data is committed
- [ ] Changes are focused and minimal

### For Trading System Changes

- **Risk Management**: Any changes to risk management or trading logic must be thoroughly tested in backtest mode first
- **Data Handling**: Ensure data integrity and proper error handling
- **Performance**: Consider performance implications, especially for real-time trading components
- **Logging**: Add appropriate logging for debugging and monitoring

## Important Notes

### Security Considerations

- Never commit API keys, passwords, or trading credentials
- Use environment variables or secure configuration management
- Be cautious with trading logic changes - always test thoroughly in simulation first

### Testing Requirements

- All trading strategies must pass backtests before deployment
- Test edge cases and error conditions
- Verify risk management rules are enforced
- Test with various market conditions (trending, ranging, volatile)

### Dependencies Management

- Keep dependencies up to date for security patches
- Document any specific version requirements
- Use virtual environments for Python projects

## Resources and References

- **Quantitative Trading**: The system uses Backtrader for backtesting, integrates with CTP/TqSdk/VNPY for trading
- **Machine Learning**: TensorFlow/PyTorch for reinforcement learning models (Phase 3)
- **Data Storage**: PostgreSQL for persistent storage, Redis for caching

## Common Tasks

### Adding a New Trading Strategy

1. Create strategy file in `quant-trading-system/strategies/`
2. Inherit from `base_strategy.py`
3. Implement required methods: `__init__`, `next`, `notify_order`
4. Add unit tests in `tests/test_strategies.py`
5. Add backtest configuration
6. Document strategy logic and parameters

### Adding New Indicators

1. Add indicator functions to `utils/indicators.py`
2. Follow existing pattern for parameter handling
3. Include docstrings with formula and usage
4. Add unit tests
5. Update documentation

### Fixing Bugs

1. Write a failing test that reproduces the bug
2. Fix the issue with minimal changes
3. Verify all tests pass
4. Update documentation if behavior changes

## Contact and Support

For questions or issues, please create an issue in the repository.

---

**Last Updated**: 2025-12-10
