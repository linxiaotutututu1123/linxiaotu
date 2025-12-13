# AI Development Rules and Guidelines

## 🎯 Purpose

This document establishes the mandatory workflow and best practices for AI-assisted development in this repository. All AI agents, developers, and contributors must follow these rules to ensure consistent, high-quality, and maintainable code.

## 📋 Core Principles

1. **Structured Workflow**: Follow the defined Task Workflow for all development tasks
2. **Quality First**: Prioritize code quality, testing, and documentation
3. **Minimal Changes**: Make the smallest possible changes to achieve the goal
4. **Test-Driven**: Write tests before or alongside implementation
5. **Documentation**: Keep documentation synchronized with code changes
6. **Security**: Always validate changes for security vulnerabilities

## 🔄 Task Workflow

All development tasks MUST follow this strict workflow sequence:

### 1️⃣ Questions Phase
- Understand the requirements completely
- Ask clarifying questions before starting
- Identify all affected components
- Review related code and documentation
- **Exit Criteria**: Clear understanding of what needs to be done

### 2️⃣ Design Phase
- Plan the minimal changes needed
- Identify files and functions to modify
- Consider edge cases and error handling
- Document the approach
- **Exit Criteria**: Approved design with clear implementation path

### 3️⃣ Models Phase
- Define or update data models
- Create or update schemas
- Plan database migrations if needed
- Document model relationships
- **Exit Criteria**: All data structures defined and validated

### 4️⃣ Tests Phase
- Write or update test cases FIRST
- Ensure test coverage for new code
- Include edge cases and error conditions
- Validate tests fail before implementation (TDD)
- **Exit Criteria**: Complete test suite ready for implementation

### 5️⃣ Implementation Phase
- Make minimal, surgical changes
- Follow existing code patterns
- Add inline comments only when necessary
- Maintain code consistency
- **Exit Criteria**: Code changes complete and linted

### 6️⃣ Verification Phase
- Run all relevant tests
- Perform manual testing if applicable
- Validate no regressions introduced
- Check for security vulnerabilities
- Review performance implications
- **Exit Criteria**: All tests pass, no new issues introduced

## 📚 Execution Library

For detailed guidance on each phase, consult the execution library:

- **Questions Phase**: [docs/ai/en/01-questions-phase.md](./ai/en/01-questions-phase.md)
- **Design Phase**: [docs/ai/en/02-design-phase.md](./ai/en/02-design-phase.md)
- **Models Phase**: [docs/ai/en/03-models-phase.md](./ai/en/03-models-phase.md)
- **Tests Phase**: [docs/ai/en/04-tests-phase.md](./ai/en/04-tests-phase.md)
- **Implementation Phase**: [docs/ai/en/05-implementation-phase.md](./ai/en/05-implementation-phase.md)
- **Verification Phase**: [docs/ai/en/06-verification-phase.md](./ai/en/06-verification-phase.md)

## 🛡️ Mandatory Rules

### Code Quality
- ✅ Follow PEP 8 for Python code
- ✅ Use type hints in Python code
- ✅ Add docstrings to all functions and classes
- ✅ Keep functions small and focused (single responsibility)
- ✅ Use meaningful variable and function names

### Testing Requirements
- ✅ Write tests for all new functionality
- ✅ Maintain or improve test coverage
- ✅ Use pytest for Python tests
- ✅ Include unit tests and integration tests where applicable
- ✅ Test edge cases and error conditions

### Security
- ✅ Never commit secrets or credentials
- ✅ Validate all user inputs
- ✅ Use parameterized queries for databases
- ✅ Sanitize data before display
- ✅ Run CodeQL security scans before finalizing

### Documentation
- ✅ Update README if functionality changes
- ✅ Add inline comments for complex logic
- ✅ Document all configuration options
- ✅ Keep API documentation synchronized
- ✅ Update changelog for significant changes

### Version Control
- ✅ Make small, focused commits
- ✅ Write clear commit messages
- ✅ Use feature branches for development
- ✅ Never force push to protected branches
- ✅ Keep commits atomic and logical

## 🚫 Prohibited Actions

- ❌ Skip any phase of the Task Workflow
- ❌ Make changes without tests
- ❌ Commit code that doesn't pass tests
- ❌ Ignore security vulnerabilities
- ❌ Remove or modify working code unnecessarily
- ❌ Add dependencies without justification
- ❌ Commit build artifacts or dependencies (node_modules, .coverage, etc.)
- ❌ Make changes outside the scope of the task
- ❌ Ignore existing code patterns and conventions

## 🎯 Project-Specific Rules

### Quantitative Trading System

For changes to the `quant-trading-system/` directory:

- All strategies must inherit from `BaseStrategy` (see `strategies/base_strategy.py`)
- All strategies must implement `generate_signals()` method
- New strategies require backtest validation before deployment
- Risk limits are defined in `config/settings.yaml`
- Use pytest for all testing
- Follow the modular architecture: core/, models/, strategies/, utils/

### Configuration Management
- All parameters must be in `config/settings.yaml`
- Never hardcode configuration values
- Document all new configuration options
- Validate configuration on startup

### Logging
- Use the project's logger utility
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Include context in log messages
- Never log sensitive information

## 📊 Success Metrics

Measure success by:
- ✅ All tests passing
- ✅ No security vulnerabilities introduced
- ✅ Code coverage maintained or improved
- ✅ Documentation is complete and accurate
- ✅ Code follows project conventions
- ✅ Changes are minimal and focused
- ✅ No regressions in existing functionality

## 🔄 Continuous Improvement

This document should be:
- Reviewed and updated regularly
- Enhanced based on lessons learned
- Kept synchronized with project evolution
- Used as a reference for code reviews

## 📞 Questions and Feedback

If you have questions about these rules or suggestions for improvement:
1. Review the execution library for detailed guidance
2. Check existing code for patterns and examples
3. Ask clarifying questions before proceeding
4. Document learnings for future reference

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-12  
**Status**: Active and Mandatory
