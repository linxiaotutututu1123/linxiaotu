# Initial Context for New Tasks

## Purpose
This file provides guidance for AI agents starting new tasks in this project. It ensures consistent workflows and prevents common mistakes.

## Before Starting Any Task

### 1. Read Project Context
- [ ] Read `.claude/CLAUDE.md` for project overview
- [ ] Understand the technology stack
- [ ] Review coding standards
- [ ] Note current constraints and requirements

### 2. Check Repository State
```bash
# Check current branch
git branch

# Check git status
git status

# Check for uncommitted changes
git diff

# Pull latest changes
git pull origin main
```

### 3. Understand the Task
- [ ] Read the task description carefully
- [ ] Identify acceptance criteria
- [ ] Understand dependencies and impacts
- [ ] Clarify any ambiguities before starting

### 4. Load Relevant Context
- [ ] Identify which context files are relevant
- [ ] Review API specifications if working with APIs
- [ ] Check security guidelines if handling sensitive data
- [ ] Review performance requirements if optimizing

## RPI Workflow

### Research Phase

#### Goals
- Gather all information needed to understand the task
- Identify existing patterns and conventions
- Understand potential impacts and dependencies

#### Activities
- [ ] **Code Analysis**
  - Search for similar implementations
  - Understand current architecture
  - Identify affected components
  
- [ ] **Dependency Mapping**
  - List direct dependencies
  - Identify indirect impacts
  - Note breaking change risks
  
- [ ] **Pattern Recognition**
  - Find existing patterns to follow
  - Identify anti-patterns to avoid
  - Note project-specific conventions

#### Research Checklist
- [ ] Current implementation understood
- [ ] Dependencies identified
- [ ] Similar patterns found
- [ ] Risks documented
- [ ] Questions answered

### Plan Phase

#### Goals
- Create detailed implementation plan
- Define validation criteria
- Establish testing strategy

#### Activities
- [ ] **Implementation Design**
  - Break task into smaller steps
  - Define interfaces and signatures
  - Plan data structures
  - Consider edge cases
  
- [ ] **Test Planning**
  - Identify test scenarios
  - Plan unit tests
  - Plan integration tests
  - Consider edge cases and errors
  
- [ ] **Validation Strategy**
  - Define success criteria
  - Plan validation steps
  - Identify verification methods

#### Planning Checklist
- [ ] Implementation steps defined
- [ ] Test cases identified
- [ ] Success criteria clear
- [ ] Risks mitigated
- [ ] Timeline estimated

### Implement Phase

#### Goals
- Write clean, tested code
- Follow project conventions
- Validate continuously

#### Activities
- [ ] **Test-Driven Development**
  - Write failing test first (Red)
  - Implement minimal code to pass (Green)
  - Refactor for quality (Refactor)
  
- [ ] **Incremental Development**
  - Implement one step at a time
  - Validate after each step
  - Commit working states
  
- [ ] **Continuous Validation**
  - Run tests frequently
  - Check linter output
  - Verify against requirements

#### Implementation Checklist
- [ ] Tests written first
- [ ] Implementation follows conventions
- [ ] All tests pass
- [ ] Code is clean and documented
- [ ] No linter errors

## Validation Guidelines

### Level 1: Syntax and Style
```bash
# Run linter
[project-specific lint command]

# Run formatter
[project-specific format command]

# Check types (if applicable)
[type check command]
```

**Checklist:**
- [ ] No syntax errors
- [ ] Passes linter
- [ ] Code formatted correctly
- [ ] Type annotations correct

### Level 2: Unit Tests
```bash
# Run unit tests
[project-specific test command]

# Run with coverage
[coverage command]
```

**Checklist:**
- [ ] All new functions have tests
- [ ] Tests cover edge cases
- [ ] Tests cover error cases
- [ ] All tests pass
- [ ] Coverage meets target

### Level 3: Integration Tests
```bash
# Run integration tests
[integration test command]
```

**Checklist:**
- [ ] Component interactions tested
- [ ] API contracts validated
- [ ] Database operations verified
- [ ] External dependencies mocked
- [ ] All integration tests pass

### Level 4: Security and Performance
```bash
# Run security checks
[security scan command]

# Run performance tests
[performance test command]
```

**Checklist:**
- [ ] No security vulnerabilities
- [ ] Input validation present
- [ ] Authentication/authorization correct
- [ ] Performance requirements met
- [ ] Resource usage acceptable

## Common Tasks

### Adding a New Feature

1. **Research**
   - Review existing similar features
   - Understand integration points
   - Check API documentation

2. **Plan**
   - Design data models
   - Define API endpoints
   - Plan test scenarios

3. **Implement**
   - Write model tests
   - Implement models
   - Write API tests
   - Implement API endpoints
   - Update documentation

### Fixing a Bug

1. **Research**
   - Reproduce the bug
   - Understand root cause
   - Check for related issues

2. **Plan**
   - Design fix approach
   - Plan regression tests
   - Consider side effects

3. **Implement**
   - Write failing test for bug
   - Implement fix
   - Verify test passes
   - Check for regressions

### Refactoring Code

1. **Research**
   - Understand current implementation
   - Identify improvement opportunities
   - Check dependencies

2. **Plan**
   - Design new structure
   - Plan migration strategy
   - Ensure backward compatibility

3. **Implement**
   - Add comprehensive tests
   - Refactor incrementally
   - Validate behavior unchanged
   - Update documentation

## Context Window Management

### Keep Context Focused
- Load only relevant context files
- Clear context when switching tasks
- Use subagents for specialized work

### Avoid Context Pollution
- Don't include irrelevant files
- Remove outdated information
- Keep focus on current task

### 40% Rule
Never fill more than 40% of context window with non-essential information.

## Best Practices

### Code Quality
- Write self-documenting code
- Use meaningful variable names
- Keep functions small and focused
- Follow single responsibility principle

### Testing
- Test behavior, not implementation
- Use descriptive test names
- Arrange-Act-Assert pattern
- One assertion per test (when possible)

### Documentation
- Update docs with code changes
- Document why, not what
- Include usage examples
- Keep README current

### Git Commits
- Commit early and often
- Write clear commit messages
- Keep commits atomic
- Reference issues/tickets

## Troubleshooting

### Tests Failing
1. Check test output carefully
2. Verify test setup/teardown
3. Check for race conditions
4. Verify mocks are correct

### Linter Errors
1. Read error message carefully
2. Check project conventions
3. Run auto-formatter
4. Fix one error at a time

### Build Failures
1. Check build logs
2. Verify dependencies installed
3. Check environment variables
4. Clear cache and rebuild

## Getting Help

### When Stuck
1. Review similar code in the project
2. Check project documentation
3. Search for related issues
4. Ask specific questions

### Before Asking
- Have you read the relevant docs?
- Have you searched existing issues?
- Can you provide a minimal reproduction?
- Have you tried debugging?

## Task Completion Checklist

Before considering a task complete:

- [ ] All requirements met
- [ ] All tests pass
- [ ] No linter errors
- [ ] Code reviewed (self-review)
- [ ] Documentation updated
- [ ] No security issues
- [ ] Performance acceptable
- [ ] Commit messages clear
- [ ] Changes validated manually

---

**Remember:** Quality over speed. It's better to take time to do it right than to rush and create problems.

**Last Updated:** [Date]
