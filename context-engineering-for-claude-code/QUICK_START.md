# Context Engineering Quick Start Guide

## 5-Minute Setup

### Step 1: Create Directory Structure (1 min)

```bash
# Navigate to your project root
cd /path/to/your/project

# Create context engineering directory structure
mkdir -p .claude/agents
mkdir -p .claude/context

# Create main context files
touch .claude/CLAUDE.md
touch .claude/INITIAL.md
```

### Step 2: Fill Basic Project Information (2 min)

Copy and customize the template to `.claude/CLAUDE.md`:

```markdown
# [Your Project Name]

## Technology Stack
- Language: [e.g., Python 3.11]
- Framework: [e.g., FastAPI]
- Database: [e.g., PostgreSQL]

## Project Structure
```
src/
├── api/
├── models/
└── services/
```

## Development Commands
```bash
# Install
pip install -r requirements.txt

# Test
pytest

# Run
python main.py
```

## Coding Standards
- Follow PEP 8
- Use type hints
- Write docstrings
```

### Step 3: Create Initial Workflow Guide (2 min)

Add to `.claude/INITIAL.md`:

```markdown
# Task Workflow

## Before Starting
1. Read CLAUDE.md
2. Check git status
3. Understand the task

## RPI Process
1. **Research**: Analyze existing code
2. **Plan**: Design implementation
3. **Implement**: Write tests first, then code

## Validation
- [ ] Tests pass
- [ ] Linter passes
- [ ] Code reviewed
```

## Using Context Engineering

### For New Features

1. **Research Phase**
```bash
# Search for similar features
grep -r "similar_feature" src/

# Understand dependencies
git log --all -- src/related_module/
```

2. **Plan Phase**
```markdown
## Feature: User Authentication

### Requirements
- OAuth 2.0 support
- JWT tokens
- Role-based access

### Implementation Steps
1. Add user model
2. Create auth endpoints
3. Add middleware
4. Write tests
```

3. **Implement Phase**
```bash
# Write test first
# Implement feature
# Run tests
pytest tests/test_auth.py

# Validate
black . && flake8 .
```

### For Bug Fixes

1. **Research**
   - Reproduce the bug
   - Find root cause
   - Check for similar issues

2. **Plan**
   - Design fix
   - Plan regression test
   - Consider side effects

3. **Implement**
   - Write failing test
   - Fix bug
   - Verify test passes

## Advanced: Using Subagents

### Create a Code Reviewer Subagent

```yaml
# .claude/agents/reviewer.yaml
name: Code Reviewer
description: Reviews code quality and security
system_prompt: |
  Review code for:
  - Quality and maintainability
  - Security vulnerabilities
  - Best practices
  - Test coverage
context_files:
  - .claude/CLAUDE.md
  - .claude/context/security-guidelines.md
```

### Using the Subagent

When you need a code review:
1. Invoke the reviewer subagent
2. Provide the code to review
3. Address the feedback
4. Re-review if significant changes made

## Best Practices

### ✅ Do's

- **Keep context files updated**: Review monthly
- **Use specific context**: Load only relevant files
- **Write clear instructions**: Be explicit about requirements
- **Validate frequently**: Test after each change
- **Document decisions**: Record why, not just what

### ❌ Don'ts

- **Don't overload context**: Follow the 40% rule
- **Don't skip research**: Understand before coding
- **Don't ignore tests**: Write tests first
- **Don't hardcode secrets**: Use environment variables
- **Don't assume**: Verify with documentation

## Common Workflows

### Adding an API Endpoint

```markdown
1. Research
   - Check existing endpoints
   - Review API conventions
   - Understand data models

2. Plan
   - Define endpoint: POST /api/users
   - Design request/response
   - Plan validation
   - Plan tests

3. Implement
   - Write endpoint tests
   - Implement endpoint
   - Add validation
   - Update docs
```

### Refactoring Code

```markdown
1. Research
   - Understand current code
   - Find all usages
   - Check dependencies

2. Plan
   - Design new structure
   - Plan migration
   - Ensure compatibility

3. Implement
   - Add comprehensive tests
   - Refactor incrementally
   - Validate behavior unchanged
   - Update documentation
```

## Troubleshooting

### Context Not Helping?

**Problem**: AI making wrong assumptions

**Solutions**:
- Add more specific context about the area
- Create a focused context file for this domain
- Use a specialized subagent
- Provide examples of correct patterns

### AI Suggestions Not Following Standards?

**Problem**: Code doesn't match project conventions

**Solutions**:
- Update CLAUDE.md with clear coding standards
- Add examples of correct patterns
- Create a linting/formatting section
- Reference specific style guides

### Performance Issues?

**Problem**: AI responses slow or low quality

**Solutions**:
- Reduce context file sizes
- Remove outdated information
- Use more focused context
- Clear context window between tasks

## Examples

### Example 1: Adding Authentication

**Context Files Needed**:
- `.claude/CLAUDE.md` - Project overview
- `.claude/context/security-guidelines.md` - Security requirements
- `.claude/context/api-specs.md` - API patterns

**Workflow**:
1. Research existing auth patterns
2. Plan OAuth 2.0 implementation
3. Implement with tests first
4. Validate security with security subagent

### Example 2: Database Migration

**Context Files Needed**:
- `.claude/CLAUDE.md` - Project overview
- `.claude/context/database-schema.md` - Current schema
- `.claude/context/migration-strategy.md` - Migration process

**Workflow**:
1. Research schema changes needed
2. Plan migration with rollback
3. Implement migration script
4. Test on staging first

## Next Steps

1. **Complete Setup**: Fill in all sections of CLAUDE.md
2. **Add Context Files**: Create domain-specific context files
3. **Create Subagents**: Set up specialized subagents for your needs
4. **Iterate**: Improve context based on usage
5. **Share**: Collaborate with team on context improvements

## Resources

- [Full README](./README.md) - Comprehensive guide
- [Templates](./templates/) - Ready-to-use templates
- [Claude Code Docs](https://code.claude.com/docs)
- [GitHub Examples](https://github.com/coleam00/context-engineering-intro)

## Checklist for Success

- [ ] Created .claude/ directory structure
- [ ] Filled out CLAUDE.md with project info
- [ ] Created INITIAL.md workflow guide
- [ ] Added project-specific context files
- [ ] Set up at least one subagent
- [ ] Tested workflow with a small task
- [ ] Shared with team for feedback
- [ ] Documented lessons learned

---

**Remember**: Context Engineering is iterative. Start simple, then improve based on what works for your team.

**Time Investment**: 
- Initial setup: 5-10 minutes
- First iteration: 30 minutes
- Monthly maintenance: 15 minutes
- **ROI**: 10-50x improvement in AI assistance quality
