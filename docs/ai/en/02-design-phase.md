# Design Phase

## 🎯 Purpose

The Design Phase translates requirements into a concrete plan for implementation. The goal is to determine HOW to achieve the objectives with minimal, surgical changes.

## 📋 Objectives

- Design the solution approach
- Plan minimal necessary changes
- Identify patterns to follow
- Document the design decisions
- Get alignment before coding

## ✅ Checklist

### Solution Design
- [ ] Identify the simplest solution that meets requirements
- [ ] Consider multiple approaches
- [ ] Evaluate trade-offs of each approach
- [ ] Choose the best approach with justification
- [ ] Document why alternatives were rejected

### Architecture Review
- [ ] Understand existing architecture
- [ ] Identify patterns used in similar code
- [ ] Ensure solution fits existing structure
- [ ] Avoid introducing new patterns unnecessarily
- [ ] Consider modularity and reusability

### Change Planning
- [ ] List exact files to be modified
- [ ] List exact files to be created
- [ ] Identify functions/classes to change
- [ ] Plan the order of changes
- [ ] Minimize the scope of changes

### Integration Planning
- [ ] How will new code integrate with existing code?
- [ ] What interfaces need to be defined?
- [ ] What dependencies are required?
- [ ] How will backward compatibility be maintained?
- [ ] What configuration changes are needed?

### Error Handling
- [ ] Identify potential failure points
- [ ] Plan error handling strategy
- [ ] Define error messages
- [ ] Plan logging approach
- [ ] Consider recovery mechanisms

## 🏗️ Design Principles

### Minimal Change Principle
- Change as few lines as possible
- Reuse existing code where possible
- Don't refactor unrelated code
- Keep changes focused and surgical
- Avoid scope creep

### Consistency Principle
- Follow existing code patterns
- Match existing naming conventions
- Use same libraries as existing code
- Maintain consistent style
- Don't introduce new paradigms

### Safety Principle
- Don't break existing functionality
- Maintain backward compatibility
- Plan rollback strategy
- Consider failure modes
- Add defensive checks

### Quality Principle
- Design for testability
- Design for maintainability
- Design for readability
- Design for performance
- Design for security

## 📝 Design Document Template

```markdown
## Solution Overview
Brief description of the approach

## Changes Required

### Files to Modify
1. path/to/file1.py - Add email validation
2. path/to/file2.py - Update error handling

### Files to Create
1. path/to/new_file.py - Email validator utility

### Files to Delete
None (avoid deletions unless absolutely necessary)

## Detailed Design

### Component 1: Email Validator
- Location: utils/validators.py
- Function: validate_email(email: str) -> bool
- Logic: Use regex + domain check
- Error handling: Raise ValueError with message

### Component 2: Integration
- Location: core/auth.py
- Changes: Add validation call in register()
- Error handling: Catch ValueError, return 400

## Dependencies
- No new dependencies (use existing 're' module)

## Configuration
- No configuration changes needed

## Backward Compatibility
- Fully backward compatible
- Existing code paths unchanged

## Testing Strategy
- Unit tests for validate_email()
- Integration test for registration flow
- Edge cases: empty, malformed, special chars

## Risks and Mitigations
- Risk: Regex may be too strict
- Mitigation: Use well-tested regex pattern
```

## 🎯 Design Patterns to Consider

### For Validation
- Use existing validation utilities
- Follow validator pattern if it exists
- Centralize validation logic
- Return clear error messages

### For Data Processing
- Use existing data handler patterns
- Follow ETL patterns if applicable
- Use existing libraries (pandas, numpy)
- Keep processing functions pure

### For API Changes
- Maintain existing signatures if possible
- Use optional parameters for new features
- Provide migration path if breaking
- Document API changes clearly

### For Database Changes
- Use existing ORM patterns
- Plan migrations carefully
- Test with sample data
- Provide rollback migrations

## 🚨 Design Anti-Patterns

Avoid these common mistakes:

- ❌ Over-engineering the solution
- ❌ Introducing new frameworks/libraries unnecessarily
- ❌ Changing unrelated code "while we're at it"
- ❌ Designing without understanding existing patterns
- ❌ Planning changes that are too broad
- ❌ Ignoring existing abstractions
- ❌ Creating new patterns when existing ones work
- ❌ Planning changes without considering tests

## 📊 Output

At the end of the Design Phase, you should have:

1. **Design Document**
   - Clear description of the approach
   - List of all changes needed
   - Justification for design decisions

2. **Change Plan**
   - Exact files to modify/create
   - Order of implementation
   - Expected diff size

3. **Integration Plan**
   - How new code connects to existing
   - What interfaces are used
   - What dependencies are needed

4. **Risk Mitigation Plan**
   - Identified risks
   - Mitigation strategies
   - Rollback plan

## ✨ Best Practices

### Do's
- ✅ Plan before you code
- ✅ Keep it simple (KISS principle)
- ✅ Follow existing patterns
- ✅ Design for testability
- ✅ Consider edge cases
- ✅ Document design decisions

### Don'ts
- ❌ Start coding before design is complete
- ❌ Over-complicate the solution
- ❌ Introduce new patterns unnecessarily
- ❌ Plan changes that are too broad
- ❌ Ignore existing architecture
- ❌ Skip documentation

## 🎓 Examples

### Good Design
```
Task: Add rate limiting to API endpoint

Design:
1. Use existing @rate_limit decorator pattern
2. Modify: api/endpoints/users.py
   - Add @rate_limit(calls=100, period=60) to get_user()
3. Configure: config/settings.yaml
   - Add rate_limit section with defaults
4. Tests: tests/test_rate_limit.py
   - Test normal usage
   - Test limit exceeded
   - Test reset after period

Justification:
- Reuses existing decorator pattern
- Minimal code changes (2 lines)
- Centralizes configuration
- Easy to test and maintain
```

### Poor Design
```
Task: Add rate limiting to API endpoint

Design:
1. Install new library 'advanced-rate-limiter'
2. Refactor all endpoints to use new pattern
3. Create new middleware system
4. Rewrite configuration system
5. Add Redis caching layer

Problems:
- Introduces unnecessary complexity
- Changes unrelated code
- Adds new dependencies
- Breaks existing patterns
- Scope too large
```

## 🔄 Transition to Models Phase

Exit Criteria:
- ✅ Solution approach is clearly defined
- ✅ All changes are planned and documented
- ✅ Design follows existing patterns
- ✅ Risks are identified and mitigated
- ✅ Design is minimal and focused

Once all exit criteria are met, proceed to: [03-models-phase.md](./03-models-phase.md)

---

**Remember**: A good design makes implementation straightforward. If implementation feels complicated, revisit the design.
