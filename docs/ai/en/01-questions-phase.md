# Questions Phase

## 🎯 Purpose

The Questions Phase is the foundation of any development task. The goal is to achieve complete understanding of requirements before proceeding to design and implementation.

## 📋 Objectives

- Understand WHAT needs to be done
- Understand WHY it needs to be done
- Identify WHO will be affected
- Clarify WHERE changes need to be made
- Determine WHEN/HOW the solution will be used

## ✅ Checklist

### Requirement Analysis
- [ ] Read and understand the complete problem statement
- [ ] Identify the main goal of the task
- [ ] List all explicit requirements
- [ ] Identify implicit requirements
- [ ] Understand constraints and limitations

### Context Gathering
- [ ] Review related issues and pull requests
- [ ] Read relevant documentation
- [ ] Examine existing code in affected areas
- [ ] Check for similar implementations in the codebase
- [ ] Review repository conventions and patterns

### Stakeholder Understanding
- [ ] Identify who requested the change
- [ ] Understand the use case
- [ ] Determine success criteria
- [ ] Identify affected users/systems
- [ ] Consider downstream impacts

### Scope Definition
- [ ] List all files that may need changes
- [ ] Identify all components affected
- [ ] Determine what is IN scope
- [ ] Determine what is OUT of scope
- [ ] Identify dependencies on other systems

### Risk Assessment
- [ ] Identify potential breaking changes
- [ ] Consider backward compatibility
- [ ] Assess security implications
- [ ] Evaluate performance impacts
- [ ] Consider edge cases and failure modes

## 🔍 Key Questions to Ask

### Functional Questions
1. What is the expected behavior after this change?
2. What specific problem does this solve?
3. Are there any existing solutions we should leverage?
4. What are the acceptance criteria?
5. How will this be tested?

### Technical Questions
1. Which modules/components need to be modified?
2. Are there any API changes required?
3. Do we need database migrations?
4. Are there configuration changes needed?
5. What dependencies are required?

### Quality Questions
1. What tests need to be written?
2. How will we verify this works correctly?
3. What documentation needs updating?
4. Are there security concerns?
5. What are the performance requirements?

### Integration Questions
1. How does this integrate with existing code?
2. Are there any breaking changes?
3. Do other components depend on what we're changing?
4. Does this affect the public API?
5. Are there migration/upgrade paths needed?

## 🚨 Red Flags

Stop and ask for clarification if you encounter:
- ❌ Ambiguous or unclear requirements
- ❌ Conflicting requirements
- ❌ Requirements that seem too broad
- ❌ Missing information about affected systems
- ❌ Unclear success criteria
- ❌ Requests to bypass normal processes
- ❌ Changes that affect many unrelated areas

## 📊 Output

At the end of the Questions Phase, you should have:

1. **Requirements Document**
   - Clear list of what needs to be done
   - Explicit and implicit requirements
   - Constraints and limitations

2. **Scope Statement**
   - What is included
   - What is excluded
   - List of affected files/components

3. **Risk Assessment**
   - Potential issues identified
   - Mitigation strategies planned
   - Breaking changes documented

4. **Open Questions List**
   - Any remaining uncertainties
   - Items needing clarification
   - Assumptions that need validation

## ✨ Best Practices

### Do's
- ✅ Ask questions early and often
- ✅ Document your understanding
- ✅ Validate assumptions
- ✅ Review similar past changes
- ✅ Consider the full context

### Don'ts
- ❌ Rush to implementation
- ❌ Assume you understand without verification
- ❌ Skip reading related documentation
- ❌ Ignore edge cases
- ❌ Make assumptions without documenting them

## 🎓 Examples

### Good Questions Phase
```
Task: Add email validation to user registration

Questions Asked:
- What email format should be accepted? (RFC 5322 compliant)
- Should we check if email domain exists?
- What error message should be shown?
- Should this apply to existing users?
- Is this frontend, backend, or both?

Context Gathered:
- Reviewed existing validation in auth module
- Checked if email library is already available
- Reviewed user model schema
- Identified test file: tests/test_auth.py

Scope Defined:
IN: Backend validation, error messages, tests
OUT: Frontend changes (separate task)
```

### Poor Questions Phase
```
Task: Add email validation

Actions Taken:
- Started implementing immediately
- Added a regex check
- Committed without tests

Problems:
- Didn't understand full requirements
- Missed frontend validation needed
- No error handling
- Broke existing functionality
```

## 🔄 Transition to Design Phase

Exit Criteria:
- ✅ All requirements are clearly understood
- ✅ Scope is well-defined
- ✅ Context has been gathered
- ✅ No critical questions remain unanswered
- ✅ Risks have been identified

Once all exit criteria are met, proceed to: [02-design-phase.md](./02-design-phase.md)

---

**Remember**: Time spent in the Questions Phase prevents wasted effort in later phases. It's better to ask "obvious" questions than to make wrong assumptions.
