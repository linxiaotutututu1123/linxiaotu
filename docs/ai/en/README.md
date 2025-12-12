# AI Development Execution Library

This directory contains detailed guidance for each phase of the AI Development Workflow. These documents provide practical, actionable guidance for implementing changes following the Task Workflow defined in [AI_RULES_EN.md](../../AI_RULES_EN.md).

## 📚 Contents

### Workflow Phases

1. **[Questions Phase](./01-questions-phase.md)** - Requirements gathering and understanding
   - Understand WHAT, WHY, WHO, WHERE, WHEN
   - Gather context and identify scope
   - Risk assessment and clarifications

2. **[Design Phase](./02-design-phase.md)** - Solution planning and architecture
   - Design minimal solution approach
   - Plan exact changes needed
   - Document design decisions

3. **[Models Phase](./03-models-phase.md)** - Data structures and schemas
   - Define or update data models
   - Plan database changes
   - Document validation rules

4. **[Tests Phase](./04-tests-phase.md)** - Test planning and creation
   - Write tests BEFORE implementation (TDD)
   - Ensure comprehensive coverage
   - Test edge cases and errors

5. **[Implementation Phase](./05-implementation-phase.md)** - Code changes
   - Make minimal, surgical changes
   - Follow existing patterns
   - Keep tests passing

6. **[Verification Phase](./06-verification-phase.md)** - Testing and validation
   - Verify all tests pass
   - Manual testing and security checks
   - Performance validation

## 🎯 How to Use This Library

### For Each Task

1. **Start with Questions Phase**
   - Read [01-questions-phase.md](./01-questions-phase.md)
   - Follow the checklist
   - Document your understanding
   - Don't proceed until questions are answered

2. **Progress Through Phases**
   - Complete each phase in order
   - Don't skip phases
   - Meet exit criteria before moving on
   - Document decisions at each phase

3. **Reference as Needed**
   - Consult relevant phase document when stuck
   - Use templates and examples provided
   - Follow best practices and avoid anti-patterns
   - Apply project-specific guidelines

### Quick Reference

**Starting a new task?**
→ Start at [01-questions-phase.md](./01-questions-phase.md)

**Ready to design solution?**
→ Go to [02-design-phase.md](./02-design-phase.md)

**Need to define data models?**
→ Check [03-models-phase.md](./03-models-phase.md)

**Time to write tests?**
→ Follow [04-tests-phase.md](./04-tests-phase.md)

**Ready to code?**
→ Implement with [05-implementation-phase.md](./05-implementation-phase.md)

**Ready to finalize?**
→ Verify with [06-verification-phase.md](./06-verification-phase.md)

## ✅ Phase Transition Checklist

Use this to track your progress through phases:

```markdown
Task: [Brief task description]

- [ ] Questions Phase Complete
  - [ ] Requirements understood
  - [ ] Scope defined
  - [ ] Risks identified
  
- [ ] Design Phase Complete
  - [ ] Solution designed
  - [ ] Changes planned
  - [ ] Patterns identified
  
- [ ] Models Phase Complete
  - [ ] Data models defined
  - [ ] Validation rules set
  - [ ] Migrations planned
  
- [ ] Tests Phase Complete
  - [ ] Tests written
  - [ ] Coverage adequate
  - [ ] Tests failing (Red)
  
- [ ] Implementation Phase Complete
  - [ ] Code implemented
  - [ ] Tests passing (Green)
  - [ ] Code linted
  
- [ ] Verification Phase Complete
  - [ ] All tests pass
  - [ ] Manual testing done
  - [ ] Security verified
  - [ ] Documentation updated
```

## 🎓 Learning Path

### For New Contributors

1. **Read the Main Rules**
   - Start with [AI_RULES_EN.md](../../AI_RULES_EN.md)
   - Understand core principles
   - Review mandatory rules

2. **Study the Workflow**
   - Read through all phase documents
   - Understand the flow
   - Note the exit criteria

3. **Review Examples**
   - Look at "Good" vs "Poor" examples
   - Understand why certain approaches are better
   - Learn common pitfalls

4. **Practice on Small Tasks**
   - Start with simple bug fixes
   - Follow the workflow strictly
   - Get feedback on your approach

### For Experienced Developers

- Use as a checklist reference
- Focus on project-specific sections
- Pay attention to anti-patterns
- Help improve the documentation

## 🔄 Workflow Overview

```
┌─────────────────┐
│  1. Questions   │  Understand requirements
│     Phase       │  Define scope, ask questions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Design      │  Plan solution approach
│     Phase       │  Document changes needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Models      │  Define data structures
│     Phase       │  Plan migrations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Tests       │  Write tests FIRST (TDD)
│     Phase       │  Ensure coverage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Implement   │  Write minimal code
│     Phase       │  Make tests pass
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Verify      │  Run all tests
│     Phase       │  Security & performance check
└────────┬────────┘
         │
         ▼
      ✅ Done!
```

## 🎯 Key Principles

Remember these throughout all phases:

1. **Minimal Changes**: Change only what's necessary
2. **Test-Driven**: Write tests before code
3. **Follow Patterns**: Use existing code patterns
4. **Quality First**: Don't compromise on quality
5. **Document Everything**: Keep docs synchronized
6. **Security Always**: Check for vulnerabilities
7. **Ask Questions**: Better to ask than assume

## 📊 Success Metrics

Your work is successful when:

- ✅ All phases completed in order
- ✅ All tests pass
- ✅ No security vulnerabilities
- ✅ Code follows conventions
- ✅ Documentation is complete
- ✅ Changes are minimal and focused
- ✅ No regressions introduced

## 🚨 Red Flags

Stop and reassess if:

- ❌ Skipping phases
- ❌ Making changes without tests
- ❌ Changing unrelated code
- ❌ Ignoring security issues
- ❌ Tests are failing
- ❌ Requirements are unclear
- ❌ Changes are too broad

## 🔗 Related Resources

- [Main AI Rules](../../AI_RULES_EN.md) - Core rules and principles
- [Project README](../../../README.md) - Project overview
- [Quantitative Trading System](../../../quant-trading-system/README.md) - Trading system docs

## 💡 Tips for Success

### Do's
- ✅ Read phase documents before starting
- ✅ Follow checklists systematically
- ✅ Document your decisions
- ✅ Ask questions early
- ✅ Test frequently
- ✅ Commit small changes
- ✅ Get code reviews

### Don'ts
- ❌ Rush through phases
- ❌ Skip documentation
- ❌ Ignore best practices
- ❌ Make assumptions
- ❌ Skip testing
- ❌ Commit broken code
- ❌ Work in isolation

## 📞 Getting Help

If you're stuck or unsure:

1. **Re-read the relevant phase document**
   - Check if you missed something
   - Review examples and best practices

2. **Review similar past work**
   - Look at git history
   - Find similar implementations
   - Learn from patterns

3. **Ask specific questions**
   - Be clear about what you need
   - Provide context
   - Share what you've tried

4. **Request code review early**
   - Get feedback on design before implementing
   - Share your approach
   - Iterate based on feedback

## 🔄 Continuous Improvement

This library is a living document:

- **Share feedback**: What's unclear or missing?
- **Contribute examples**: Share good patterns you've found
- **Update for changes**: Keep it current with project evolution
- **Learn and teach**: Help others learn the workflow

## 🎉 Conclusion

Following this workflow ensures:
- 🎯 Clear requirements understanding
- 🏗️ Well-designed solutions
- 📊 Proper data modeling
- 🧪 Comprehensive testing
- 💻 Clean implementation
- ✅ Thorough verification

The result: **High-quality, maintainable code that meets requirements and doesn't break existing functionality.**

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-12  
**Maintained by**: Repository Contributors

For questions or suggestions, please open an issue or submit a pull request.
