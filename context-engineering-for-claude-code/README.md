# Context Engineering for Claude Code

> 彻底改写 Claude Code 编程方式：从提示词工程到上下文工程
> 
> Transforming AI Programming: From Prompt Engineering to Context Engineering

## 目录 / Table of Contents

- [什么是上下文工程 / What is Context Engineering](#what-is-context-engineering)
- [为什么重要 / Why It Matters](#why-it-matters)
- [核心实践 / Core Practices](#core-practices)
- [进阶技巧 / Advanced Tips](#advanced-tips)
- [工具和资源 / Tools and Resources](#tools-and-resources)
- [实战指南 / Practical Guide](#practical-guide)
- [总结 / Summary](#summary)

---

## What is Context Engineering?

### 定义 / Definition

上下文工程是一种超越传统提示词工程的方法论，它不仅关注"如何询问"，更关注如何设计、传递和持续优化 AI 编程助手（如 Claude Code）在执行编码任务前后可用的**整个上下文**。

**Context Engineering** is a methodology that goes beyond traditional prompt engineering. Rather than optimizing single prompts, it structures all relevant information—project rules, code examples, documentation links, system design notes, validation scripts, and architecture decisions—into well-organized context files and workflows that an AI coding agent can access.

### 核心区别 / Key Differences

| 维度 | 提示词工程 Prompt Engineering | 上下文工程 Context Engineering |
|------|------------------------------|--------------------------------|
| 焦点 | 单次请求优化 | 整体信息架构 |
| 范围 | 单个提示词 | 项目规则、代码示例、文档、验证脚本等 |
| 比喻 | 递便签纸 (sticky note) | 提供完整剧本和参考资料 (detailed screenplay) |
| 目标 | 获得更好的单次响应 | 建立持续准确的编程助手系统 |

---

## Why It Matters

### 为什么上下文工程对 Claude Code 和 AI 代理至关重要？

#### 1. 减少错误 / Reduces Errors

大多数 AI 错误源于**缺失或模糊的上下文**。当代理"假设"不符合项目实际的情况时，就会导致错误。

Most AI errors stem from missing or ambiguous context, leading to mistakes when the agent "assumes" things that aren't true for your project.

#### 2. 处理复杂性 / Enables Complexity

有了全面的上下文，AI 代理可以推理多步骤功能和遗留系统，而不仅仅是编写全新代码。

With comprehensive context, AI agents can reason about multi-step features and legacy systems, instead of only writing greenfield code.

#### 3. 自我修正 / Self-Correction

验证循环和脚本化检查确保代理能够学习并自我修正，随着时间推移改进输出。

Validation loops and scripted checks ensure the agent learns and corrects itself, improving outputs over time.

---

## Core Practices

### 三阶段工作流：RPI / Three-Phase Workflow (RPI)

#### **R - Research (研究)**

收集项目需求、代码引用、系统依赖和业务逻辑到上下文文件中。

**Gather** project requirements, code references, system dependencies, and business logic into context files.

**关键活动：**
- 分析现有代码库结构
- 识别依赖关系和技术栈
- 记录业务规则和约束
- 收集相关文档和 API 规范

#### **P - Plan (规划)**

使用这些文件创建详细计划（功能规范、实现蓝图），然后再生成代码。

**Use** those files to create detailed plans (feature specs, implementation blueprints) before generating code.

**关键活动：**
- 创建功能规范文档
- 设计实现方案
- 定义验证标准
- 规划测试策略

#### **I - Implement (实现)**

仅使用**最少、最相关的上下文**来生成代码，以优化代理的"专注力"。提供的不必要上下文越多，性能下降的可能性就越大。

**Generate** code using only the minimum, most relevant context to optimize the agent's "focus"—the more unnecessary context you provide, the more likely performance drops.

**关键原则：**
- 避免上下文污染
- 保持窗口清晰
- 逐步迭代
- 及时验证

### 上下文文件 / Context Files

#### 文件结构 / File Structure

```
.claude/
├── CLAUDE.md          # 项目总览和规则
├── INITIAL.md         # 初始化指南
├── agents/            # 子代理配置
│   ├── reviewer.yaml
│   ├── architect.yaml
│   └── tester.yaml
└── context/           # 专项上下文
    ├── api-specs.md
    ├── conventions.md
    └── migrations.md
```

#### 内容组织 / Content Organization

**必须包含的信息：**
- 项目架构和目录结构
- 模块关系和依赖
- 编码规范和最佳实践
- 当前已知问题和待迁移项
- 使用历史和常见模式
- API 规范和数据结构
- 验证脚本和测试策略

**示例 - CLAUDE.md 结构：**

```markdown
# Project Context

## Architecture
- Overview of system design
- Key modules and their relationships
- Technology stack

## Conventions
- Coding standards (PEP 8, ESLint, etc.)
- Naming conventions
- File organization patterns

## Current State
- Active branches and features
- Known issues and technical debt
- Pending migrations

## Development Workflow
- Build commands
- Test procedures
- Deployment process

## Constraints
- Performance requirements
- Security considerations
- Compliance needs
```

### 子代理 / Subagents and Specialized Workflows

#### 什么是子代理？/ What are Subagents?

在 Claude Code 中为不同领域创建定制的"子代理"（例如，代码审查员、数据库架构师）。每个子代理都有自己的上下文窗口、系统提示和工具集。

Create custom "subagents" within Claude Code for different domains (e.g., code reviewer, database architect). Each has its own context window, system prompt, and toolset.

#### 优势 / Benefits

- **防止上下文污染** - Prevent context pollution
- **提高特定领域响应质量** - Improve domain-specific responses
- **专业化任务处理** - Specialized task handling
- **并行工作能力** - Parallel workflow capability

#### 示例配置 / Example Configuration

```yaml
# .claude/agents/code-reviewer.yaml
name: Code Reviewer
description: Expert in code quality and best practices
system_prompt: |
  You are an expert code reviewer focused on:
  - Code quality and maintainability
  - Security vulnerabilities
  - Performance optimization
  - Best practices adherence
context_files:
  - .claude/context/conventions.md
  - .claude/context/security-guidelines.md
tools:
  - static_analysis
  - security_scanner
```

---

## Advanced Tips

### 1. 优化上下文窗口 / Optimize the Context Window

#### 40% 愚蠢区规则 / The "40% Dumb Zone" Rule

当上下文窗口被不相关信息填充超过 40% 时，性能会急剧下降。

Over-filling the context window with irrelevant info tanks performance sharply.

**最佳实践：**
- 只包含当前任务相关的上下文
- 定期清理和更新上下文文件
- 使用分层的上下文结构
- 根据任务类型选择性加载上下文

### 2. 强效关键词 / Power Keywords

使用文档化良好的关键词进行指令，但要强调清晰性和具体性，而不是模糊的请求。

Use well-documented keywords for instructions, but emphasize clarity and specificity over vague requests.

**有效关键词：**
- `IMPORTANT:` - 标记关键约束
- `MUST:` - 强制性要求
- `NEVER:` - 禁止操作
- `ALWAYS:` - 必须遵循的规则
- `CRITICAL:` - 关键安全或业务逻辑

**避免：**
- 过度使用感叹号
- 模糊的描述词（"好的"、"优秀的"）
- 没有具体标准的主观要求

### 3. 验证脚本 / Validation Scripts

提示代理生成检查其自身输出的脚本，而不仅仅是代码本身。

Prompt the agent to generate scripts that check its own output, not just the code itself.

**验证层次：**

```markdown
## Validation Checklist

### Level 1: Syntax and Linting
- [ ] Code passes linter
- [ ] No syntax errors
- [ ] Follows style guide

### Level 2: Unit Tests
- [ ] All new functions have tests
- [ ] Tests cover edge cases
- [ ] Tests pass locally

### Level 3: Integration Tests
- [ ] API contracts maintained
- [ ] Database migrations work
- [ ] External dependencies mocked

### Level 4: Security and Performance
- [ ] No security vulnerabilities
- [ ] Performance benchmarks met
- [ ] Resource usage acceptable
```

### 4. 版本化的需求和约束 / Versioned Requirements and Constraints

维护更新的文档，以便 AI 不会做出"合理"但在上下文中无效的假设。

Maintain updated documentation so the AI does not make "reasonable" but contextually invalid assumptions.

**示例：**

```markdown
# API Schema v2.1.0

## Breaking Changes from v2.0.0
- `user_id` field renamed to `userId` (camelCase)
- Authentication now requires OAuth 2.0
- Rate limiting: 100 requests/minute

## Migration Guide
1. Update all field references
2. Implement OAuth flow
3. Add rate limiting handling
```

### 5. 迭代改进和同行评审 / Iterative Improvement and Peer Review

将上下文工程视为代码审查的一种形式；保持上下文文件和工作流规范经过同行评审并保持最新。

Treat context engineering as a form of code review; keep context files and workflow specs peer-reviewed and up-to-date.

---

## Tools and Resources

### 官方资源 / Official Resources

1. **Claude Code 官方文档**
   - [Subagents Documentation](https://code.claude.com/docs/en/sub-agents)
   - Context Engineering best practices

2. **GitHub 模板和起步工具**
   - [Context Engineering Intro](https://github.com/coleam00/context-engineering-intro)
   - [Full Workflow Guide](https://github.com/coleam00/context-engineering-intro/tree/main/claude-code-full-guide)

3. **学术资源**
   - [arXiv Paper: Multi-Agent Context Engineering for LLM Code Assistants](https://arxiv.org/abs/2508.08322)

4. **实践指南和案例研究**
   - [AngeloDiPaolo's Blog](https://angelodipaolo.com/blog/claude-code-context-engineering/)
   - [LiquidMetal AI Guide](https://liquidmetal.ai/casesAndBlogs/context-engineering-claude-code/)
   - [ClaudeKit Blog](https://claudekit.cc/blog/context-engineering-how-to-turn-ai-coding-agents-into-production-ready-tools)

### 工具生态 / Tool Ecosystem

| 工具类型 | 推荐工具 | 用途 |
|---------|---------|------|
| 上下文管理 | Claude Code, Cursor | IDE 集成 |
| 文档生成 | Markdown, MDX | 上下文文件创建 |
| 验证工具 | ESLint, Pylint, Prettier | 代码质量检查 |
| 测试框架 | Jest, Pytest, Mocha | 自动化测试 |
| 版本控制 | Git, GitHub | 上下文版本管理 |

---

## Practical Guide

### 快速开始 / Quick Start

#### 第一步：创建基础上下文文件

```bash
# 创建目录结构
mkdir -p .claude/agents .claude/context

# 创建主上下文文件
touch .claude/CLAUDE.md .claude/INITIAL.md
```

#### 第二步：填充项目信息

```markdown
# .claude/CLAUDE.md

# Project: [Your Project Name]

## Overview
[Brief description of the project]

## Technology Stack
- Language: [e.g., Python 3.11]
- Framework: [e.g., Django 4.2]
- Database: [e.g., PostgreSQL 15]
- Key Libraries: [list main dependencies]

## Project Structure
\`\`\`
src/
├── api/          # REST API endpoints
├── models/       # Data models
├── services/     # Business logic
└── utils/        # Helper functions
\`\`\`

## Coding Standards
- Follow PEP 8 for Python
- Use type hints for all functions
- Document with docstrings (Google style)
- Maximum line length: 88 characters (Black formatter)

## Development Commands
\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linter
black . && flake8 .

# Start development server
python manage.py runserver
\`\`\`

## Current Priorities
1. [Priority 1]
2. [Priority 2]
3. [Priority 3]
```

#### 第三步：定义工作流程

```markdown
# .claude/INITIAL.md

# Initial Setup for New Tasks

## Before Starting
1. Read CLAUDE.md for project context
2. Check current branch and git status
3. Review relevant context files in .claude/context/
4. Understand the specific task requirements

## RPI Workflow

### Research Phase
- [ ] Analyze existing code related to the task
- [ ] Identify dependencies and impacts
- [ ] Review similar implementations
- [ ] Document findings

### Plan Phase
- [ ] Create implementation plan
- [ ] Define acceptance criteria
- [ ] List required tests
- [ ] Identify potential risks

### Implement Phase
- [ ] Write tests first (TDD)
- [ ] Implement changes incrementally
- [ ] Run tests after each change
- [ ] Update documentation

## Validation Checklist
- [ ] All tests pass
- [ ] Code follows project conventions
- [ ] No linting errors
- [ ] Documentation updated
- [ ] Security considerations addressed
```

#### 第四步：配置子代理（可选）

```yaml
# .claude/agents/security-reviewer.yaml
name: Security Reviewer
description: Expert in identifying security vulnerabilities
system_prompt: |
  You are a security expert. Review code for:
  - SQL injection vulnerabilities
  - XSS vulnerabilities
  - Authentication/authorization issues
  - Sensitive data exposure
  - OWASP Top 10 risks
context_files:
  - .claude/context/security-guidelines.md
```

### 常见场景应用 / Common Use Cases

#### 场景 1：新功能开发

```markdown
## Task: Add User Authentication

### Research
- Existing auth patterns in codebase
- Available authentication libraries
- Security requirements

### Plan
1. Choose OAuth 2.0 framework
2. Design user model and database schema
3. Implement JWT token management
4. Create authentication middleware
5. Add tests for auth flows

### Context Files Needed
- .claude/context/security-guidelines.md
- .claude/context/api-specs.md
- .claude/context/database-schema.md
```

#### 场景 2：Bug 修复

```markdown
## Task: Fix Race Condition in Order Processing

### Research
- Current order processing flow
- Identified race condition scenario
- Transaction handling approach

### Plan
1. Add database transaction wrapper
2. Implement optimistic locking
3. Add retry logic
4. Write concurrency tests

### Context Files Needed
- .claude/context/database-transactions.md
- .claude/context/testing-strategy.md
```

#### 场景 3：代码重构

```markdown
## Task: Refactor Legacy Payment Module

### Research
- Current payment module structure
- Dependencies and usage patterns
- Breaking change impacts

### Plan
1. Create new payment interface
2. Implement adapter pattern
3. Migrate existing integrations
4. Maintain backward compatibility
5. Comprehensive test coverage

### Context Files Needed
- .claude/context/api-contracts.md
- .claude/context/migration-strategy.md
- .claude/context/backward-compatibility.md
```

---

## Summary

### 核心要点 / Key Takeaways

| 实践 | 重要性 | 实施方法 |
|-----|-------|---------|
| **上下文文件** | 减少猜测，提高准确性 | .claude/CLAUDE.md, INITIAL.md |
| **RPI 工作流** | 防止上下文污染 | Research, Plan, Implement, 清理窗口 |
| **子代理** | 任务专业化，更好的上下文 | 自定义 YAML 在 .claude/agents/ |
| **验证脚本** | 自我纠错，减少监督 | PRP 中的脚本提示 |
| **同行评审上下文** | 确保完整性，降低风险 | 维护文档，与团队共享 |

### 成功的上下文工程 = Success Formula

```
优秀的 AI 编程助手 = 
  强大的基础模型 (Claude) 
  + 清晰的上下文架构 (Context Engineering)
  + 持续的验证循环 (Validation)
  + 团队协作和知识共享 (Collaboration)
```

### 下一步行动 / Next Steps

1. **立即开始** - 创建 `.claude/` 目录和基础上下文文件
2. **迭代改进** - 根据实际使用情况不断优化上下文
3. **分享学习** - 与团队共享最佳实践和经验教训
4. **持续学习** - 关注上下文工程的最新发展和工具

### 持续改进 / Continuous Improvement

上下文工程不是一次性工作，而是持续的过程：

- **每周回顾** - 检查上下文文件的有效性
- **定期更新** - 保持文档与代码库同步
- **收集反馈** - 记录 AI 代理的错误和改进机会
- **优化结构** - 根据使用模式调整上下文组织

---

## 参考文献 / References

1. Context Engineering for Claude Code - aivi.fyi
2. GitHub - coleam00/context-engineering-intro
3. Context Engineering with Claude Code - AngeloDiPaolo
4. Complete Context Engineering Guide - Claude AI
5. Context Engineering for Claude Code - LiquidMetal AI
6. Context Engineering: AI Coding Agents for Production - ClaudeKit
7. Subagents - Claude Code Docs
8. Context Engineering for Multi-Agent LLM Code Assistants - arXiv

---

## 附录 / Appendix

### A. 上下文文件模板 / Context File Templates

详见 `templates/` 目录下的示例文件。

### B. 常见问题解答 / FAQ

**Q: 上下文文件应该多详细？**
A: 包含足够的信息让 AI 做出正确决策，但避免不相关的细节。遵循"最小必要上下文"原则。

**Q: 多久更新一次上下文文件？**
A: 当项目架构、约束或最佳实践发生变化时立即更新。建议每个 Sprint 或重大功能完成后进行审查。

**Q: 如何处理敏感信息？**
A: 永远不要在上下文文件中包含密码、API 密钥或其他敏感数据。使用环境变量引用或占位符。

**Q: 子代理是必需的吗？**
A: 不是必需的，但对于复杂项目或专业化任务，子代理可以显著提高质量和效率。

### C. 检查清单 / Checklists

#### 项目设置检查清单
- [ ] 创建 .claude/ 目录结构
- [ ] 编写 CLAUDE.md 项目概述
- [ ] 创建 INITIAL.md 工作流指南
- [ ] 配置必要的子代理
- [ ] 添加项目特定的上下文文件
- [ ] 设置验证脚本
- [ ] 与团队共享和审查

#### 任务执行检查清单
- [ ] 阅读相关上下文文件
- [ ] 完成 Research 阶段
- [ ] 创建详细的 Plan
- [ ] 执行 Implementation
- [ ] 运行验证脚本
- [ ] 更新文档
- [ ] 提交变更

---

**最后更新 / Last Updated:** 2024-12-12

**维护者 / Maintainer:** linxiaotu Repository

**许可证 / License:** MIT
