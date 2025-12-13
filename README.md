# linxiaotu

## Repository Overview

This repository contains multiple projects including a quantitative trading system and AI engineering resources.

## AI Development Guidelines

All AI-assisted development in this repository must follow the structured Task Workflow documented in:

📚 **[AI Rules and Guidelines](docs/AI_RULES_EN.md)**

The workflow consists of 6 mandatory phases:
1. **Questions** - Requirements gathering
2. **Design** - Solution planning
3. **Models** - Data structure definition
4. **Tests** - Test creation (TDD)
5. **Implementation** - Code changes
6. **Verification** - Testing and validation

For detailed guidance on each phase, see the [Execution Library](docs/ai/en/README.md).

## Projects

- **[Quantitative Trading System](quant-trading-system/)** - Automated futures trading system
- AI Engineering Resources
- Reference Materials
## Uncommitted Changes Detection

This repository includes tools to detect uncommitted changes before deployments or commits:

### Scripts

- **check_uncommitted.py**: Python script to check for uncommitted changes
- **check-uncommitted.sh**: Bash script to check for uncommitted changes

### Usage

Run the check manually:

```bash
# Using Python script
python3 check_uncommitted.py

# Using Bash script
./check-uncommitted.sh
```

### GitHub Actions

The repository includes a GitHub Actions workflow that automatically checks for uncommitted changes:
- Runs on pushes to `main` and `develop` branches
- Runs on pull requests to `main` and `develop` branches
- Fails if uncommitted changes are detected after running tests/builds

### Exit Codes

- **0**: No uncommitted changes detected (success)
- **1**: Uncommitted changes detected (failure)
Personal Learning and Development Repository

## Contents

### 📚 Context Engineering for Claude Code
Comprehensive guide on Context Engineering methodology for AI programming assistants.

- **[Complete Guide](./context-engineering-for-claude-code/README.md)** - Full documentation with bilingual content (中文/English)
- **[Quick Start](./context-engineering-for-claude-code/QUICK_START.md)** - Get started in 5 minutes
- **[Templates](./context-engineering-for-claude-code/templates/)** - Ready-to-use templates for your projects

**Key Topics Covered:**
- What is Context Engineering and why it matters
- Three-Phase Workflow (RPI: Research, Plan, Implement)
- Context files and project documentation
- Subagents and specialized workflows
- Advanced optimization techniques
- Practical implementation guide

### 🔧 Quantitative Trading System
Python-based quantitative trading system with multi-factor strategies.

- Location: `./quant-trading-system/`
- [Documentation](./quant-trading-system/README.md)

### 🤖 AI Engineering Projects
Various AI engineering projects and experiments.

- AI Podcast Generation: `./ai-engineering-hub-main/ai-podcast-generation/`
- Paralegal Agent Crew: `./ai-engineering-hub-main/paralegal-agent-crew/`

### 📖 Reference Materials
- Awesome Mac Tools: `./awesome-mac-master/`
- North Star Framework: `./northstar-master/`
- AI Tools System Prompts: `./system-prompts-and-models-of-ai-tools-main/`

## Quick Links

- [Context Engineering Guide](./context-engineering-for-claude-code/README.md)
- [Trading System](./quant-trading-system/README.md)

## License

MIT
