# Project Context Template

## Project Overview

**Project Name:** [Your Project Name]  
**Version:** [e.g., 1.0.0]  
**Description:** [Brief description of what the project does]  
**Repository:** [Link to repository]

## Technology Stack

### Primary Language(s)
- [e.g., Python 3.11, TypeScript 5.0]

### Frameworks & Libraries
- **Web Framework:** [e.g., FastAPI, Express.js, Django]
- **Database:** [e.g., PostgreSQL 15, MongoDB 6.0]
- **ORM/Database Library:** [e.g., SQLAlchemy, Prisma]
- **Testing:** [e.g., pytest, Jest]
- **Other Key Dependencies:** [List important libraries]

## Project Structure

```
project-root/
├── src/                    # Source code
│   ├── api/               # API endpoints/routes
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   ├── utils/             # Helper functions
│   └── config/            # Configuration files
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── docs/                  # Documentation
├── scripts/               # Build and deployment scripts
└── .claude/              # Context engineering files
```

## Architecture

### System Design
[Describe the overall architecture - monolith, microservices, etc.]

### Key Components
1. **Component A:** [Purpose and responsibilities]
2. **Component B:** [Purpose and responsibilities]
3. **Component C:** [Purpose and responsibilities]

### Data Flow
[Describe how data flows through the system]

## Coding Standards

### General Guidelines
- Follow [PEP 8 / Google Style Guide / Airbnb Style Guide]
- Use [snake_case / camelCase] for variables and functions
- Use [PascalCase] for classes
- Use [UPPER_CASE] for constants

### Documentation
- All public functions must have docstrings
- Use [Google / NumPy / reStructuredText] docstring format
- Include type hints for all function parameters and returns

### Code Quality
- Maximum line length: [80 / 88 / 100] characters
- Maximum function length: [50 / 100] lines
- Maximum file length: [500 / 1000] lines
- Cyclomatic complexity: max [10 / 15]

## Development Workflow

### Setup
```bash
# Clone repository
git clone [repository-url]

# Install dependencies
[npm install / pip install -r requirements.txt / etc.]

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

### Development Commands
```bash
# Start development server
[npm run dev / python manage.py runserver / etc.]

# Run tests
[npm test / pytest / etc.]

# Run linter
[npm run lint / black . && flake8 / etc.]

# Format code
[npm run format / black . / prettier --write . / etc.]

# Build for production
[npm run build / python setup.py build / etc.]
```

### Git Workflow
- Branch naming: [feature/*, bugfix/*, hotfix/*]
- Commit message format: [Conventional Commits / etc.]
- Pull request requirements: [tests passing, code review, etc.]

## Current State

### Active Features
- [Feature 1]: [Status and notes]
- [Feature 2]: [Status and notes]

### Known Issues
- [Issue 1]: [Description and workaround if any]
- [Issue 2]: [Description and workaround if any]

### Technical Debt
- [Debt item 1]: [Description and priority]
- [Debt item 2]: [Description and priority]

### Pending Migrations
- [Migration 1]: [What needs to be migrated and why]
- [Migration 2]: [What needs to be migrated and why]

## Constraints & Requirements

### Performance
- API response time: < [100ms / 500ms]
- Database query time: < [50ms / 200ms]
- Page load time: < [2s / 3s]

### Security
- Authentication: [OAuth 2.0 / JWT / Session-based]
- Authorization: [RBAC / ABAC]
- Data encryption: [at rest / in transit]
- Compliance: [GDPR / HIPAA / SOC 2]

### Scalability
- Expected load: [requests/second]
- Scaling strategy: [horizontal / vertical]
- Caching strategy: [Redis / Memcached / CDN]

## API Conventions

### REST API
- Base URL: [/api/v1]
- Authentication: [Bearer token in Authorization header]
- Response format: JSON
- Error format: [Standard error response structure]

### Status Codes
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Database

### Schema Overview
[Link to schema documentation or brief description]

### Naming Conventions
- Tables: [plural / singular], [snake_case / camelCase]
- Columns: [snake_case / camelCase]
- Foreign keys: [table_id / tableId]
- Indexes: [idx_table_column]

### Migrations
- Tool: [Alembic / Prisma Migrate / Flyway]
- Process: [Description of migration workflow]

## Testing Strategy

### Test Coverage
- Target: [80% / 90%] code coverage
- Required: All new features must have tests

### Test Types
- **Unit Tests:** Test individual functions and methods
- **Integration Tests:** Test component interactions
- **E2E Tests:** Test complete user workflows

### Mocking
- Use [unittest.mock / jest.mock] for external dependencies
- Mock all network calls in unit tests
- Use test fixtures for database operations

## Deployment

### Environments
- **Development:** [URL or description]
- **Staging:** [URL or description]
- **Production:** [URL or description]

### CI/CD
- Platform: [GitHub Actions / GitLab CI / Jenkins]
- Pipeline stages: [lint → test → build → deploy]

### Monitoring
- Logging: [Structured logging with Winston / Python logging]
- Metrics: [Prometheus / DataDog / New Relic]
- Alerts: [PagerDuty / Opsgenie]

## Common Patterns

### Error Handling
[Description of error handling patterns used in the project]

### Logging
[Description of logging patterns and levels]

### Configuration
[How configuration is managed - environment variables, config files, etc.]

## Resources

### Documentation
- [Link to main documentation]
- [Link to API documentation]
- [Link to architecture diagrams]

### External Resources
- [Link to relevant external docs, tutorials, etc.]

## Notes

### Design Decisions
[Record important design decisions and their rationale]

### Future Considerations
[Things to keep in mind for future development]

---

**Last Updated:** [Date]  
**Maintainer:** [Name/Team]
