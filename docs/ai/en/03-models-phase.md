# Models Phase

## 🎯 Purpose

The Models Phase focuses on defining or updating data structures, schemas, and models. This ensures data integrity and provides a clear contract for how information flows through the system.

## 📋 Objectives

- Define data models and schemas
- Plan database changes if needed
- Document data relationships
- Ensure data validation rules
- Plan migrations if necessary

## ✅ Checklist

### Data Model Review
- [ ] Review existing data models
- [ ] Identify models that need updates
- [ ] Identify new models needed
- [ ] Understand model relationships
- [ ] Check for naming conventions

### Schema Definition
- [ ] Define all required fields
- [ ] Define field types and constraints
- [ ] Define optional vs required fields
- [ ] Define default values
- [ ] Define validation rules

### Relationships
- [ ] Identify foreign key relationships
- [ ] Define one-to-one relationships
- [ ] Define one-to-many relationships
- [ ] Define many-to-many relationships
- [ ] Document relationship constraints

### Database Planning
- [ ] Plan any table changes
- [ ] Plan index updates if needed
- [ ] Identify performance implications
- [ ] Plan migration scripts
- [ ] Plan rollback scripts

### Validation
- [ ] Define input validation rules
- [ ] Define business logic constraints
- [ ] Define format requirements
- [ ] Plan error messages
- [ ] Consider edge cases

## 🗂️ Model Types

### Data Models (ORM)
For database-backed models:
```python
class User:
    id: int
    email: str
    created_at: datetime
    is_active: bool = True
```

### Schema Models (Validation)
For API request/response:
```python
class UserSchema:
    email: EmailStr
    password: constr(min_length=8)
    age: conint(ge=18)
```

### Domain Models (Business Logic)
For business entities:
```python
class TradingStrategy:
    name: str
    signals: List[Signal]
    parameters: Dict[str, float]
```

### Configuration Models
For settings and config:
```python
class AppConfig:
    database_url: str
    api_key: SecretStr
    debug: bool = False
```

## 📝 Model Definition Template

```python
"""
Module: models/user.py
Purpose: User account data model
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean

class User(Base):
    """
    User account model
    
    Attributes:
        id: Unique user identifier
        email: User email address (unique)
        created_at: Account creation timestamp
        is_active: Account status flag
    """
    __tablename__ = 'users'
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Required Fields
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Optional Fields
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    
    # Validation
    def validate_email(self):
        """Validate email format"""
        if '@' not in self.email:
            raise ValueError("Invalid email format")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
```

## 🏗️ Design Principles

### Single Responsibility
- Each model represents one concept
- Don't mix concerns in a single model
- Keep models focused and cohesive

### Data Integrity
- Use appropriate data types
- Add constraints at model level
- Validate data before saving
- Use transactions for consistency

### Performance
- Add indexes for frequently queried fields
- Avoid N+1 query problems
- Use lazy loading appropriately
- Consider caching strategies

### Maintainability
- Use clear, descriptive names
- Document all fields
- Keep models simple
- Version your schemas

## 🗄️ Database Migration Planning

### When Migrations Are Needed
- Adding new tables
- Adding/removing columns
- Changing column types
- Adding/removing indexes
- Changing constraints

### Migration Template
```python
"""
Migration: Add email_verified field to users table
Date: 2025-12-12
"""

def upgrade():
    """Add email_verified column"""
    op.add_column('users',
        sa.Column('email_verified', sa.Boolean(), 
                  default=False, nullable=False)
    )
    
    # Set existing users to verified
    op.execute("UPDATE users SET email_verified = TRUE")

def downgrade():
    """Remove email_verified column"""
    op.drop_column('users', 'email_verified')
```

## 🎯 Project-Specific Guidelines

### For Quantitative Trading System

#### Strategy Models
```python
class BaseStrategy:
    """Base class for all trading strategies"""
    def generate_signals(self) -> Dict[str, int]:
        """Return trading signals for each symbol"""
        raise NotImplementedError
```

#### Factor Models
```python
class MultiFactorModel:
    """Multi-factor analysis model"""
    technical_weight: float = 0.30
    fund_flow_weight: float = 0.25
    sentiment_weight: float = 0.25
    correlation_weight: float = 0.15
    anomaly_weight: float = 0.05
```

#### Trade Models
```python
class Trade:
    """Represents a single trade"""
    symbol: str
    direction: int  # 1=buy, -1=sell
    quantity: int
    price: float
    timestamp: datetime
```

### Configuration Models
All configuration should be in `config/settings.yaml`:
```yaml
multi_factor:
  weights:
    technical: 0.30
    fund_flow: 0.25
    sentiment: 0.25
    correlation: 0.15
    anomaly: 0.05
```

## ✅ Validation Rules

### Common Validations
- **Email**: RFC 5322 compliant
- **Phone**: E.164 format
- **URL**: Valid URL format
- **Date**: ISO 8601 format
- **Numeric**: Range validation

### Business Logic Validations
- **Trading**: Position limits, risk checks
- **Authentication**: Password strength, rate limits
- **Financial**: Decimal precision, non-negative amounts

### Example Validation
```python
def validate(self):
    """Validate model data"""
    if not self.symbol:
        raise ValueError("Symbol is required")
    
    if self.quantity <= 0:
        raise ValueError("Quantity must be positive")
    
    if self.direction not in [-1, 0, 1]:
        raise ValueError("Direction must be -1, 0, or 1")
```

## 🚨 Common Pitfalls

Avoid these mistakes:

- ❌ Not validating data at model level
- ❌ Using wrong data types (e.g., float for money)
- ❌ Forgetting to add indexes
- ❌ Not documenting field meanings
- ❌ Making fields required without defaults
- ❌ Not planning migrations
- ❌ Mixing business logic with model definitions
- ❌ Creating overly complex models

## 📊 Output

At the end of the Models Phase, you should have:

1. **Model Definitions**
   - All new models defined
   - All updated models documented
   - Validation rules specified

2. **Schema Documentation**
   - Field descriptions
   - Type information
   - Constraint documentation
   - Relationship diagrams

3. **Migration Plan**
   - Database changes documented
   - Migration scripts written
   - Rollback scripts prepared
   - Test data plan

4. **Validation Rules**
   - All validation rules documented
   - Error messages defined
   - Edge cases considered

## ✨ Best Practices

### Do's
- ✅ Use type hints
- ✅ Add field documentation
- ✅ Validate at model level
- ✅ Use appropriate data types
- ✅ Plan for null values
- ✅ Index frequently queried fields
- ✅ Test migrations with sample data

### Don'ts
- ❌ Skip validation
- ❌ Use generic field names
- ❌ Make everything nullable
- ❌ Forget to document relationships
- ❌ Use wrong data types
- ❌ Skip migration testing
- ❌ Mix concerns in models

## 🎓 Examples

### Good Model Definition
```python
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, validator, Field

class TradeSignal(BaseModel):
    """
    Trading signal generated by strategy
    
    Attributes:
        symbol: Trading symbol (e.g., 'RB', 'IF')
        signal: Trade direction (-1=sell, 0=hold, 1=buy)
        confidence: Signal confidence score [0.0, 1.0]
        timestamp: Signal generation time
    """
    symbol: str = Field(..., min_length=1, max_length=10)
    signal: int = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isupper():
            raise ValueError('Symbol must be uppercase')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### Poor Model Definition
```python
class Trade:
    """A trade"""
    s: str  # What is 's'?
    d: int  # What is 'd'?
    q: float  # Wrong type for quantity
    # No validation, no documentation
```

## 🔄 Transition to Tests Phase

Exit Criteria:
- ✅ All data models are defined
- ✅ All schemas are documented
- ✅ Validation rules are specified
- ✅ Database migrations are planned
- ✅ Relationships are documented

Once all exit criteria are met, proceed to: [04-tests-phase.md](./04-tests-phase.md)

---

**Remember**: Well-defined models are the foundation of maintainable code. Take time to get them right.
