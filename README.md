# linxiaotu

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
