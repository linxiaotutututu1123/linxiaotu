#!/usr/bin/env python3
"""
Script to check for uncommitted changes in a Git repository.
This can be used as a pre-commit hook or run manually before deployments.
"""

import subprocess
import sys


def get_git_status():
    """
    Get the git status output.
    
    Returns:
        str: Output from git status --porcelain
    
    Raises:
        SystemExit: If git command fails
    """
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print("Make sure you're in a git repository.", file=sys.stderr)
        sys.exit(2)


def main():
    """Main function to check for uncommitted changes and exit accordingly."""
    print("Checking for uncommitted changes...")
    
    status_output = get_git_status()
    
    if status_output:
        print("❌ ERROR: Uncommitted changes detected!")
        print()
        print("The following files have uncommitted changes:")
        print(status_output)
        print()
        print("Please commit or stash these changes before proceeding.")
        sys.exit(1)
    else:
        print("✅ SUCCESS: No uncommitted changes detected")
        sys.exit(0)


if __name__ == '__main__':
    main()
