#!/usr/bin/env python3
"""
Script to check for uncommitted changes in a Git repository.
This can be used as a pre-commit hook or run manually before deployments.
"""

import subprocess
import sys


def check_uncommitted_changes():
    """
    Check if there are any uncommitted changes in the repository.
    
    Returns:
        bool: True if there are uncommitted changes, False otherwise
    """
    try:
        # Run git status --porcelain to check for changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        return False


def main():
    """Main function to check for uncommitted changes and exit accordingly."""
    print("Checking for uncommitted changes...")
    
    if check_uncommitted_changes():
        print("❌ ERROR: Uncommitted changes detected!")
        print()
        
        # Show the uncommitted changes
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            print("The following files have uncommitted changes:")
            print(result.stdout)
        except subprocess.CalledProcessError:
            pass
        
        print("Please commit or stash these changes before proceeding.")
        sys.exit(1)
    else:
        print("✅ SUCCESS: No uncommitted changes detected")
        sys.exit(0)


if __name__ == '__main__':
    main()
