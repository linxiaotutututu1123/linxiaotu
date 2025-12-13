#!/bin/bash
# Script to check for uncommitted changes
# This can be used as a pre-commit hook or run manually

set -e

echo "Checking for uncommitted changes..."

# Check if there are any uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ ERROR: Uncommitted changes detected!"
    echo ""
    echo "The following files have uncommitted changes:"
    git status --porcelain
    echo ""
    echo "Please commit or stash these changes before proceeding."
    exit 1
else
    echo "✅ SUCCESS: No uncommitted changes detected"
    exit 0
fi
