#!/usr/bin/env python3
"""
Test script to validate that no commits from December 2025 exist in the repository.

This script is used as a verification test to ensure all December 2025 commits
have been properly removed from the git history.

Exit codes:
  0 - Success: No December 2025 commits found
  1 - Failure: December 2025 commits still exist
"""

import subprocess
import sys
from typing import List


def get_december_commits() -> List[str]:
    """Get list of commit hashes from December 2025."""
    try:
        result = subprocess.run(
            [
                "git", "log", "--all",
                "--pretty=format:%H",
                "--since=2025-12-01",
                "--until=2025-12-31"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        commits = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error checking git history: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    """Validate that no December 2025 commits exist."""
    print("Validating December 2025 commits deletion...")
    print("-" * 60)
    
    commits = get_december_commits()
    
    if not commits:
        print("✓ PASS: No commits found in December 2025")
        print("  All December 2025 commits have been successfully deleted.")
        return 0
    else:
        print(f"✗ FAIL: Found {len(commits)} commit(s) in December 2025")
        print("\nCommit hashes that need to be deleted:")
        for commit in commits:
            print(f"  - {commit}")
        print("\nPlease remove these commits before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
