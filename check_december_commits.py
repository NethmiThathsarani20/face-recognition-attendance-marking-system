#!/usr/bin/env python3
"""
Script to check for commits made in December 2025 and verify if they should be deleted.

This script analyzes the git history to identify all commits made in December 2025
and provides information about them.
"""

import subprocess
import sys
from datetime import datetime
from typing import List, Dict


def run_git_command(args: List[str]) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return ""


def get_december_2025_commits() -> List[Dict[str, str]]:
    """Get all commits made in December 2025."""
    # Get commits from December 2025
    output = run_git_command([
        "log",
        "--all",
        "--pretty=format:%H|%ai|%an|%s",
        "--since=2025-12-01",
        "--until=2025-12-31"
    ])
    
    if not output:
        return []
    
    commits = []
    for line in output.split('\n'):
        if line.strip():
            parts = line.split('|', 3)
            if len(parts) == 4:
                commits.append({
                    'hash': parts[0],
                    'date': parts[1],
                    'author': parts[2],
                    'message': parts[3]
                })
    
    return commits


def main():
    """Main function to check December 2025 commits."""
    print("=" * 80)
    print("December 2025 Commits Check")
    print("=" * 80)
    print()
    
    commits = get_december_2025_commits()
    
    if not commits:
        print("✓ No commits found in December 2025")
        print("All December 2025 commits have been deleted or none were made.")
        return 0
    
    print(f"⚠ Found {len(commits)} commit(s) in December 2025:")
    print()
    
    for i, commit in enumerate(commits, 1):
        print(f"Commit {i}:")
        print(f"  Hash:    {commit['hash'][:12]}")
        print(f"  Date:    {commit['date']}")
        print(f"  Author:  {commit['author']}")
        print(f"  Message: {commit['message']}")
        print()
    
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print()
    print("To delete December 2025 commits, you can:")
    print("1. Use 'git rebase -i' to remove specific commits (requires force push)")
    print("2. Reset to a commit before December 2025")
    print("3. Create a new branch from a pre-December commit")
    print()
    print("⚠ WARNING: Deleting commits from shared branches requires force push")
    print("and should be coordinated with your team.")
    print()
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
