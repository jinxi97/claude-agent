#!/usr/bin/env python3
"""CLI for publishing/removing artifact files in the database.

Usage:
    python publish.py add <path> [path2 ...]     # register files (reads CHAT_ID env var)
    python publish.py remove <path> [path2 ...]   # unregister files
"""

import os
import sys

import db


def main():
    if len(sys.argv) < 3:
        print("Usage: python publish.py add|remove <path> [path2 ...]", file=sys.stderr)
        sys.exit(1)

    action = sys.argv[1]
    paths = sys.argv[2:]

    if action == "add":
        chat_id = os.environ.get("CHAT_ID")
        if not chat_id:
            print("Error: CHAT_ID environment variable not set", file=sys.stderr)
            sys.exit(1)
        for path in paths:
            db.publish_file(path, chat_id)
            print(f"Published: {path}")

    elif action == "remove":
        for path in paths:
            db.remove_file(path)
            print(f"Removed: {path}")

    else:
        print(f"Unknown action: {action}. Use 'add' or 'remove'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
