"""Manage local accounts without passwords in command arguments or logs."""
import argparse
from getpass import getpass
from pathlib import Path
import sqlite3

from ihsanvision.auth import Accounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("action", choices=("add", "disable"))
    parser.add_argument("username")
    parser.add_argument("--role", choices=("admin", "operator"), default="operator")
    args = parser.parse_args()
    try:
        accounts = Accounts(args.db)
        if args.action == "disable":
            accounts.disable(args.username)
        else:
            password = getpass("Password (12-128 characters): ")
            if password != getpass("Confirm password: "):
                parser.error("Passwords do not match")
            accounts.add_user(args.username, password, args.role)
    except (ValueError, sqlite3.IntegrityError) as error:
        parser.error("Account already exists" if isinstance(error, sqlite3.IntegrityError) else str(error))
    print("Account operation completed. No credentials were printed.")


if __name__ == "__main__":
    main()
