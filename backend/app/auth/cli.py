"""Account maintenance.

Give documents uploaded before accounts existed to a user:
    python -m app.auth.cli claim --email you@example.com [--dry-run]
"""
import argparse
import sys

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.auth.security import normalize_email
from app.database import get_engine, init_db
from app.models import Document, User


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    claim = commands.add_parser("claim", help="assign every unowned document to a user")
    claim.add_argument("--email", required=True)
    claim.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    init_db()  # makes sure the owner column exists
    with Session(get_engine()) as db:
        user = db.scalar(select(User).where(User.email == normalize_email(args.email)))
        if user is None:
            print(f"No account for {args.email}. Register in the app first.")
            return 1
        unowned = db.scalar(
            select(func.count()).select_from(Document).where(Document.owner_id.is_(None))
        )
        print(f"{unowned} documents have no owner.")
        if args.dry_run or not unowned:
            return 0
        db.execute(
            update(Document).where(Document.owner_id.is_(None)).values(owner_id=user.id)
        )
        db.commit()
        print(f"Assigned {unowned} documents to {user.email}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())