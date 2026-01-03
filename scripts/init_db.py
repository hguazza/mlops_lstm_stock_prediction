"""Database initialization script.

This script:
1. Runs Alembic migrations to create/update database schema
2. Creates an initial admin user if one doesn't exist

Usage:
    python scripts/init_db.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def run_migrations():
    """Run database migrations using SQLAlchemy directly."""
    from src.infrastructure.database.connection import init_db

    print("Running database migrations...")

    # Initialize database (create tables)
    await init_db()

    print("✓ Migrations completed")


async def create_admin_user():
    """Create initial admin user if it doesn't exist."""
    from src.domain.auth.password import hash_password
    from src.infrastructure.database.connection import async_session_maker
    from src.infrastructure.database.repositories import UserRepository

    # Get admin credentials from environment
    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "changeme123")
    admin_full_name = os.getenv("ADMIN_FULL_NAME", "Admin User")

    print(f"\nChecking for admin user: {admin_email}")

    async with async_session_maker() as session:
        user_repo = UserRepository(session)

        # Check if admin user already exists
        existing_user = await user_repo.get_by_email(admin_email)

        if existing_user:
            print(f"✓ Admin user already exists: {admin_email}")
            if not existing_user.is_superuser:
                print("  Warning: User exists but is not a superuser!")
            return

        # Create admin user
        print(f"Creating admin user: {admin_email}")

        hashed_password = hash_password(admin_password)

        await user_repo.create(
            email=admin_email,
            hashed_password=hashed_password,
            full_name=admin_full_name,
            is_superuser=True,
        )

        await session.commit()

        print("✓ Admin user created successfully")
        print(f"  Email: {admin_email}")
        print(f"  Password: {admin_password}")
        print("\n⚠️  IMPORTANT: Change the admin password after first login!")


async def main():
    """Main initialization function."""
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)

    try:
        # Run migrations (async)
        await run_migrations()

        # Create admin user (async)
        await create_admin_user()

        print("\n" + "=" * 60)
        print("Database initialization completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during initialization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
