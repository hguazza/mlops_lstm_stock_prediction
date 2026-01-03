"""Database connection and session management."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from src.config import get_config

config = get_config()

# Configure engine based on database type
if config.database.is_sqlite:
    # SQLite-specific configuration
    engine: AsyncEngine = create_async_engine(
        config.database.url,
        echo=config.database.echo,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Use StaticPool for SQLite
    )
else:
    # PostgreSQL configuration
    engine: AsyncEngine = create_async_engine(
        config.database.url,
        echo=config.database.echo,
        poolclass=NullPool,  # Use NullPool or default pool
        pool_pre_ping=True,  # Verify connections before using them
    )

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides async database session.

    Yields:
        AsyncSession: Database session

    Example:
        @router.get("/users")
        async def get_users(session: AsyncSession = Depends(get_async_session)):
            # Use session here
            pass
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database tables.

    Creates all tables defined in SQLAlchemy models.
    This is primarily for development/testing.
    In production, use Alembic migrations instead.
    """
    from src.infrastructure.database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
