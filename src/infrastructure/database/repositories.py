"""Data access repositories for database operations."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models import RefreshToken, User


class UserRepository:
    """Repository for User database operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize UserRepository.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(
        self,
        email: str,
        hashed_password: str,
        full_name: str,
        is_superuser: bool = False,
    ) -> User:
        """
        Create a new user.

        Args:
            email: User email
            hashed_password: Hashed password
            full_name: User's full name
            is_superuser: Whether user is superuser

        Returns:
            Created User instance
        """
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_superuser=is_superuser,
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User instance or None
        """
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User instance or None
        """
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def update(self, user: User) -> User:
        """
        Update user.

        Args:
            user: User instance to update

        Returns:
            Updated User instance
        """
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def delete(self, user: User) -> None:
        """
        Delete user.

        Args:
            user: User instance to delete
        """
        await self.session.delete(user)
        await self.session.flush()


class RefreshTokenRepository:
    """Repository for RefreshToken database operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize RefreshTokenRepository.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(
        self,
        token: str,
        user_id: str,
        expires_at: datetime,
    ) -> RefreshToken:
        """
        Create a new refresh token.

        Args:
            token: Refresh token string
            user_id: User ID
            expires_at: Token expiration datetime

        Returns:
            Created RefreshToken instance
        """
        refresh_token = RefreshToken(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
        )
        self.session.add(refresh_token)
        await self.session.flush()
        await self.session.refresh(refresh_token)
        return refresh_token

    async def get_by_token(self, token: str) -> Optional[RefreshToken]:
        """
        Get refresh token by token string.

        Args:
            token: Refresh token string

        Returns:
            RefreshToken instance or None
        """
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token == token)
        )
        return result.scalar_one_or_none()

    async def revoke(self, refresh_token: RefreshToken) -> RefreshToken:
        """
        Revoke a refresh token.

        Args:
            refresh_token: RefreshToken instance to revoke

        Returns:
            Updated RefreshToken instance
        """
        refresh_token.revoked = True
        await self.session.flush()
        await self.session.refresh(refresh_token)
        return refresh_token

    async def revoke_all_for_user(self, user_id: str) -> int:
        """
        Revoke all refresh tokens for a user.

        Args:
            user_id: User ID

        Returns:
            Number of tokens revoked
        """
        result = await self.session.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked == False,  # noqa: E712
            )
        )
        tokens = result.scalars().all()
        count = 0
        for token in tokens:
            token.revoked = True
            count += 1
        await self.session.flush()
        return count

    async def delete_expired(self) -> int:
        """
        Delete expired refresh tokens.

        Returns:
            Number of tokens deleted
        """
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.expires_at < now)
        )
        tokens = result.scalars().all()
        count = 0
        for token in tokens:
            await self.session.delete(token)
            count += 1
        await self.session.flush()
        return count
