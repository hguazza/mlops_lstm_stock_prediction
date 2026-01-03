"""Security dependencies for FastAPI endpoints."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.auth.jwt import decode_token
from src.infrastructure.database.connection import get_async_session
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories import UserRepository

# HTTP Bearer token security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Get current user from JWT token.

    Args:
        credentials: HTTP Bearer token credentials
        session: Database session

    Returns:
        User instance

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    token_data = decode_token(token, expected_type="access")

    if token_data is None:
        raise credentials_exception

    user_repo = UserRepository(session)
    user = await user_repo.get_by_id(token_data.sub)

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user from token

    Returns:
        User instance

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def require_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Require current user to be a superuser.

    Args:
        current_user: Current active user

    Returns:
        User instance

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user
