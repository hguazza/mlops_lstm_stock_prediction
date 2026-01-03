"""Authentication router - User registration and login endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.domain.auth.jwt import create_access_token, create_refresh_token, decode_token
from src.domain.auth.password import hash_password, verify_password
from src.domain.auth.security import get_current_active_user
from src.infrastructure.database.connection import get_async_session
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories import (
    RefreshTokenRepository,
    UserRepository,
)
from src.presentation.schemas.auth import (
    LoginResponse,
    MessageResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = structlog.get_logger(__name__)
config = get_config()


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    request: UserRegisterRequest,
    session: AsyncSession = Depends(get_async_session),
) -> UserResponse:
    """
    Register a new user.

    Creates a new user account with email and password.

    Args:
        request: User registration data
        session: Database session

    Returns:
        UserResponse with created user information

    Raises:
        HTTPException: If email already exists
    """
    logger.info("user_registration_attempt", email=request.email)

    user_repo = UserRepository(session)

    # Check if user already exists
    existing_user = await user_repo.get_by_email(request.email)
    if existing_user:
        logger.warning("registration_failed_email_exists", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Hash password
    hashed_password = hash_password(request.password)

    # Create user
    user = await user_repo.create(
        email=request.email,
        hashed_password=hashed_password,
        full_name=request.full_name,
    )

    logger.info("user_registered", user_id=user.id, email=user.email)

    return UserResponse.model_validate(user)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: UserLoginRequest,
    session: AsyncSession = Depends(get_async_session),
) -> LoginResponse:
    """
    Login user and return tokens.

    Authenticates user with email and password, returns access and refresh tokens.

    Args:
        request: User login credentials
        session: Database session

    Returns:
        LoginResponse with tokens and user information

    Raises:
        HTTPException: If credentials are invalid
    """
    logger.info("user_login_attempt", email=request.email)

    user_repo = UserRepository(session)
    token_repo = RefreshTokenRepository(session)

    # Get user by email
    user = await user_repo.get_by_email(request.email)
    if not user:
        logger.warning("login_failed_user_not_found", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Verify password
    if not verify_password(request.password, user.hashed_password):
        logger.warning("login_failed_invalid_password", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Check if user is active
    if not user.is_active:
        logger.warning("login_failed_inactive_user", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    # Create tokens
    access_token = create_access_token(user.id)
    refresh_token, refresh_expires_at = create_refresh_token(user.id)

    # Store refresh token in database
    await token_repo.create(
        token=refresh_token,
        user_id=user.id,
        expires_at=refresh_expires_at,
    )

    logger.info("user_logged_in", user_id=user.id, email=user.email)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=config.auth.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Exchanges a valid refresh token for a new access token.

    Args:
        request: Refresh token request
        session: Database session

    Returns:
        TokenResponse with new access token

    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    logger.info("token_refresh_attempt")

    token_repo = RefreshTokenRepository(session)

    # Decode and validate refresh token
    token_data = decode_token(request.refresh_token, expected_type="refresh")
    if not token_data:
        logger.warning("token_refresh_failed_invalid_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # Check if refresh token exists in database and is valid
    stored_token = await token_repo.get_by_token(request.refresh_token)
    if not stored_token or not stored_token.is_valid():
        logger.warning("token_refresh_failed_token_not_found_or_expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or revoked",
        )

    # Create new access token
    access_token = create_access_token(token_data.sub)

    logger.info("token_refreshed", user_id=token_data.sub)

    # Calculate expires_in
    expires_in = config.auth.access_token_expire_minutes * 60
    logger.info(
        "creating_token_response", expires_in=expires_in, type=type(expires_in).__name__
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,  # Return same refresh token
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_active_user),
) -> MessageResponse:
    """
    Logout user by revoking refresh token.

    Revokes the provided refresh token, preventing future token refreshes.

    Args:
        request: Refresh token to revoke
        session: Database session
        current_user: Current authenticated user

    Returns:
        MessageResponse confirming logout

    Raises:
        HTTPException: If refresh token not found
    """
    logger.info("user_logout_attempt", user_id=current_user.id)

    token_repo = RefreshTokenRepository(session)

    # Find and revoke refresh token
    stored_token = await token_repo.get_by_token(request.refresh_token)
    if not stored_token:
        logger.warning("logout_failed_token_not_found", user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Refresh token not found",
        )

    # Revoke token
    await token_repo.revoke(stored_token)

    logger.info("user_logged_out", user_id=current_user.id)

    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """
    Get current user information.

    Returns information about the currently authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        UserResponse with user information
    """
    logger.info("get_current_user_info", user_id=current_user.id)

    return UserResponse.model_validate(current_user)
