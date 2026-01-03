"""JWT token creation and validation utilities."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import get_config

config = get_config()


class TokenData(BaseModel):
    """Token payload data model."""

    sub: str  # Subject (user ID)
    exp: datetime  # Expiration time
    type: str  # Token type: "access" or "refresh"


def create_access_token(user_id: str) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User ID to encode in token

    Returns:
        Encoded JWT token string
    """
    expires_delta = timedelta(minutes=config.auth.access_token_expire_minutes)
    expire = datetime.now(timezone.utc) + expires_delta

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "access",
    }

    encoded_jwt = jwt.encode(
        to_encode,
        config.auth.secret_key,
        algorithm=config.auth.algorithm,
    )
    return encoded_jwt


def create_refresh_token(user_id: str) -> tuple[str, datetime]:
    """
    Create a JWT refresh token.

    Args:
        user_id: User ID to encode in token

    Returns:
        Tuple of (encoded JWT token string, expiration datetime)
    """
    expires_delta = timedelta(days=config.auth.refresh_token_expire_days)
    expire = datetime.now(timezone.utc) + expires_delta

    # Add random jti (JWT ID) to make each refresh token unique
    jti = secrets.token_hex(16)

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
        "jti": jti,
    }

    encoded_jwt = jwt.encode(
        to_encode,
        config.auth.secret_key,
        algorithm=config.auth.algorithm,
    )
    return encoded_jwt, expire


def decode_token(token: str, expected_type: str = "access") -> Optional[TokenData]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string to decode
        expected_type: Expected token type ("access" or "refresh")

    Returns:
        TokenData if valid, None if invalid

    Raises:
        JWTError: If token is invalid
    """
    try:
        payload = jwt.decode(
            token,
            config.auth.secret_key,
            algorithms=[config.auth.algorithm],
        )

        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        exp: int = payload.get("exp")

        if user_id is None or token_type is None or exp is None:
            return None

        if token_type != expected_type:
            return None

        # Convert exp to datetime
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)

        return TokenData(
            sub=user_id,
            exp=exp_datetime,
            type=token_type,
        )

    except JWTError:
        return None
