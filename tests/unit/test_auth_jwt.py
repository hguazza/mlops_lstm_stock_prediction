"""Unit tests for JWT token creation and validation."""

from datetime import datetime, timedelta, timezone

from jose import jwt

from src.config import get_config
from src.domain.auth.jwt import (
    TokenData,
    create_access_token,
    create_refresh_token,
    decode_token,
)

config = get_config()


class TestJWTTokens:
    """Test JWT token functionality."""

    def test_create_access_token(self):
        """Test access token creation."""
        user_id = "test_user_123"
        token = create_access_token(user_id)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = jwt.decode(
            token, config.auth.secret_key, algorithms=[config.auth.algorithm]
        )
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user_id = "test_user_123"
        token, expires_at = create_refresh_token(user_id)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        assert isinstance(expires_at, datetime)
        assert expires_at > datetime.now(timezone.utc)

        # Decode and verify
        payload = jwt.decode(
            token, config.auth.secret_key, algorithms=[config.auth.algorithm]
        )
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "jti" in payload  # JWT ID for uniqueness

    def test_decode_access_token_valid(self):
        """Test decoding valid access token."""
        user_id = "test_user_123"
        token = create_access_token(user_id)

        token_data = decode_token(token, expected_type="access")

        assert token_data is not None
        assert isinstance(token_data, TokenData)
        assert token_data.sub == user_id
        assert token_data.type == "access"
        assert token_data.exp > datetime.now(timezone.utc)

    def test_decode_refresh_token_valid(self):
        """Test decoding valid refresh token."""
        user_id = "test_user_123"
        token, _ = create_refresh_token(user_id)

        token_data = decode_token(token, expected_type="refresh")

        assert token_data is not None
        assert isinstance(token_data, TokenData)
        assert token_data.sub == user_id
        assert token_data.type == "refresh"

    def test_decode_token_wrong_type(self):
        """Test decoding token with wrong expected type."""
        user_id = "test_user_123"
        token = create_access_token(user_id)

        # Try to decode access token as refresh token
        token_data = decode_token(token, expected_type="refresh")

        assert token_data is None

    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        invalid_token = "invalid.token.here"

        token_data = decode_token(invalid_token)

        assert token_data is None

    def test_decode_token_expired(self):
        """Test decoding expired token."""
        user_id = "test_user_123"

        # Create expired token
        expire = datetime.now(timezone.utc) - timedelta(minutes=1)
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "type": "access",
        }
        expired_token = jwt.encode(
            to_encode,
            config.auth.secret_key,
            algorithm=config.auth.algorithm,
        )

        token_data = decode_token(expired_token)

        assert token_data is None

    def test_refresh_tokens_unique(self):
        """Test that refresh tokens are unique."""
        user_id = "test_user_123"
        token1, _ = create_refresh_token(user_id)
        token2, _ = create_refresh_token(user_id)

        assert token1 != token2
