"""Authentication request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# Request Schemas


class UserRegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., min_length=8, description="Password (minimum 8 characters)"
    )
    full_name: str = Field(..., min_length=1, description="User's full name")


class UserLoginRequest(BaseModel):
    """User login request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str = Field(
        ..., description="Refresh token to exchange for new access token"
    )


# Response Schemas


class TokenResponse(BaseModel):
    """Token response with access and refresh tokens."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    """User information response."""

    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email address")
    full_name: str = Field(..., description="User's full name")
    is_active: bool = Field(..., description="Whether user is active")
    is_superuser: bool = Field(default=False, description="Whether user is superuser")
    created_at: datetime = Field(..., description="User creation timestamp")

    class Config:
        """Pydantic config."""

        from_attributes = True


class LoginResponse(BaseModel):
    """Login response with tokens and user info."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str = Field(..., description="Response message")
