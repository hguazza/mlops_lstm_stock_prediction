"""Integration tests for authentication endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.auth.password import hash_password
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories import UserRepository


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user_repo = UserRepository(db_session)

    # Delete existing test user if exists
    existing_user = await user_repo.get_by_email("test@example.com")
    if existing_user:
        await user_repo.delete(existing_user)

    user = await user_repo.create(
        email="test@example.com",
        hashed_password=hash_password("testpassword123"),
        full_name="Test User",
    )
    await db_session.commit()
    return user


@pytest.mark.integration
class TestAuthEndpoints:
    """Test authentication API endpoints."""

    @pytest.mark.asyncio
    async def test_register_user(self, client: AsyncClient):
        """Test user registration endpoint."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "password123",
                "full_name": "New User",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert data["is_active"] is True
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user: User):
        """Test registration with duplicate email."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "password123",
                "full_name": "Duplicate User",
            },
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient, test_user: User):
        """Test successful login."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == test_user.email

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient, test_user: User):
        """Test login with wrong password."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword",
            },
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with nonexistent user."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user(self, client: AsyncClient, test_user: User):
        """Test getting current user info."""
        # Login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        access_token = login_response.json()["access_token"]

        # Get current user
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["full_name"] == test_user.full_name

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, client: AsyncClient):
        """Test getting current user without token."""
        response = await client.get("/api/v1/auth/me")

        assert response.status_code == 401  # No credentials

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, client: AsyncClient):
        """Test getting current user with invalid token."""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, client: AsyncClient, test_user: User):
        """Test token refresh."""
        # Login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, client: AsyncClient):
        """Test refresh with invalid token."""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_token"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_logout(self, client: AsyncClient, test_user: User):
        """Test logout (revoke refresh token)."""
        # Login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        access_token = login_response.json()["access_token"]
        refresh_token = login_response.json()["refresh_token"]

        # Logout
        response = await client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh_token},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()

        # Try to use revoked refresh token
        refresh_response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert refresh_response.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self, client: AsyncClient):
        """Test that protected endpoints require authentication."""
        response = await client.get("/api/v1/health")

        assert response.status_code == 401  # No credentials provided

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_auth(
        self, client: AsyncClient, test_user: User
    ):
        """Test accessing protected endpoint with authentication."""
        # Login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        access_token = login_response.json()["access_token"]

        # Access protected endpoint
        response = await client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
