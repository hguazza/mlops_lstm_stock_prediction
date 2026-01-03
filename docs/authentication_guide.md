# Authentication Guide

This guide explains how to use JWT authentication in the Stock Price Prediction API.

## Overview

The API uses JWT (JSON Web Tokens) for authentication with the following features:

- **User Registration**: Create new user accounts
- **Login**: Authenticate and receive access/refresh tokens
- **Token Refresh**: Renew expired access tokens
- **Protected Endpoints**: All API endpoints require authentication
- **Token Revocation**: Logout to revoke refresh tokens

## Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Database

    Client->>API: POST /auth/register (email, password)
    API->>Database: Create user
    API-->>Client: User created (201)

    Client->>API: POST /auth/login (email, password)
    API->>Database: Validate credentials
    API-->>Client: access_token + refresh_token

    Client->>API: GET /health (Bearer token)
    API->>API: Validate JWT
    API-->>Client: Protected resource

    Note over Client,API: Access token expires after 30 minutes

    Client->>API: POST /auth/refresh (refresh_token)
    API->>Database: Validate refresh token
    API-->>Client: New access_token

    Client->>API: POST /auth/logout (refresh_token)
    API->>Database: Revoke refresh token
    API-->>Client: Logged out
```

## Getting Started

### 1. Register a New User

Create a new user account:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2025-01-02T10:30:00Z"
}
```

### 2. Login

Authenticate and receive tokens:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true,
    "is_superuser": false,
    "created_at": "2025-01-02T10:30:00Z"
  }
}
```

### 3. Access Protected Endpoints

Use the access token in the `Authorization` header:

```bash
curl -X GET "http://localhost:8000/api/v1/health" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 4. Refresh Access Token

When the access token expires (after 30 minutes), use the refresh token to get a new one:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 5. Get Current User Info

Retrieve information about the currently authenticated user:

```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 6. Logout

Revoke the refresh token to logout:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/logout" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

## Using the Interactive API Documentation

The API includes Swagger UI for easy testing:

1. Navigate to `http://localhost:8000/docs`
2. Click on the **Authorize** button (lock icon)
3. Enter your access token in the format: `Bearer <your_token>`
4. Click **Authorize**
5. Now you can test all endpoints directly from the browser

## Token Lifetimes

- **Access Token**: 30 minutes (configurable via `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`)
- **Refresh Token**: 7 days (configurable via `JWT_REFRESH_TOKEN_EXPIRE_DAYS`)

## Python Client Example

Here's a complete example using Python's `requests` library:

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Register
response = requests.post(
    f"{BASE_URL}/auth/register",
    json={
        "email": "user@example.com",
        "password": "securepassword123",
        "full_name": "John Doe"
    }
)
print("User registered:", response.json())

# 2. Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    json={
        "email": "user@example.com",
        "password": "securepassword123"
    }
)
tokens = response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]

# 3. Access protected endpoint
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{BASE_URL}/health", headers=headers)
print("Health check:", response.json())

# 4. Train a model (protected endpoint)
response = requests.post(
    f"{BASE_URL}/train",
    headers=headers,
    json={
        "symbol": "AAPL",
        "period": "1y"
    }
)
print("Training result:", response.json())

# 5. Refresh token when needed
response = requests.post(
    f"{BASE_URL}/auth/refresh",
    json={"refresh_token": refresh_token}
)
new_tokens = response.json()
access_token = new_tokens["access_token"]

# 6. Logout
response = requests.post(
    f"{BASE_URL}/auth/logout",
    headers={"Authorization": f"Bearer {access_token}"},
    json={"refresh_token": refresh_token}
)
print("Logged out:", response.json())
```

## Error Handling

### Common Error Responses

**401 Unauthorized** - Invalid or expired token:
```json
{
  "detail": "Could not validate credentials"
}
```

**403 Forbidden** - User is inactive or insufficient permissions:
```json
{
  "detail": "Inactive user"
}
```

**400 Bad Request** - Invalid input (e.g., duplicate email):
```json
{
  "detail": "Email already registered"
}
```

### Handling Token Expiration

When you receive a 401 error, try refreshing the token:

```python
def make_authenticated_request(url, access_token, refresh_token):
    """Make request with automatic token refresh."""
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 401:
        # Token expired, refresh it
        refresh_response = requests.post(
            f"{BASE_URL}/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        if refresh_response.status_code == 200:
            new_tokens = refresh_response.json()
            access_token = new_tokens["access_token"]

            # Retry original request
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(url, headers=headers)

    return response, access_token
```

## Security Best Practices

### 1. Secure Token Storage

**Never store tokens in:**
- Local storage (vulnerable to XSS)
- URL parameters
- Unencrypted cookies

**Recommended storage:**
- HTTP-only, secure cookies (for web apps)
- Secure storage APIs (for mobile apps)
- Environment variables (for server-to-server)

### 2. Token Transmission

- Always use HTTPS in production
- Never log tokens
- Don't include tokens in URLs

### 3. Password Requirements

- Minimum 8 characters
- Use strong, unique passwords
- Consider implementing password complexity rules

### 4. Environment Variables

Set these environment variables in production:

```bash
# Generate a secure secret key
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Set token expiration
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Database configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/db
```

### 5. Rate Limiting

Consider implementing rate limiting for authentication endpoints to prevent brute force attacks.

## Database Setup

### Development (SQLite)

SQLite is used by default for development:

```bash
# Set environment variable
export DATABASE_URL=sqlite+aiosqlite:///./auth.db

# Run migrations
python scripts/init_db.py
```

### Production (PostgreSQL)

For production, use PostgreSQL:

```bash
# Set environment variables
export DATABASE_URL=postgresql+asyncpg://mlops_user:password@localhost:5432/stock_prediction
export DB_PASSWORD=your_secure_password

# Run migrations
python scripts/init_db.py
```

## Admin User

The initialization script creates an admin user with credentials from environment variables:

```bash
export ADMIN_EMAIL=admin@example.com
export ADMIN_PASSWORD=changeme123
export ADMIN_FULL_NAME="Admin User"

python scripts/init_db.py
```

**Important**: Change the admin password after first login!

## Troubleshooting

### "Could not validate credentials"

- Check that the token is valid and not expired
- Ensure the token is properly formatted in the Authorization header
- Verify the JWT_SECRET_KEY hasn't changed

### "Email already registered"

- The email is already in use
- Try logging in instead of registering
- Use a different email address

### "Incorrect email or password"

- Verify credentials are correct
- Check for typos in email/password
- Ensure user account exists

### Database Connection Errors

- Verify DATABASE_URL is set correctly
- Check database is running (PostgreSQL in production)
- Run migrations: `python scripts/init_db.py`

## API Reference

### Authentication Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/v1/auth/register` | POST | Register new user | No |
| `/api/v1/auth/login` | POST | Login and get tokens | No |
| `/api/v1/auth/refresh` | POST | Refresh access token | No |
| `/api/v1/auth/logout` | POST | Revoke refresh token | Yes |
| `/api/v1/auth/me` | GET | Get current user info | Yes |

### Protected Endpoints

All other API endpoints require authentication:

- `/api/v1/health` - Health check
- `/api/v1/train` - Train models
- `/api/v1/predict` - Make predictions
- `/api/v1/multivariate` - Multivariate predictions
- `/api/v1/models` - Model registry
- `/metrics` - Prometheus metrics

## Support

For issues or questions:
1. Check this documentation
2. Review the API documentation at `/docs`
3. Check application logs for error details
4. Consult the project README
