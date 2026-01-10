#!/bin/bash
set -e

echo "=== Docker Entrypoint Script Started ==="
echo "Starting Stock Prediction API..."

# Ensure data directory exists and has correct permissions
mkdir -p /data
chmod 755 /data

# Function to initialize authentication database
init_auth_db() {
    echo "Initializing authentication database..."
    echo "Database URL: $DATABASE_URL"

    python -c "
import asyncio
from src.infrastructure.database.models import Base
from src.infrastructure.database.connection import engine

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Ensure background threads/connections are closed so this process can exit
    await engine.dispose()
    print('✓ Authentication tables created successfully!')

try:
    asyncio.run(create_tables())
except Exception as e:
    print(f'⚠ Warning: Could not create auth tables: {e}')
" || true
    echo "=== init_auth_db function completed ==="
    return 0
}

# Initialize authentication database
echo "Checking authentication database..."
init_auth_db
echo "=== Auth DB initialization completed ==="

# Wait a moment to ensure MLflow server is ready
echo "Waiting for MLflow server to be ready..."
sleep 5

# Execute the main command
echo "=== About to execute main command ==="
echo "Executing command: $@"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Uvicorn location: $(which uvicorn 2>&1)"

echo "=== Calling exec ==="
exec "$@"
