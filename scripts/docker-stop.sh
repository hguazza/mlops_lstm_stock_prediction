#!/bin/bash
set -e

echo "🛑 Stopping Docker services..."
docker-compose down

echo "✅ Services stopped"
