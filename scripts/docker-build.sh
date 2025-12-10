#!/bin/bash
set -e

echo "🐳 Building Docker image..."
docker build -t stock-prediction-api:latest .

echo "✅ Build complete!"
docker images | grep stock-prediction-api
