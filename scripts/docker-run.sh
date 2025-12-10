#!/bin/bash
set -e

echo "🚀 Starting services with docker-compose..."
docker-compose up -d

echo "⏳ Waiting for services to be healthy..."
sleep 10

echo "📊 Services status:"
docker-compose ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🔗 Access URLs:"
echo "   API:         http://localhost:8000"
echo "   API Docs:    http://localhost:8000/docs"
echo "   MLflow UI:   http://localhost:5000"
echo "   Metrics:     http://localhost:8000/metrics"
echo ""
echo "📝 Useful commands:"
echo "   Logs:  docker-compose logs -f"
echo "   Stop:  docker-compose down"
