#!/bin/bash
# GCP VM Startup Script
# This script automatically starts Docker containers on VM reboot

set -e

# Wait for Docker to be available
until docker info > /dev/null 2>&1; do
  echo "Waiting for Docker to be ready..."
  sleep 2
done

# Change to application directory
APP_DIR="/home/$(whoami)/app"
if [ -d "$APP_DIR" ]; then
  cd "$APP_DIR"
  
  # Start Docker Compose services
  echo "Starting Docker Compose services..."
  docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
  
  echo "Services started successfully!"
  docker-compose -f docker-compose.prod.yml ps
else
  echo "Application directory not found at $APP_DIR"
  exit 1
fi
