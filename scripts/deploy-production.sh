#!/bin/bash

##############################################################################
# Production Deployment Script
#
# This script automates the deployment of the Stock Prediction API in
# production mode with PostgreSQL backend.
#
# Usage: ./scripts/deploy-production.sh [command]
# Commands: start, stop, restart, status, logs, health
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Stock Prediction API - Production${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi

    # Check if .env.production exists
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env.production not found. Creating from template..."
        cat > "$ENV_FILE" << 'EOF'
# Production Environment Variables
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=CHANGE_THIS_SECURE_PASSWORD
POSTGRES_DB=mlflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_ALLOWED_HOSTS=*
API_WORKERS=4
LOG_LEVEL=info
EOF
        print_warning "Please edit $ENV_FILE and set secure passwords!"
        print_info "Then run this script again."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

start_services() {
    print_header
    check_prerequisites

    print_info "Starting production services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    print_success "Services started successfully"
    print_info "Waiting for services to be healthy (this may take 1-2 minutes)..."

    # Wait for services to be healthy
    sleep 10

    check_health

    print_success "Deployment complete!"
    print_info "API: http://localhost:8000"
    print_info "MLflow UI: http://localhost:5000"
    print_info "API Docs: http://localhost:8000/docs"
}

stop_services() {
    print_header
    print_info "Stopping production services..."
    docker-compose -f "$COMPOSE_FILE" down
    print_success "Services stopped"
}

restart_services() {
    print_header
    print_info "Restarting production services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
    print_success "Services restarted"
    sleep 5
    check_health
}

check_status() {
    print_header
    print_info "Checking service status..."
    docker-compose -f "$COMPOSE_FILE" ps
}

show_logs() {
    print_header
    print_info "Showing logs (Ctrl+C to exit)..."
    docker-compose -f "$COMPOSE_FILE" logs -f
}

check_health() {
    print_info "Checking health endpoints..."

    # Check API health
    if curl -s -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_error "API is not responding"
    fi

    # Check MLflow health
    if curl -s -f http://localhost:5000/health > /dev/null 2>&1; then
        print_success "MLflow is healthy"
    else
        print_warning "MLflow might still be starting up"
    fi

    # Check PostgreSQL
    if docker exec mlflow-postgres pg_isready -U mlflow > /dev/null 2>&1; then
        print_success "PostgreSQL is healthy"
    else
        print_error "PostgreSQL is not responding"
    fi
}

backup_database() {
    print_header
    BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
    print_info "Creating database backup: $BACKUP_FILE"

    docker exec mlflow-postgres pg_dump -U mlflow mlflow > "$BACKUP_FILE"

    if [ -f "$BACKUP_FILE" ]; then
        print_success "Backup created: $BACKUP_FILE"
    else
        print_error "Backup failed"
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: $0 [command]

Commands:
    start       Start production services
    stop        Stop production services
    restart     Restart production services
    status      Show service status
    logs        Show service logs
    health      Check service health
    backup      Backup PostgreSQL database
    help        Show this help message

Examples:
    $0 start        # Start all services
    $0 health       # Check if services are healthy
    $0 logs         # View logs
    $0 backup       # Create database backup

EOF
}

# Main script logic
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    health)
        check_health
        ;;
    backup)
        backup_database
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Invalid command: $1"
        show_usage
        exit 1
        ;;
esac
