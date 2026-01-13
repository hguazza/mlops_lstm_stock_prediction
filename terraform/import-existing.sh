#!/bin/bash
# Import existing GCP resources into Terraform state

set -e

echo "Importing existing GCP resources into Terraform..."

# Static IP
echo "1/4: Importing static IP..."
terraform import google_compute_address.static_ip \
  projects/stock-prediction-lstm-api-prod/regions/us-central1/addresses/stock-api-ip

# API Firewall Rule
echo "2/4: Importing API firewall rule..."
terraform import google_compute_firewall.allow_api \
  projects/stock-prediction-lstm-api-prod/global/firewalls/allow-api

# MLflow Firewall Rule
echo "3/4: Importing MLflow firewall rule..."
terraform import google_compute_firewall.allow_mlflow \
  projects/stock-prediction-lstm-api-prod/global/firewalls/allow-mlflow

# VM Instance
echo "4/4: Importing VM instance..."
terraform import google_compute_instance.stock_prediction_vm \
  projects/stock-prediction-lstm-api-prod/zones/us-central1-a/instances/stock-prediction-vm

echo ""
echo "✅ Import complete! All resources are now managed by Terraform."
echo ""
echo "Next steps:"
echo "  terraform plan   # Verify no changes"
echo "  terraform output # View service URLs"
