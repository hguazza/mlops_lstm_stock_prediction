# Terraform Setup - Simple Guide

## Install Terraform (Choose one)

### Option 1: Direct Download (Recommended)
```bash
curl -O https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_darwin_amd64.zip
unzip terraform_1.7.0_darwin_amd64.zip
sudo mv terraform /usr/local/bin/
terraform version
```

### Option 2: No sudo (Install to ~/bin)
```bash
mkdir -p ~/bin
cd ~/bin
curl -O https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_darwin_amd64.zip
unzip terraform_1.7.0_darwin_amd64.zip
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
terraform version
```

## Quick Start

```bash
# 1. Initialize Terraform
cd terraform
terraform init

# 2. Import existing resources
terraform import google_compute_address.static_ip projects/stock-prediction-lstm-api-prod/regions/us-central1/addresses/stock-api-ip
terraform import google_compute_firewall.allow_api projects/stock-prediction-lstm-api-prod/global/firewalls/allow-api
terraform import google_compute_firewall.allow_mlflow projects/stock-prediction-lstm-api-prod/global/firewalls/allow-mlflow
terraform import google_compute_instance.stock_prediction_vm projects/stock-prediction-lstm-api-prod/zones/us-central1-a/instances/stock-prediction-vm

# 3. Verify
terraform plan
```

## Usage

### Stop VM (Save ~$55/month)
```bash
terraform apply -var="vm_running=false"
```

### Start VM
```bash
terraform apply -var="vm_running=true"
```

### View URLs
```bash
terraform output
```

### Check what Terraform manages
```bash
terraform state list
```

## Using Make Commands

From project root:

```bash
make tf-stop   # Stop VM
make tf-start  # Start VM
make tf-output # Show URLs
```
