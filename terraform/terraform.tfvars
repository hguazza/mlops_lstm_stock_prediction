# GCP Project Configuration
project_id = "stock-prediction-lstm-api-prod"
region     = "us-central1"
zone       = "us-central1-a"

# VM Configuration
vm_name         = "stock-prediction-vm"
machine_type    = "e2-standard-2"
boot_disk_size  = 50
boot_disk_type  = "pd-standard"
image_family    = "ubuntu-2204-lts"
image_project   = "ubuntu-os-cloud"

# Network Configuration
network_name   = "default"
static_ip_name = "stock-api-ip"

# Firewall Configuration
allow_api_firewall_name    = "allow-api"
allow_mlflow_firewall_name = "allow-mlflow"
api_port                   = "8001"
mlflow_port                = "5002"

# VM Tags
network_tags = ["api-server", "http-server", "https-server"]

# Startup Script Path
startup_script_path = "../scripts/gcp-startup.sh"

# VM State Control - Set to false to stop the VM, true to start it
vm_running = true
