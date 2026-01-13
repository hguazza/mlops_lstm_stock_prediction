variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "stock-prediction-lstm-api-prod"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "vm_name" {
  description = "Name of the VM instance"
  type        = string
  default     = "stock-prediction-vm"
}

variable "machine_type" {
  description = "Machine type for the VM"
  type        = string
  default     = "e2-standard-2"
}

variable "boot_disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "boot_disk_type" {
  description = "Boot disk type"
  type        = string
  default     = "pd-standard"
}

variable "image_family" {
  description = "Image family for the VM"
  type        = string
  default     = "ubuntu-2204-lts"
}

variable "image_project" {
  description = "Project containing the image"
  type        = string
  default     = "ubuntu-os-cloud"
}

variable "network_name" {
  description = "VPC network name"
  type        = string
  default     = "default"
}

variable "static_ip_name" {
  description = "Name of the static IP address"
  type        = string
  default     = "stock-api-ip"
}

variable "allow_api_firewall_name" {
  description = "Name of the API firewall rule"
  type        = string
  default     = "allow-api"
}

variable "allow_mlflow_firewall_name" {
  description = "Name of the MLflow firewall rule"
  type        = string
  default     = "allow-mlflow"
}

variable "api_port" {
  description = "Port for API access"
  type        = string
  default     = "8001"
}

variable "mlflow_port" {
  description = "Port for MLflow UI access"
  type        = string
  default     = "5002"
}

variable "network_tags" {
  description = "Network tags for the VM"
  type        = list(string)
  default     = ["api-server", "http-server", "https-server"]
}

variable "startup_script_path" {
  description = "Path to the startup script"
  type        = string
  default     = "../scripts/gcp-startup.sh"
}

variable "vm_running" {
  description = "Whether the VM should be running (true) or stopped (false)"
  type        = bool
  default     = true
}
