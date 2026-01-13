output "external_ip" {
  description = "External IP address"
  value       = google_compute_address.static_ip.address
}

output "api_url" {
  description = "API URL"
  value       = "http://${google_compute_address.static_ip.address}:${var.api_port}"
}

output "mlflow_url" {
  description = "MLflow UI URL"
  value       = "http://${google_compute_address.static_ip.address}:${var.mlflow_port}"
}

output "ssh_command" {
  description = "SSH command"
  value       = "gcloud compute ssh ${var.vm_name} --zone=${var.zone}"
}
