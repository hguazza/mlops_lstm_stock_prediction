# Static IP Address
resource "google_compute_address" "static_ip" {
  name         = var.static_ip_name
  region       = var.region
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
}

# Firewall Rule - API Access
resource "google_compute_firewall" "allow_api" {
  name    = var.allow_api_firewall_name
  network = var.network_name

  allow {
    protocol = "tcp"
    ports    = [var.api_port]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["api-server"]
  description   = "Allow API access"
}

# Firewall Rule - MLflow UI Access
resource "google_compute_firewall" "allow_mlflow" {
  name    = var.allow_mlflow_firewall_name
  network = var.network_name

  allow {
    protocol = "tcp"
    ports    = [var.mlflow_port]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["api-server"]
  description   = "Allow MLflow UI access"
}

# Compute Engine Instance
resource "google_compute_instance" "stock_prediction_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = var.network_tags

  boot_disk {
    auto_delete = true
    initialize_params {
      image = "${var.image_project}/${var.image_family}"
      size  = var.boot_disk_size
      type  = var.boot_disk_type
    }
  }

  network_interface {
    network = var.network_name
    access_config {
      nat_ip       = google_compute_address.static_ip.address
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    startup-script = file("${path.module}/${var.startup_script_path}")
  }

  service_account {
    email = "default"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/pubsub",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append"
    ]
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  # Control VM power state - change this to stop/start VM
  desired_status = var.vm_running ? "RUNNING" : "TERMINATED"

  lifecycle {
    ignore_changes = [
      metadata["ssh-keys"],
      service_account[0].email
    ]
  }
}
