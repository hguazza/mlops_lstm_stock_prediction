# Terraform Infrastructure Management Guide

This guide explains how to use Terraform to manage your Stock Prediction API infrastructure on Google Cloud Platform.

## Why Terraform?

Terraform allows you to:
- **Start/Stop VM easily** - Save ~$45/month when not in use
- **Infrastructure as Code** - Track all changes in version control
- **Reproducible** - Recreate infrastructure from code
- **Safe changes** - Preview changes before applying
- **Team collaboration** - Share infrastructure state

## Quick Start

### 1. Install Terraform

**macOS:**
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

**Linux:**
```bash
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### 2. Import Existing Infrastructure

Run the quick-start script:

```bash
cd terraform
./quick-start.sh
```

This will:
1. Initialize Terraform
2. Import your existing VM, firewall rules, and static IP
3. Verify everything matches

### 3. Verify Setup

```bash
make tf-plan
```

Should show "No changes" if everything is imported correctly.

## Daily Usage

### Start/Stop VM (Most Common)

**Stop VM to save costs:**
```bash
make tf-stop
```

**Start VM when needed:**
```bash
make tf-start
```

The VM will automatically start all Docker services via the startup script when started.

### View Service Information

```bash
make tf-output
```

This shows:
- External IP address
- API URL
- MLflow UI URL
- Health check URL
- SSH command

### Check VM Status

```bash
make gcp-status
```

### SSH into VM

```bash
make gcp-ssh
```

## Making Infrastructure Changes

### Change VM Size

1. Stop the VM first:
   ```bash
   make tf-stop
   ```

2. Edit `terraform/terraform.tfvars`:
   ```hcl
   machine_type = "e2-standard-4"  # Upgrade
   ```

3. Apply changes:
   ```bash
   make tf-apply
   ```

4. Start VM:
   ```bash
   make tf-start
   ```

### Increase Disk Size

Edit `terraform/terraform.tfvars`:
```hcl
boot_disk_size = 100  # Increase to 100GB
```

Apply:
```bash
make tf-apply
```

### Change Firewall Rules

Edit `terraform/main.tf` to modify firewall rules, then:

```bash
make tf-plan   # Preview changes
make tf-apply  # Apply changes
```

## Cost Savings

### Current Costs

```bash
make gcp-cost-estimate
```

**Output:**
```
Current Infrastructure Costs (Approximate):
  VM (e2-standard-2): ~$50/month (RUNNING)
  Boot Disk (50GB):   ~$2/month
  Static IP:          ~$3/month
  Traffic:            ~$5-20/month
  --------------------------------
  Total:              ~$60-75/month

💡 To save costs: 'make tf-stop' when not in use (~$45/month savings)
```

### Stop During Weekends

If you only use the API during weekdays:

**Friday evening:**
```bash
make tf-stop
```

**Monday morning:**
```bash
make tf-start
```

**Savings:** ~$25/month (stopping 2 days/week)

### Automate with Cron

Stop VM at 6 PM daily:
```bash
0 18 * * * cd /path/to/project && make tf-stop
```

Start VM at 8 AM daily:
```bash
0 8 * * * cd /path/to/project && make tf-start
```

## All Terraform Commands

### Basic Commands

```bash
make tf-init          # Initialize Terraform (first time)
make tf-plan          # Preview changes
make tf-apply         # Apply changes
make tf-output        # Show outputs
make tf-show          # Show current state
```

### VM Control

```bash
make tf-stop          # Stop VM (save costs)
make tf-start         # Start VM
```

### State Management

```bash
make tf-state-list    # List all resources
make tf-refresh       # Sync state with reality
make tf-validate      # Validate configuration
make tf-fmt           # Format Terraform files
```

### Advanced

```bash
make tf-import        # Import existing resources
make tf-graph         # Generate infrastructure diagram
make tf-destroy       # Destroy everything (careful!)
```

## GCP Commands

### Check Services

```bash
make gcp-status       # VM status
make gcp-health       # Check API and MLflow health
```

### Access VM

```bash
make gcp-ssh          # SSH into VM
make gcp-logs         # View Docker logs
make gcp-restart-services  # Restart Docker Compose
```

## Troubleshooting

### VM Won't Stop

```bash
# Force stop
gcloud compute instances stop stock-prediction-vm --zone=us-central1-a

# Refresh Terraform state
make tf-refresh
```

### Changes Not Applying

```bash
# See detailed plan
make tf-plan

# Refresh state first
make tf-refresh

# Then apply
make tf-apply
```

### Import Failed

If the quick-start script fails:

```bash
cd terraform

# Check what's already in state
terraform state list

# Remove problematic resource
terraform state rm google_compute_instance.stock_prediction_vm

# Try importing again
./import-existing.sh
```

### State Locked

If you see "state lock" errors:

```bash
cd terraform
terraform force-unlock <LOCK_ID>
```

## Best Practices

### 1. Always Preview Changes

```bash
make tf-plan  # Review before applying
```

### 2. Use Version Control

Commit Terraform changes:
```bash
git add terraform/
git commit -m "Update VM configuration"
```

### 3. Stop When Not Using

Remember to stop the VM when not needed:
```bash
make tf-stop
```

### 4. Monitor Costs

Check GCP console regularly:
```
https://console.cloud.google.com/billing
```

### 5. Backup Before Major Changes

```bash
cd terraform
cp terraform.tfstate terraform.tfstate.backup
```

## Architecture Overview

Your infrastructure consists of:

```
┌─────────────────────────────────────────┐
│ GCP Project: stock-prediction-lstm-api  │
│                                          │
│  ┌────────────────────────────────┐     │
│  │ Compute Engine VM              │     │
│  │ - Type: e2-standard-2          │     │
│  │ - Disk: 50GB                   │     │
│  │ - OS: Ubuntu 22.04             │     │
│  │                                │     │
│  │   ┌──────────────────────┐     │     │
│  │   │ Docker Compose       │     │     │
│  │   │ - PostgreSQL         │     │     │
│  │   │ - MLflow Server      │     │     │
│  │   │ - Stock API          │     │     │
│  │   └──────────────────────┘     │     │
│  └────────────────────────────────┘     │
│            ↕                             │
│  ┌────────────────────────────────┐     │
│  │ Static IP: 34.61.75.148        │     │
│  └────────────────────────────────┘     │
│            ↕                             │
│  ┌────────────────────────────────┐     │
│  │ Firewall Rules                 │     │
│  │ - allow-api (8001)             │     │
│  │ - allow-mlflow (5002)          │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
```

## Remote State (Optional)

For team collaboration, store state in GCS:

### 1. Create Bucket

```bash
gsutil mb -p stock-prediction-lstm-api-prod -l us-central1 gs://stock-api-terraform-state
gsutil versioning set on gs://stock-api-terraform-state
```

### 2. Update Configuration

Edit `terraform/versions.tf`:
```hcl
backend "gcs" {
  bucket = "stock-api-terraform-state"
  prefix = "terraform/state"
}
```

### 3. Migrate State

```bash
cd terraform
terraform init -migrate-state
```

## Common Scenarios

### Scenario 1: Weekly Development

**Monday AM:**
```bash
make tf-start
# Wait 2 minutes for services to start
make gcp-health
```

**Friday PM:**
```bash
make tf-stop
```

**Savings:** ~$25/month

### Scenario 2: Testing New Features

**Before changes:**
```bash
make tf-plan  # See current state
```

**Make changes in terraform.tfvars**

**Preview and apply:**
```bash
make tf-plan
make tf-apply
```

**Verify:**
```bash
make gcp-health
```

### Scenario 3: Emergency Stop

**If something goes wrong:**
```bash
make tf-stop  # Immediate stop
```

**Or via gcloud:**
```bash
gcloud compute instances stop stock-prediction-vm --zone=us-central1-a
```

### Scenario 4: Scale Up for Load Test

**Upgrade VM:**
```bash
make tf-stop
# Edit terraform.tfvars: machine_type = "e2-standard-4"
make tf-apply
make tf-start
```

**Downgrade after test:**
```bash
make tf-stop
# Edit terraform.tfvars: machine_type = "e2-standard-2"
make tf-apply
make tf-start
```

## Summary

**Most Common Commands:**

| Task | Command |
|------|---------|
| Stop VM | `make tf-stop` |
| Start VM | `make tf-start` |
| View URLs | `make tf-output` |
| Check status | `make gcp-status` |
| Check health | `make gcp-health` |
| SSH to VM | `make gcp-ssh` |
| Preview changes | `make tf-plan` |
| Apply changes | `make tf-apply` |

**Remember:**
- Always `make tf-plan` before `make tf-apply`
- Stop VM when not in use to save ~$45/month
- Commit Terraform changes to version control
- Keep `.env.production` secure and never commit it

## Additional Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [Google Cloud Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GCP Compute Engine](https://cloud.google.com/compute/docs)
- [Full Terraform README](../terraform/README.md)
