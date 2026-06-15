# =============================================================================
# Root Module — Outputs
# =============================================================================

output "ec2_public_ip" {
  description = "Public IP address of the Sanad AI EC2 instance"
  value       = module.compute.public_ip
}

output "ec2_instance_id" {
  description = "Instance ID of the EC2 instance"
  value       = module.compute.instance_id
}

output "app_url" {
  description = "URL to access the Sanad AI application"
  value       = "http://${module.compute.public_ip}:8000"
}
