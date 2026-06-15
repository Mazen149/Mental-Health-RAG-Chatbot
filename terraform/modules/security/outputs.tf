# =============================================================================
# Security Module — Outputs
# =============================================================================

output "security_group_id" {
  description = "ID of the application security group"
  value       = aws_security_group.app.id
}
