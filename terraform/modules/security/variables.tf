# =============================================================================
# Security Module — Input Variables
# =============================================================================

variable "vpc_id" {
  description = "ID of the VPC to create the security group in"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "sanad-ai"
}
