# =============================================================================
# Networking Module — Input Variables
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  description = "AWS availability zone for the subnet"
  type        = string
  default     = "eu-central-1a"
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "sanad-ai"
}
