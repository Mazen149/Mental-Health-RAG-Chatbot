# =============================================================================
# Root Module — Input Variables
# =============================================================================

variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "sanad-ai"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "Name of the AWS key pair for SSH access"
  type        = string
}

variable "docker_image" {
  description = "Docker Hub image to deploy"
  type        = string
  default     = "mazen1393/sanad-ai-backend:latest"
}
