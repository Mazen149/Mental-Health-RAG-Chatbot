# =============================================================================
# Compute Module — Input Variables
# =============================================================================

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "key_pair_name" {
  description = "Name of the AWS key pair for SSH access"
  type        = string
}

variable "subnet_id" {
  description = "ID of the subnet to launch the instance in"
  type        = string
}

variable "security_group_id" {
  description = "ID of the security group to attach"
  type        = string
}

variable "docker_image" {
  description = "Docker image to pull and run"
  type        = string
  default     = "mazen1393/sanad-ai-backend:latest"
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "sanad-ai"
}
