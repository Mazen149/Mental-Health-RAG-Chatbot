# =============================================================================
# Sanad AI — Root Terraform Configuration
# =============================================================================
# Provisions a minimal-cost AWS infrastructure to run the Sanad AI chatbot
# as a Docker container on a single EC2 instance.
# =============================================================================

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Networking: VPC, Subnet, IGW, Route Table ──
module "networking" {
  source = "./modules/networking"

  vpc_cidr          = "10.0.0.0/16"
  subnet_cidr       = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"
  project_name      = var.project_name
}

# ── Security: Security Group ──
module "security" {
  source = "./modules/security"

  vpc_id       = module.networking.vpc_id
  project_name = var.project_name
}

# ── Compute: EC2 Instance with Docker ──
module "compute" {
  source = "./modules/compute"

  instance_type     = var.instance_type
  key_pair_name     = var.key_pair_name
  subnet_id         = module.networking.subnet_id
  security_group_id = module.security.security_group_id
  docker_image      = var.docker_image
  project_name      = var.project_name
}
