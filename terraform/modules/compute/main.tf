# =============================================================================
# Compute Module — EC2 Instance with Docker Bootstrap
# =============================================================================

# Fetch the latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  # 20 GB root volume (free-tier allows up to 30 GB)
  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail

    # ── System Updates ──
    dnf update -y

    # ── Install Docker ──
    dnf install -y docker
    systemctl enable docker
    systemctl start docker

    # Add ec2-user to docker group so SSH deploys don't need sudo
    usermod -aG docker ec2-user

    # ── Create 2 GB Swap File (critical for ML model loading on limited RAM) ──
    dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' >> /etc/fstab

    # ── Pull & Run the Application Container ──
    docker pull ${var.docker_image}
    docker run -d \
      --name sanad-ai \
      -p 8000:8000 \
      --restart unless-stopped \
      ${var.docker_image}

  EOF

  tags = {
    Name = "${var.project_name}-ec2"
  }
}
