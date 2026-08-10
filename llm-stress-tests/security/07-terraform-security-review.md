# Terraform Security Review

You are given a Terraform configuration for an AWS infrastructure. It contains **8 security misconfigurations**.

Find every one. For each, state the resource, the vulnerability, and the corrected HCL.

```hcl
provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
}

# Public subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

# Security group — overly permissive
resource "aws_security_group" "web" {
  name        = "web-sg"
  description = "Allow all inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# S3 bucket — public and unencrypted
resource "aws_s3_bucket" "data" {
  bucket = "my-app-data-bucket"
  acl    = "public-read"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# RDS instance — hardcoded credentials, public
resource "aws_db_instance" "main" {
  identifier           = "app-database"
  engine               = "postgres"
  engine_version       = "14.7"
  instance_class       = "db.t3.medium"
  allocated_storage    = 20
  username             = "admin"
  password             = "SuperSecret123!"
  publicly_accessible  = true
  storage_encrypted    = false
  skip_final_snapshot  = true
}

# IAM role — overly permissive
resource "aws_iam_role" "app_role" {
  name = "app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "app_policy" {
  name = "app-policy"
  role = aws_iam_role.app_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "*"
        Resource = "*"
      }
    ]
  })
}

# EC2 instance
resource "aws_instance" "app" {
  ami                    = "ami-0c55b159cbfafe1f0"
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.web.id]
  iam_instance_profile   = aws_iam_role.app_role.name

  user_data = <<-EOF
    #!/bin/bash
    export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
    export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    echo "Starting application..."
  EOF
}

# Lambda with environment secrets
resource "aws_lambda_function" "processor" {
  filename         = "lambda.zip"
  function_name    = "data-processor"
  role             = aws_iam_role.app_role.arn
  handler          = "index.handler"
  runtime          = "python3.11"

  environment {
    variables = {
      DB_PASSWORD     = "SuperSecret123!"
      API_KEY         = "sk-live-abc123def456"
      STRIPE_SECRET   = "sk_live_1234567890"
    }
  }
}
```

**Requirements:**

1. List all 8 misconfigurations with resource references.
2. Classify each as critical, high, or medium severity.
3. Provide the corrected HCL block for each resource.
4. Summarize the principle violated (least privilege, encryption at rest, no secrets in code, etc.).

**Constraints:**

- Do not remove functionality — fix the security issue while keeping the resource operational.
- Use AWS-managed KMS keys for encryption (no custom key creation required).
- Replace hardcoded secrets with references to `aws_secretsmanager_secret_version` or `var.*`.
