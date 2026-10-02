terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # The bucket name comes from `terraform init -backend-config=backend.hcl`, so it is
  # not committed. use_lockfile locks the state with an S3 object: two applies at once
  # cannot corrupt it, and no DynamoDB table is needed.
  backend "s3" {
    key          = "footyvision/terraform.tfstate"
    region       = "eu-central-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "footyvision" } # every resource carries it, so Cost Explorer can filter by project
  }
}
