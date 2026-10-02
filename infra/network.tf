# VPC across two availability zones. RDS requires a subnet group spanning two zones,
# even for a single-instance database.
#
#   public  10.0.0.0/24, 10.0.1.0/24   -> Internet Gateway   (only the NAT instance lives here)
#   private 10.0.10.0/24, 10.0.11.0/24 -> fck-nat instance   (RDS, Lambda)

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true # RDS hands out a DNS name, not an IP
  tags                 = { Name = "footyvision" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "footyvision" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index) # 10.0.0.0/24, 10.0.1.0/24
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "footyvision-public-${local.azs[count.index]}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 10) # 10.0.10.0/24, 10.0.11.0/24
  availability_zone = local.azs[count.index]
  tags              = { Name = "footyvision-private-${local.azs[count.index]}" }
}

# What makes a subnet public: a default route to the Internet Gateway.
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "footyvision-public" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# What makes a subnet private: its default route goes to the NAT instance, which can
# open connections to the internet but accepts none from it.
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block           = "0.0.0.0/0"
    network_interface_id = aws_instance.nat.primary_network_interface_id
  }
  tags = { Name = "footyvision-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# --- fck-nat: a t4g.micro doing the NAT Gateway's job for ~1/5 of the price ----------
# One instance in one zone: if that zone fails, the private subnets lose internet access
# (the database stays reachable). An accepted trade-off for a portfolio project.

data "aws_ami" "fck_nat" {
  most_recent = true
  owners      = ["568608671756"] # the fck-nat project's AWS account
  filter {
    name   = "name"
    values = ["fck-nat-al2023-*-arm64-ebs"]
  }
}

resource "aws_security_group" "nat" {
  name        = "footyvision-nat"
  description = "Accepts traffic from inside the VPC and forwards it out"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Anything from the VPC"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "nat" {
  ami                    = data.aws_ami.fck_nat.id
  instance_type          = "t4g.micro" # the smallest ARM type the AWS Free plan allows; t4g.nano is refused
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.nat.id]
  source_dest_check      = false                                 # it forwards packets addressed to other machines; AWS drops those by default
  iam_instance_profile   = aws_iam_instance_profile.nat_ssm.name # database.tf: the SSM tunnel

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  tags = { Name = "footyvision-fck-nat" }
}

# S3 traffic from the private subnets takes this free shortcut instead of the NAT.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "footyvision-s3" }
}
