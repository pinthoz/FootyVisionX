# PostgreSQL 18 in the private subnets. Nothing on the internet can reach it: the only
# ways in are the Lambda (phase 2) and an SSM tunnel through the fck-nat instance.

# The Lambda's security group exists before the Lambda does, so the database rule can
# name it now. Phase 2 attaches it to the function.
resource "aws_security_group" "lambda" {
  name        = "footyvision-lambda"
  description = "Attached to the API Lambda; the database admits this group"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds" {
  name        = "footyvision-rds"
  description = "PostgreSQL, from the Lambda and the SSM tunnel only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "API Lambda"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.lambda.id]
  }

  ingress {
    description     = "SSM tunnel through the fck-nat instance"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.nat.id]
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "footyvision"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "main" {
  identifier     = "footyvision"
  engine         = "postgres"
  engine_version = "18"
  instance_class = "db.t4g.micro" # Free plan eligible

  allocated_storage = 20 # the Free plan's storage allowance
  storage_type      = "gp2"
  storage_encrypted = true

  db_name  = "footyvision"
  username = "footyadmin"
  # RDS generates the password, stores it in Secrets Manager and rotates it. It never
  # appears in this code, in the plan output or in a variable.
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period = 1 # the Free plan caps automated backups at one day

  # A portfolio database: destroy deletes it without leaving a snapshot that keeps
  # billing. Turn both around before this holds anything that cannot be reloaded.
  skip_final_snapshot = true
  deletion_protection = false
  apply_immediately   = true
}

# SSM: lets `aws ssm start-session` reach the fck-nat instance, which then forwards
# the tunnel to the database. No SSH key, no port 22 open.

resource "aws_iam_role" "nat_ssm" {
  name = "footyvision-nat-ssm"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "nat_ssm" {
  role       = aws_iam_role.nat_ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "nat_ssm" {
  name = "footyvision-nat-ssm"
  role = aws_iam_role.nat_ssm.name
}
