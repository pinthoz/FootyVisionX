# Phase 2: the API. The image registry and the Gemini key first, then the Lambda that
# runs the image and the API Gateway in front of it.

# ECR: private registry for the API image

resource "aws_ecr_repository" "api" {
  name                 = "footyvision-api"
  image_tag_mutability = "IMMUTABLE" # a tag, once pushed, always means the same image
  force_delete         = true        # destroy removes it even with images inside

  image_scanning_configuration {
    scan_on_push = true # every push is checked against known CVEs, for free
  }
}

# Each image is a few hundred MB and storage is billed per GB-month: keep the last 10.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# The Gemini key
# Terraform creates the empty secret; the value is put in with the AWS CLI. Had the key
# been a Terraform value, it would sit in plain text in the state file.

resource "aws_secretsmanager_secret" "gemini" {
  name        = "footyvision/gemini"
  description = "GEMINI_API_KEY for the API Lambda"
  # Deleted at once on destroy, instead of the default 30-day recovery window that
  # blocks re-creating a secret with the same name.
  recovery_window_in_days = 0
}

# The Lambda
# Runs the image pushed to ECR, inside the private subnets so it can reach RDS. It goes
# out to Gemini and to Secrets Manager through the fck-nat instance.

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/footyvision-api"
  retention_in_days = 14 # the default keeps logs forever, and storage is billed
}

resource "aws_iam_role" "api" {
  name = "footyvision-api"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Write logs, and create the network interfaces a Lambda needs to sit inside a VPC.
resource "aws_iam_role_policy_attachment" "api_vpc" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Read exactly two secrets, and nothing else in Secrets Manager.
resource "aws_iam_role_policy" "api_secrets" {
  name = "read-secrets"
  role = aws_iam_role.api.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "secretsmanager:GetSecretValue"
      Resource = [
        aws_secretsmanager_secret.gemini.arn,
        aws_db_instance.main.master_user_secret[0].secret_arn,
      ]
    }]
  })
}

resource "aws_lambda_function" "api" {
  function_name = "footyvision-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
  architectures = ["x86_64"] # matches the image built on a regular PC, no emulation

  # The fine-tuned embedder holds ~1.2GB of weights on top of the API's ~400MB, and
  # Lambda CPU scales with memory, which shortens the model load.
  memory_size = 3008
  # Visitors' requests are cut at 30s by the API Gateway; the extra time is for the
  # scheduled warm-up, which loads the model outside any visitor's request. The first
  # warm-up after a new image ran past 60s: the image is fetched lazily on first read.
  timeout = 120

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      # ARNs only: footyvision/aws_secrets.py fetches the values at cold start.
      DB_SECRET_ARN      = aws_db_instance.main.master_user_secret[0].secret_arn
      DB_HOST            = aws_db_instance.main.address
      DB_NAME            = aws_db_instance.main.db_name
      GEMINI_SECRET_ARN  = aws_secretsmanager_secret.gemini.arn
      CLOUD_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
      # Free tier: 15 requests/min and 500/day, against 5/min and 20/day for
      # gemini-2.5-flash, which the dashboard alone used up in a day.
      CLOUD_LLM_MODEL = "gemini-3.5-flash-lite"
      # Tried in order when the model above answers 429 (out of quota) or 5xx. Each has
      # its own free daily quota: ~1,040 questions a day across the chain, at no cost.
      CLOUD_LLM_FALLBACK_MODELS = "gemini-3.1-flash-lite,gemini-3.8-flash,gemini-2.5-flash"
      # Only a fallback: with FINETUNED_EMBED_PATH set, queries are embedded by the
      # fine-tuned model.
      CLOUD_LLM_EMBED_MODEL = "gemini-embedding-2"
      RATE_LIMIT_PER_MINUTE = "20"
      # The retriever FootyVision runs locally, and the one the stored index was built
      # with: query and index vectors have to come from the same model.
      FINETUNED_EMBED_PATH = "models/embeddinggemma-footyvision"
    }
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy_attachment.api_vpc]
}

# The API Gateway
# The public HTTPS front door. Every path goes to the Lambda; the stage throttles the
# total, so a burst of traffic is refused here before it costs Lambda time or Gemini calls.

resource "aws_apigatewayv2_api" "api" {
  name          = "footyvision-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "api" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "$default" # any method, any path
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "api" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = 5  # requests per second, sustained
    throttling_burst_limit = 10 # requests allowed at once above that
  }
}

# Without this the gateway is not allowed to call the function.
resource "aws_lambda_permission" "api" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# Warm-up
# Every 5 minutes EventBridge Scheduler invokes the Lambda directly. The Web Adapter
# turns that event into POST /events (api/routers/warm.py), which loads the embedder and
# the index. One environment then stays warm with both in memory, so a visitor's
# question does not pay the ~20s model load. ~8,600 short invocations a month: inside
# the Lambda free tier.

resource "aws_iam_role" "warm" {
  name = "footyvision-warm"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "warm" {
  name = "invoke-api"
  role = aws_iam_role.warm.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.api.arn
    }]
  })
}

resource "aws_scheduler_schedule" "warm" {
  name                = "footyvision-warm"
  schedule_expression = "rate(5 minutes)"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.api.arn
    role_arn = aws_iam_role.warm.arn
    input    = jsonencode({ source = "footyvision.warm" })
  }
}
