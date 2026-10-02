variable "region" {
  type    = string
  default = "eu-west-1" # Ireland; Bedrock is available here. The state bucket stays in eu-central-1
}

variable "budget_email" {
  type        = string
  description = "Where AWS Budgets sends the spending alerts."
}

variable "budget_limit_usd" {
  type    = number
  default = 15
}

variable "api_image_tag" {
  type        = string
  default     = "v5" # v1: arm64; v2: no fine-tuned embedder; v3: cold first query; v4: no LLM fallback
  description = "Tag of the API image in ECR that the Lambda runs."
}
