output "vpc_id" {
  value = aws_vpc.main.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "nat_instance_id" {
  value = aws_instance.nat.id
}

output "rds_endpoint" {
  value = aws_db_instance.main.address
}

output "rds_secret_arn" {
  value = aws_db_instance.main.master_user_secret[0].secret_arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "gemini_secret_arn" {
  value = aws_secretsmanager_secret.gemini.arn
}

output "api_url" {
  value = aws_apigatewayv2_api.api.api_endpoint
}

output "site_bucket" {
  value = aws_s3_bucket.site.id
}

output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.site.id
}

output "site_url" {
  value = "https://${aws_cloudfront_distribution.site.domain_name}"
}
