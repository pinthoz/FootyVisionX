resource "aws_budgets_budget" "monthly" {
  name         = "footyvision-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # One third of the limit already spent: something is running that should not be.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 33
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  # The month's trend is heading past the limit, warned before it gets there.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
