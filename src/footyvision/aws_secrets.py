"""Fill DATABASE_URL and GEMINI_API_KEY from AWS Secrets Manager, when running on AWS.

On Lambda the environment carries only the secrets' ARNs, never their values: a value in
an environment variable is shown in plain text in the Lambda console and stored in the
Terraform state. The values are fetched once per cold start, before Settings is built,
and written into os.environ under the names Settings already reads, so nothing else in
the app knows where they came from.

Off AWS (local, Render, CI) none of the *_SECRET_ARN variables is set and this is a
no-op; boto3 is only imported when there is something to fetch, so the runtime install
does not need it.

The database secret is the one RDS manages and rotates. A rotation changes the password
under a running Lambda: open connections survive, new ones fail until the next cold
start. Acceptable at this traffic; a retry that re-reads the secret is the fix if not.
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote


def load() -> None:
    db_arn = os.environ.get("DB_SECRET_ARN")
    gemini_arn = os.environ.get("GEMINI_SECRET_ARN")
    if not (db_arn or gemini_arn):
        return

    import boto3

    client = boto3.client("secretsmanager")

    if db_arn and not os.environ.get("DATABASE_URL"):
        # RDS stores {"username": ..., "password": ...}; host and name come from Terraform.
        secret = json.loads(client.get_secret_value(SecretId=db_arn)["SecretString"])
        user = quote(secret["username"], safe="")
        password = quote(secret["password"], safe="")  # may hold @, / or :
        host = os.environ["DB_HOST"]
        name = os.environ.get("DB_NAME", "footyvision")
        os.environ["DATABASE_URL"] = (
            f"postgresql+psycopg2://{user}:{password}@{host}:5432/{name}?sslmode=require"
        )

    if gemini_arn and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = client.get_secret_value(SecretId=gemini_arn)[
            "SecretString"
        ].strip()
