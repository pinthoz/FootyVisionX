<div align="center">

<img src="docs/images/logo.svg" alt="FootyVisionX" width="110" height="110">

# FootyVisionX

**A football scouting project built with public data, machine learning and AWS.**

[![CI](https://github.com/pinthoz/FootyVisionX/actions/workflows/ci.yml/badge.svg)](https://github.com/pinthoz/FootyVisionX/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Features](#features) · [Architecture](#architecture) · [Run locally](#run-locally) · [AWS deployment](#aws-deployment)

</div>

## About

FootyVisionX is the AWS version of
[FootyVision](https://footy-vision-tau.vercel.app/), which runs locally with free tools.

I built it to explore a practical question: how much can public football data tell us about
a player? The project combines event data, traditional machine-learning models and an LLM.
The models calculate scores and comparisons, while the LLM turns that context into readable
reports and answers.

The AWS version keeps the same analysis and moves the application to Lambda, RDS, S3 and
CloudFront. It also uses Gemini for reports, natural-language search and the RAG assistant.

## Features

| Feature | What it does |
|---|---|
| Player comparison | Finds players with similar statistical profiles |
| Radar charts | Compares a player with others in the same position group |
| Performance score | Combines position-specific metrics into a score from 0 to 100 |
| Position models | Predicts position group, role and likely exact positions |
| Scouting reports | Generates a written report from calculated player data |
| Natural-language search | Converts a question into validated filters instead of raw SQL |
| Market value | Estimates a value range and highlights possible bargains |
| Pitch maps | Shows where a player acts, shoots and creates threat |
| Team analysis | Estimates team strength and expected league tables |
| RAG assistant | Retrieves relevant player profiles before answering a question |

Example questions include:

- Who could replace this player?
- Which forwards have more than 0.5 xG per 90?
- Find a defensive midfielder who wins the ball and intercepts passes.
- Is this player good value for the estimated price?

## Architecture

```text
Browser
   |
   v
CloudFront ----------------> private S3 bucket
   |                          Next.js static dashboard
   | /api/*
   v
API Gateway -------------> FastAPI on AWS Lambda
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
              PostgreSQL on RDS         Model artifacts and
                                        fine-tuned embeddings
                     |                         |
                     +------------+------------+
                                  |
                           private subnets
                                  |
                           fck-nat instance
                                  |
                                  v
                                Gemini
```

CloudFront serves the dashboard from a private S3 bucket. Requests under `/api/*` are sent
to API Gateway and then to the containerized FastAPI application on Lambda. Lambda and RDS
run in private subnets. The NAT instance gives Lambda outbound access to Gemini without
making the database public.

Terraform defines the AWS resources in [`infra/`](infra). CloudWatch stores the Lambda logs,
Secrets Manager holds credentials and EventBridge periodically warms the function.

For more detail about the data and modelling layers, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Technology

- FastAPI, SQLAlchemy, Alembic and PostgreSQL
- pandas, scikit-learn, XGBoost, LightGBM and SHAP
- Next.js and React
- Gemini and a fine-tuned EmbeddingGemma retriever
- AWS Lambda, API Gateway, RDS, ECR, S3, CloudFront and CloudWatch
- Terraform, Docker and GitHub Actions

## Run locally

You need Python 3.11+, Node.js, Docker and an OpenAI-compatible local server such as LM
Studio or Ollama if you want to use the LLM features.

```bash
# Start PostgreSQL
docker compose up -d db

# Create the Python environment
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Create the schema and load a season
footyvision init-db
footyvision competitions
footyvision load -c 11 -s 27

# Build the outputs used by the API
footyvision spatial
footyvision precompute
footyvision index

# Start the API
uvicorn footyvision.api.main:app --reload
```

In another terminal, start the dashboard:

```bash
cd frontend/web
npm install
npm run dev
```

Open `http://localhost:3000`. API documentation is available at
`http://localhost:8000/docs`.

You can also start the Docker services together:

```bash
docker compose up --build
```

## AWS deployment

The Terraform configuration expects two local files that are not committed:

- `infra/backend.hcl` for the remote state bucket
- `infra/terraform.tfvars` for values such as the budget email and image tag

The deployment follows this order:

1. Create the Terraform state bucket with `infra/bootstrap`.
2. Initialize the main Terraform stack.
3. Create the ECR repository.
4. Build `Dockerfile.lambda` and push the image to ECR with an immutable tag.
5. Apply Terraform using that tag.
6. Add the Gemini key to the secret created in Secrets Manager.
7. Build the static frontend and upload it to S3.

To update only the frontend:

```powershell
npm ci --prefix frontend/web
$env:STATIC_EXPORT = "1"
$env:NEXT_PUBLIC_API_URL = "/api"
npm --prefix frontend/web run build

$bucket = terraform -chdir=infra output -raw site_bucket
$distribution = terraform -chdir=infra output -raw cloudfront_distribution_id

aws s3 sync .\frontend\web\out\ "s3://$bucket/" --delete
aws cloudfront create-invalidation --distribution-id $distribution --paths "/*"
```

Terraform prints the deployed addresses:

```powershell
terraform -chdir=infra output -raw site_url
terraform -chdir=infra output -raw api_url
```

## API

The main endpoints are:

- Players: `/players`, `/players/{id}`, `/players/{id}/similar`, `/players/{id}/radar`
- Models: `/players/{id}/score`, `/players/{id}/value`, `/rankings`, `/talent/model-info`
- Language: `POST /search`, `POST /assistant`, `POST /players/{id}/report`
- Teams: `/teams/strength`, `/teams/expected-table`
- Project information: `/coverage`, `/eval/assistant`, `/health`

The endpoints that call an LLM are rate-limited. Questions are validated before reaching
the model, and natural-language searches are converted into Pydantic models rather than SQL.

## Data and limitations

The project uses StatsBomb Open Data. The current dataset contains 2,322 matches and 2,562
player-seasons from ten competitions. It covers the five major men's leagues in 2015/16 and
five women's competitions in 2023/24.

Public data has gaps. Date of birth and preferred foot are only available for part of the
dataset, and the Portuguese league is not included. The API reports when a filter excludes
players because an attribute is missing.

The models should be read as decision-support tools, not objective player rankings. Their
evaluation results and known weaknesses are kept in the repository so that the numbers can
be interpreted with the right context.

## Repository structure

```text
src/footyvision/       Python package, API, ETL, ML and RAG
frontend/web/          Next.js dashboard
infra/                 Terraform for AWS
migrations/            Alembic database migrations
models/                Precomputed model artifacts
scripts/               Evaluation and training scripts
tests/                 Unit and API tests
docs/                  Architecture notes and roadmap
```

## Development

```bash
pytest
ruff check src tests scripts
ruff format --check src tests
```

CI runs the backend checks on Python 3.11 and 3.12 and creates a production build of the
dashboard. Contribution notes are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © Ana Pinto.

Football data comes from [StatsBomb Open Data](https://github.com/statsbomb/open-data).
Market values come from the public Transfermarkt dataset on Kaggle. This project is not
affiliated with StatsBomb or Transfermarkt.
