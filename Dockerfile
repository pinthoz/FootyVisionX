FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps for psycopg2 + runtime libgomp1 for LightGBM/XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Escape hatch for TLS-intercepting corporate proxies. Empty by default (certificates are
# verified); build with --build-arg PIP_TRUSTED_HOSTS="pypi.org files.pythonhosted.org"
# only if your network requires it.
ARG PIP_TRUSTED_HOSTS=""
RUN pip install --upgrade pip \
    && if [ -n "$PIP_TRUSTED_HOSTS" ]; then \
           pip install -e . $(printf -- '--trusted-host %s ' $PIP_TRUSTED_HOSTS); \
       else \
           pip install -e .; \
       fi

# Run the API as an unprivileged user.
RUN useradd --create-home --uid 1000 footy && chown -R footy:footy /app
USER footy

EXPOSE 8000

# Create tables (idempotent) then serve the API.
CMD ["sh", "-c", "footyvision init-db && uvicorn footyvision.api.main:app --host 0.0.0.0 --port 8000"]
