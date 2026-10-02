# Security policy

## Scope

FootyVision is a research/portfolio project that runs locally: Postgres in Docker, a local
LLM server, and an API bound to `localhost`. It has **no authentication layer** and is not
hardened for public deployment. Do not expose the API to the internet as-is.

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/pinthoz/FootyVisionX/security/advisories/new)
rather than a public issue. I aim to respond within a week.

## What is already accounted for

- **No raw SQL from user input.** Natural-language search is constrained to a validated
  Pydantic `PlayerQuery`; the SQL is built by trusted code with bound parameters.
- **No secrets in the repo.** `.env` is gitignored; `.env.example` holds local defaults only.
- **No outbound data.** Chat and embeddings go to an OpenAI-compatible endpoint you run
  yourself; no player data leaves the machine.
- **Container images** run as a non-root user and verify TLS when installing dependencies.

## Known limitations

- The API is unauthenticated and not rate-limited, by design (local use).
- LLM output is grounded in computed statistics but is still model-generated prose — it is
  not a substitute for verifying the underlying numbers, which the API exposes directly.
