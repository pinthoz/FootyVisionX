"""Per-client rate limiting for the endpoints that spend money.

`/assistant`, `/search` and the report endpoint each call an LLM, so an unthrottled
public deployment lets anyone drain the API key attached to it. This is the actual
defence: CORS restricts which *sites* a browser will let call the API, but it does
nothing about curl, so it is a good hygiene measure and a bad security control.

Deliberately in-process and dependency-free. A single Render instance has one process,
so a shared counter is exact there; behind several replicas each would hold its own
window and the effective limit multiplies by the replica count. Redis is the swap-in if
that day comes — the interface here would not change.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from footyvision.config import get_settings

WINDOW_SECONDS = 60.0

# client key -> timestamps of the calls still inside the window.
_HITS: defaultdict[str, deque[float]] = defaultdict(deque)

# Past this many tracked clients, every window is swept before a new one is opened. Keys
# used to be pruned of old timestamps but never removed, so each distinct caller left an
# entry behind for the life of the process — and with a spoofable header, a caller can be
# as many distinct callers as it likes. On a 512MB instance that is an out-of-memory
# restart on demand.
MAX_TRACKED_CLIENTS = 10_000


def client_key(request: Request) -> str:
    """Identify the caller by the address the proxy saw, not the one the caller claims.

    Render and Vercel both terminate TLS in front of the app, so `request.client.host`
    is the proxy and every caller would share one bucket; the address has to come from
    `X-Forwarded-For`. But every proxy *appends* to that header, which makes the leftmost
    entry whatever the client chose to send. Reading that one — as this used to — let any
    caller pick a fresh identity per request by setting the header, and so skip the limit
    entirely. The rightmost entry is the one written by the proxy directly in front of the
    app, from the connection it actually accepted, and the client cannot forge it.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def reset() -> None:
    """Drop every window. Tests need this; nothing in the app calls it."""
    _HITS.clear()


def _sweep(now: float) -> None:
    """Forget every client with nothing left inside the window."""
    for key in [k for k, hits in _HITS.items() if not hits or now - hits[-1] > WINDOW_SECONDS]:
        del _HITS[key]


def rate_limit(request: Request) -> None:
    """FastAPI dependency: allow N calls per minute per client, then 429.

    Set `RATE_LIMIT_PER_MINUTE=0` to disable entirely, which is what local development
    and the test suite do.
    """
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return

    now = time.monotonic()
    key = client_key(request)
    if key not in _HITS and len(_HITS) >= MAX_TRACKED_CLIENTS:
        _sweep(now)
    hits = _HITS[key]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= limit:
        retry_after = int(WINDOW_SECONDS - (now - hits[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {limit} requests per minute exceeded.",
            headers={"Retry-After": str(retry_after)},
        )

    hits.append(now)
