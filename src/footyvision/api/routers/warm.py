"""Warm-up for AWS Lambda: load the retriever before a visitor's question needs it.

Loading the fine-tuned embedder (google/embeddinggemma-300m plus the LoRA adapter) takes
about 20s on a cold environment. Inside a visitor's /assistant call, that plus the LLM
answer runs past the API Gateway's 30s limit. An EventBridge schedule invokes the Lambda
every few minutes instead; the Lambda Web Adapter delivers such non-HTTP events as a
POST to /events, and this route pays the loading cost there, outside any visitor's
request. Lambda freezes an environment between invocations, so a background thread at
startup would not get the work done.

Harmless if called through the public URL: it only loads what /assistant would load.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.llm import local_embedder
from footyvision.rag.service import get_store

router = APIRouter()


@router.post("/events", include_in_schema=False)
def warm(session: Session = Depends(get_session)) -> dict:
    settings = get_settings()
    path = settings.finetuned_embed_path
    if local_embedder.is_configured(path):
        # Loading only maps the weights (safetensors uses mmap); the bytes are read on the
        # first forward pass. On Lambda that read comes from the lazily-fetched image and
        # took ~40s, so one query is embedded here to pay it outside a visitor's request.
        local_embedder.embed(path, ["warm-up"], settings.llm_embed_query_prefix)
    store = get_store(session)
    return {"warm": True, "profiles": len(store)}
