"""In-process embeddings from a fine-tuned sentence-transformers model.

LM Studio serves the stock EmbeddingGemma. The LoRA-tuned copy produced by
`scripts/finetune_embeddings.py` lives on disk instead, and both sides of retrieval have
to use the *same* model: an index built with the tuned weights and queried with the stock
ones would compare vectors from two different spaces and quietly get worse, not better.
Routing both through here is what keeps them in step.

Optional by construction. The import of sentence-transformers (and therefore torch) only
happens when `finetuned_embed_path` is set, so the API still runs on a machine that has
neither installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL: Any | None = None
_LOADED_FROM: str | None = None


def is_configured(path: str | None) -> bool:
    """Whether a fine-tuned model is configured *and* actually on disk."""
    return bool(path) and Path(path).is_dir()


def get_model(path: str) -> Any:
    """Load the model once per process and keep it.

    A 300M model costs about a gigabyte of RAM and several seconds to load, so it is
    cached like the vector store is — and, for the same reason, the API has to be
    restarted after the model on disk is replaced.
    """
    global _MODEL, _LOADED_FROM
    if _MODEL is not None and _LOADED_FROM == path:
        return _MODEL

    from sentence_transformers import SentenceTransformer

    logger.info("loading fine-tuned embedding model from %s", path)
    _MODEL = SentenceTransformer(path)
    _LOADED_FROM = path
    return _MODEL


def embed(path: str, texts: list[str], prefix: str) -> list[list[float]]:
    """Embed with the fine-tuned model, applying the same task prefix it was trained with.

    The prefix is not cosmetic: the model was fine-tuned on prefixed text, so dropping it
    at inference asks it to score inputs it never saw in that shape.
    """
    model = get_model(path)
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]
