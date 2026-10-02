"""Compare embedding models (and their task prefixes) on FootyVision's own retrieval task.

Choosing an embedding model by MTEB leaderboard is guesswork: what matters is whether it
retrieves the right *players* from these 411 profiles. So this scores each candidate on
two things that can be checked against the data itself — no manual labels, no LLM judge:

  position@k  fraction of the retrieved players who play the position the query asked for
  trait pct   mean percentile (within their own position group) of the metrics the query
              asked about — "wins the ball" should retrieve players who actually do

Queries come in Portuguese and English pairs, because the profiles are written in English
while the dashboard is used in Portuguese: the PT/EN gap measures cross-lingual retrieval,
which is the main reason to consider replacing an English-first model.

Usage (LM Studio must be serving every model named in CONFIGS):
    python scripts/eval_embeddings.py
    python scripts/eval_embeddings.py --k 6 --no-cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.config import get_settings  # noqa: E402
from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.llm.client import LLMClient  # noqa: E402
from footyvision.ml.features import PER90_FEATURES, load_feature_frame  # noqa: E402
from footyvision.rag.profiles import build_profiles  # noqa: E402

CACHE_DIR = Path(".eval_cache")

# Each config is (label, embedding model served by LM Studio, document prefix, query prefix).
# The prefixes are the ones each model was trained with — using none, or the wrong ones,
# is itself one of the things being measured.
CONFIGS: list[tuple[str, str, str, str]] = [
    ("nomic (no prefix)", "text-embedding-nomic-embed-text-v1.5", "", ""),
    (
        "nomic + prefixes",
        "text-embedding-nomic-embed-text-v1.5",
        "search_document: ",
        "search_query: ",
    ),
    ("gemma (no prefix)", "text-embedding-embeddinggemma-300m-qat", "", ""),
    (
        "gemma + prefixes",
        "text-embedding-embeddinggemma-300m-qat",
        "title: none | text: ",
        "task: search result | query: ",
    ),
]

# (query, language, expected position group, metrics the query is really asking about)
QUERIES: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "médio defensivo que ganhe bolas e intercete",
        "pt",
        "MID",
        ("tackles_per90", "interceptions_per90", "ball_recoveries_per90"),
    ),
    (
        "defensive midfielder who wins the ball and intercepts passes",
        "en",
        "MID",
        ("tackles_per90", "interceptions_per90", "ball_recoveries_per90"),
    ),
    (
        "avançado goleador que remata muito",
        "pt",
        "FWD",
        ("goals_per90", "shots_per90", "xg_per90"),
    ),
    (
        "clinical striker who scores goals and shoots often",
        "en",
        "FWD",
        ("goals_per90", "shots_per90", "xg_per90"),
    ),
    (
        "defesa central que corta o perigo e bloqueia remates",
        "pt",
        "DEF",
        ("clearances_per90", "blocks_per90"),
    ),
    (
        "centre back who clears danger and blocks shots",
        "en",
        "DEF",
        ("clearances_per90", "blocks_per90"),
    ),
    (
        "extremo driblador que passa pelos defesas",
        "pt",
        "FWD",
        ("dribbles_per90", "dribbles_completed_per90", "progressive_carries_per90"),
    ),
    (
        "winger who dribbles past defenders",
        "en",
        "FWD",
        ("dribbles_per90", "dribbles_completed_per90", "progressive_carries_per90"),
    ),
    (
        "médio criativo que faz passes progressivos e assistências",
        "pt",
        "MID",
        ("progressive_passes_per90", "assists_per90"),
    ),
    (
        "creative midfielder with progressive passes and assists",
        "en",
        "MID",
        ("progressive_passes_per90", "assists_per90"),
    ),
]


def embed_all(client: LLMClient, texts: list[str], batch_size: int = 16) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(client.embed(texts[i : i + batch_size]))
    return np.asarray(vectors, dtype=np.float32)


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def within_group_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Percentile of every per-90 feature within the player's own position group."""
    out = frame.copy()
    for feature in PER90_FEATURES:
        out[feature] = frame.groupby("position_group")[feature].rank(pct=True) * 100.0
    return out.set_index("player_id")


def evaluate(
    label: str,
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ids: np.ndarray,
    percentiles: pd.DataFrame,
    groups: pd.Series,
    k: int,
) -> list[dict]:
    rows = []
    for (question, lang, expected_group, metrics), qv in zip(QUERIES, query_vectors, strict=True):
        scores = doc_vectors @ qv
        top = np.argsort(scores)[::-1][:k]
        retrieved = [int(ids[i]) for i in top]

        position_hits = sum(1 for pid in retrieved if groups.loc[pid] == expected_group)
        trait = float(np.mean([percentiles.loc[pid, list(metrics)].mean() for pid in retrieved]))
        rows.append(
            {
                "config": label,
                "lang": lang,
                "query": question,
                "position@k": position_hits / k,
                "trait_pct": trait,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6, help="Documents retrieved per query.")
    parser.add_argument("--no-cache", action="store_true", help="Re-embed even if cached.")
    args = parser.parse_args()

    session = SessionLocal()
    frame = load_feature_frame(session, get_settings().min_minutes)
    docs = build_profiles(frame)
    texts = [d["text"] for d in docs]
    ids = np.asarray([d["player_id"] for d in docs])
    print(f"{len(docs)} player profiles, k={args.k}\n")

    percentiles = within_group_percentiles(frame)
    groups = frame.set_index("player_id")["position_group"]

    CACHE_DIR.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    for label, model, doc_prefix, query_prefix in CONFIGS:
        cache = CACHE_DIR / f"{label.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        if cache.exists() and not args.no_cache:
            data = np.load(cache)
            doc_vectors, query_vectors = data["docs"], data["queries"]
            print(f"{label:20} cached")
        else:
            client = LLMClient(embed_model=model, timeout=600.0)
            start = time.time()
            doc_vectors = normalize(embed_all(client, [doc_prefix + t for t in texts]))
            query_vectors = normalize(
                embed_all(client, [query_prefix + q for q, _, _, _ in QUERIES])
            )
            np.savez(cache, docs=doc_vectors, queries=query_vectors)
            print(f"{label:20} embedded in {time.time() - start:.0f}s (dim {doc_vectors.shape[1]})")

        all_rows += evaluate(label, doc_vectors, query_vectors, ids, percentiles, groups, args.k)

    results = pd.DataFrame(all_rows)
    order = [c[0] for c in CONFIGS]

    print("\n=== Overall (higher is better) ===")
    overall = results.groupby("config")[["position@k", "trait_pct"]].mean().loc[order]
    print(overall.round(3).to_string())

    print("\n=== By query language ===")
    by_lang = (
        results.groupby(["config", "lang"])[["position@k", "trait_pct"]].mean().unstack().loc[order]
    )
    print(by_lang.round(3).to_string())

    print("\n=== Per query (position@k) ===")
    per_query = results.pivot_table(index=["lang", "query"], columns="config", values="position@k")[
        order
    ]
    print(per_query.round(2).to_string())


if __name__ == "__main__":
    main()
