"""Control: does the *untrained* bge-reranker already beat the bi-encoder here?

The fine-tuned reranker scored 0.162 against a 0.419 baseline, and 5/30 of the 0.798
shortlist ceiling is 0.133 -- so it is ordering the shortlist very nearly at random.
This script tells the two apart: it scores the same shortlists with the base checkpoint,
untouched.

The answer was neither. The base model loses 0.153 zero-shot and the fine-tune loses 0.258,
while the score spread stays wide (0.093 base, 0.159 fine-tuned) -- so nothing collapsed and
the training did not ruin a working model. The cross-encoder is confidently ranking by how
well a profile fits the description, which many profiles do. `finetune_reranker.py` carries
the full reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_embeddings import build_pairs, load_frame, split_by_player  # noqa: E402
from finetune_reranker import SHORTLIST, recall_at_k, shortlists  # noqa: E402

from footyvision.config import get_settings  # noqa: E402

SAMPLE = 900


def main() -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = load_frame(get_settings().min_minutes)
    _, test_pairs = split_by_player(build_pairs(frame))

    bi = SentenceTransformer("models/embeddinggemma-footyvision", device=device)
    bi.max_seq_length = 192
    ids, ranked, _ = shortlists(bi, test_pairs, SHORTLIST)
    del bi
    torch.cuda.empty_cache()

    # Subsample queries only: the corpus (and so the difficulty) stays whole.
    rng = np.random.default_rng(42)
    pick = rng.choice(len(test_pairs), size=min(SAMPLE, len(test_pairs)), replace=False)
    pairs = [test_pairs[i] for i in pick]
    rows = ranked[pick]
    profiles = {p.player_id: p.profile for p in test_pairs}

    base = recall_at_k(ids, rows, pairs, 5)
    print(f"bi-encoder            recall@5: {base['all']:.3f}  (n={len(pairs)})")
    ceiling = recall_at_k(ids, rows, pairs, SHORTLIST)["all"]
    print(f"shortlist@{SHORTLIST} ceiling:            {ceiling:.3f}\n")

    for label, path in (
        ("base (zero-shot)", "BAAI/bge-reranker-base"),
        ("fine-tuned", "models/reranker-footyvision"),
    ):
        cross = CrossEncoder(path, max_length=256, device=device)
        out = np.empty_like(rows)
        spreads = []
        for i, (pair, row) in enumerate(zip(pairs, rows, strict=True)):
            scores = np.asarray(
                cross.predict(
                    [[pair.query, profiles[ids[j]]] for j in row], show_progress_bar=False
                )
            ).ravel()
            spreads.append(float(scores.std()))
            out[i] = row[np.argsort(-scores)]
        got = recall_at_k(ids, out, pairs, 5)
        # A collapsed model gives every candidate the same score; the spread says so.
        print(
            f"{label:22s}recall@5: {got['all']:.3f}  "
            f"({got['all'] - base['all']:+.3f})  score spread {np.mean(spreads):.4f}"
        )
        del cross
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
