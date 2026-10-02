"""Fine-tune a cross-encoder to rerank what the bi-encoder retrieves.

The retriever already fine-tuned in `finetune_embeddings.py` is a **bi-encoder**: it
embeds the question and each profile separately and compares the two vectors. That is what
makes it fast enough to score two thousand profiles per query, and also what limits it —
the two texts never meet, so nothing in the model can weigh "left-footed" in the question
against "left-footed" in the profile.

A **cross-encoder** reads question and profile together as one input and scores the pair
directly. It is far more accurate and far too slow to run over the whole pool, so the two
work in sequence: the bi-encoder proposes a shortlist, the cross-encoder reorders it. This
is the standard two-stage retrieval pattern, and the interesting part is the training
data — a reranker learns nothing from random negatives, because telling a winger from a
goalkeeper is a problem the bi-encoder already solved. It has to be trained on the
mistakes the bi-encoder actually makes.

Base model: BAAI/bge-reranker-base, multilingual because half the queries are Portuguese.

RESULT: this does not work here, and the script is kept for the measurement, not the model.

    bi-encoder            recall@5  0.408
    base (zero-shot)      recall@5  0.254   -0.153
    fine-tuned            recall@5  0.150   -0.258

The base checkpoint loses fifteen points before any training touches it, so the fine-tune
did not spoil a good model -- a cross-encoder is simply worse at this than the retriever.
Training then made it worse still, and the reason is the labels. A query here is generated
from a profile ("young centre-back", "extremo canhoto que dribla") and the pool holds
dozens of players who fit each one, so the hard negatives -- profiles the bi-encoder ranked
highly and "got wrong" -- are frequently other correct answers labelled 0.0 against an
identical query labelled 1.0. The bi-encoder never suffers this because
MultipleNegativesRankingLoss only asks the positive to outrank its batch, and tolerates ties;
binary cross-entropy on contradictory labels does not.

The scores are not collapsed -- their spread across a shortlist is 0.159, wider than the
base model's 0.093 -- so the fine-tune is confidently ranking by how well a profile matches
the description, which is true of many profiles and does not single out the seed player.

The deeper point is that recall@5 against one seed player was the wrong target all along.
See `eval_retrieval_constraints.py`, which asks whether the returned five satisfy what the
question stated, and finds retrieval already strong.

Usage:
    python scripts/finetune_reranker.py --eval-only     # bi-encoder baseline
    python scripts/finetune_reranker.py --epochs 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_embeddings import (  # noqa: E402
    DOC_PREFIX,
    QUERY_PREFIX,
    Pair,
    build_pairs,
    load_frame,
    split_by_player,
)

from footyvision.config import get_settings  # noqa: E402

BASE_RERANKER = "BAAI/bge-reranker-base"
BI_ENCODER = "models/embeddinggemma-footyvision"
OUTPUT_DIR = Path("models/reranker-footyvision")

# How deep the bi-encoder's shortlist goes before reranking. Too shallow and the right
# profile is already gone; too deep and the cross-encoder's cost stops being worth it.
SHORTLIST = 30


def encode_corpus(model, profiles: dict[int, str]) -> tuple[list[int], np.ndarray]:
    ids = list(profiles)
    vectors = model.encode(
        [DOC_PREFIX + profiles[pid] for pid in ids],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return ids, vectors


def shortlists(model, pairs: list[Pair], depth: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """The bi-encoder's top `depth` profiles for every query, as corpus indices."""
    profiles: dict[int, str] = {}
    for pair in pairs:
        profiles.setdefault(pair.player_id, pair.profile)
    ids, corpus = encode_corpus(model, profiles)

    queries = model.encode(
        [QUERY_PREFIX + p.query for p in pairs],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = queries @ corpus.T
    return ids, np.argsort(-scores, axis=1)[:, :depth], scores


def recall_at_k(ids: list[int], ranked: np.ndarray, pairs: list[Pair], k: int) -> dict[str, float]:
    index = {pid: i for i, pid in enumerate(ids)}
    out: dict[str, float] = {}
    for language in ("pt", "en"):
        hits = [
            index[p.player_id] in row[:k]
            for p, row in zip(pairs, ranked, strict=True)
            if p.language == language
        ]
        if hits:
            out[language] = float(np.mean(hits))
    out["all"] = float(
        np.mean([index[p.player_id] in row[:k] for p, row in zip(pairs, ranked, strict=True)])
    )
    return out


def build_training_rows(
    ids: list[int], ranked: np.ndarray, pairs: list[Pair], profiles: dict[int, str], seed: int
) -> tuple[list[list[str]], list[float]]:
    """One positive and several hard negatives per query.

    Hard negatives are profiles the bi-encoder ranked highly and got wrong. Random
    negatives would teach nothing: separating a winger from a goalkeeper is already
    solved, and a reranker exists precisely to settle the cases the first stage confuses.
    """
    rng = random.Random(seed)
    texts: list[list[str]] = []
    labels: list[float] = []

    for pair, row in zip(pairs, ranked, strict=True):
        texts.append([pair.query, pair.profile])
        labels.append(1.0)

        wrong = [ids[i] for i in row if ids[i] != pair.player_id]
        for negative in rng.sample(wrong[:15], k=min(3, len(wrong[:15]))):
            texts.append([pair.query, profiles[negative]])
            labels.append(0.0)

    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"gpu: {torch.cuda.get_device_name(0)} ({free / 1e9:.1f}GB free)")

    frame = load_frame(get_settings().min_minutes)
    pairs = build_pairs(frame)
    train_pairs, test_pairs = split_by_player(pairs)
    print(
        f"{len(pairs)} pairs | train {len(train_pairs)} "
        f"({len({p.player_id for p in train_pairs})} players) | "
        f"test {len(test_pairs)} ({len({p.player_id for p in test_pairs})} players)\n"
    )

    bi = SentenceTransformer(BI_ENCODER, device=device)
    bi.max_seq_length = 192

    test_profiles = {p.player_id: p.profile for p in test_pairs}
    ids, ranked, _ = shortlists(bi, test_pairs, SHORTLIST)
    before = recall_at_k(ids, ranked, test_pairs, args.k)
    ceiling = recall_at_k(ids, ranked, test_pairs, SHORTLIST)
    print(
        f"bi-encoder      recall@{args.k}:  PT {before['pt']:.3f}  EN {before['en']:.3f}  "
        f"all {before['all']:.3f}"
    )
    # Reranking can only reorder what the first stage returned, so this is the hard limit.
    print(f"shortlist@{SHORTLIST} ceiling: {ceiling['all']:.3f}\n")

    if args.eval_only:
        return

    train_profiles = {p.player_id: p.profile for p in train_pairs}
    train_ids, train_ranked, _ = shortlists(bi, train_pairs, SHORTLIST)
    texts, labels = build_training_rows(
        train_ids, train_ranked, train_pairs, train_profiles, seed=42
    )
    if args.max_train and len(texts) > args.max_train:
        picked = random.Random(42).sample(range(len(texts)), args.max_train)
        texts = [texts[i] for i in picked]
        labels = [labels[i] for i in picked]
    positives = int(sum(labels))
    print(
        f"training rows: {len(texts)} "
        f"({positives} positive, {len(texts) - positives} hard negative)"
    )

    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader

    cross = CrossEncoder(BASE_RERANKER, num_labels=1, max_length=256, device=device)
    loader = DataLoader(
        [InputExample(texts=t, label=lbl) for t, lbl in zip(texts, labels, strict=True)],
        shuffle=True,
        batch_size=args.batch_size,
    )
    cross.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=100,
        show_progress_bar=False,
    )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    cross.save(str(OUTPUT_DIR))
    print(f"saved to {OUTPUT_DIR}\n")

    # Rerank each shortlist and re-measure at the same k.
    reranked = np.empty_like(ranked)
    for i, (pair, row) in enumerate(zip(test_pairs, ranked, strict=True)):
        scores = cross.predict(
            [[pair.query, test_profiles[ids[j]]] for j in row], show_progress_bar=False
        )
        reranked[i] = row[np.argsort(-np.asarray(scores))]

    after = recall_at_k(ids, reranked, test_pairs, args.k)
    print(
        f"+ reranker      recall@{args.k}:  PT {after['pt']:.3f}  EN {after['en']:.3f}  "
        f"all {after['all']:.3f}"
    )
    print(
        f"\ndelta: PT {after['pt'] - before['pt']:+.3f}  EN {after['en'] - before['en']:+.3f}  "
        f"all {after['all'] - before['all']:+.3f}"
    )


if __name__ == "__main__":
    main()
