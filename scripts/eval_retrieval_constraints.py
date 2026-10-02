"""Score retrieval on what the question actually asked, not on one arbitrary player.

`recall@k` treats a search as correct only when it returns the exact player whose profile
generated the query. But the queries are things like "young centre-back" and "extremo
canhoto que dribla", and the pool holds dozens of players who fit each one equally well.
Under that metric, returning five perfectly good left-footed wingers scores zero unless one
of them happens to be the seed player -- so it measures the wrong thing, and it is the
reason the reranker's hard negatives were largely not negatives at all.

This measures the property a scout would actually check: of the k profiles returned, how
many satisfy each constraint the question stated? Queries are parsed with the very
vocabularies that built them, so the parse is exact rather than inferred.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_embeddings import (  # noqa: E402
    FOOT_WORDS,
    ROLE_WORDS,
    TRAIT_WORDS,
    VETERAN_MIN_AGE,
    YOUNG_MAX_AGE,
    build_pairs,
    build_profiles,
    load_frame,
    split_by_player,
)
from finetune_reranker import SHORTLIST, shortlists  # noqa: E402

from footyvision.config import get_settings  # noqa: E402

K = 5
TRAIT_PERCENTILE = 70.0


def _longest_first(mapping: dict[str, str]) -> list[tuple[str, str]]:
    """Match phrases longest first: 'ambidestro' contains 'destro', 'medio ala' contains
    'ala', and a shortest-first scan would silently pick the wrong one."""
    return sorted(mapping.items(), key=lambda kv: -len(kv[0]))


ROLE_LOOKUP = _longest_first({word: role for role, words in ROLE_WORDS.items() for word in words})
FOOT_LOOKUP = _longest_first({word: foot for foot, words in FOOT_WORDS.items() for word in words})
TRAIT_LOOKUP = _longest_first(
    {word: column for column, words in TRAIT_WORDS.items() for word in words}
)
AGE_LOOKUP = _longest_first(
    {"young": "young", "jovem": "young", "experienced": "old", "experiente": "old"}
)


def parse(query: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for field, lookup in (
        ("role", ROLE_LOOKUP),
        ("foot", FOOT_LOOKUP),
        ("trait", TRAIT_LOOKUP),
        ("age", AGE_LOOKUP),
    ):
        for word, value in lookup:
            if word in query:
                out[field] = value
                break
    return out


def main() -> None:
    import argparse

    import torch
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/embeddinggemma-footyvision",
        help="Which retriever to judge, so a retrained one can be compared against the old.",
    )
    args = parser.parse_args()

    frame = load_frame(get_settings().min_minutes)
    _, ranks = build_profiles(frame)
    pairs = build_pairs(frame)
    _, test_pairs = split_by_player(pairs)

    # Every query is built from a role, so a missing role means the parser is broken.
    unparsed = [p.query for p in test_pairs if "role" not in parse(p.query)]
    assert not unparsed, f"{len(unparsed)} queries did not parse, e.g. {unparsed[:3]}"

    attributes = frame.drop_duplicates("player_id").set_index(
        frame.drop_duplicates("player_id")["player_id"].astype(int)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi = SentenceTransformer(args.model, device=device)
    bi.max_seq_length = 192
    ids, ranked, _ = shortlists(bi, test_pairs, SHORTLIST)

    hits: dict[str, list[float]] = defaultdict(list)
    # The retriever was fine-tuned before the women's leagues were loaded, so it has never
    # seen a female profile. Splitting by the seed player's gender says whether that shows.
    by_gender: dict[tuple[str, str], list[float]] = defaultdict(list)

    for pair, row in zip(test_pairs, ranked, strict=True):
        wanted = parse(pair.query)
        returned = [ids[j] for j in row[:K]]
        seed_gender = str(attributes.loc[pair.player_id].get("gender") or "unknown")
        for field, value in wanted.items():
            ok = []
            for pid in returned:
                attrs = attributes.loc[pid]
                if field == "role":
                    ok.append(attrs["position_role"] == value)
                elif field == "foot":
                    ok.append(attrs["foot"] == value)
                elif field == "age":
                    age = attrs["age"]
                    if pd.isna(age):
                        ok.append(False)
                    else:
                        ok.append(
                            age <= YOUNG_MAX_AGE if value == "young" else age >= VETERAN_MIN_AGE
                        )
                else:
                    percentiles = ranks.get(pid)
                    ok.append(
                        percentiles is not None and percentiles.get(value, 0) >= TRAIT_PERCENTILE
                    )
            share = float(np.mean(ok))
            hits[field].append(share)
            by_gender[(field, seed_gender)].append(share)
        found = float(any(pid == pair.player_id for pid in returned))
        hits["exact player"].append(found)
        by_gender[("exact player", seed_gender)].append(found)

    print(f"\n{len(test_pairs)} test queries, top-{K} judged against the stated constraints\n")
    for field in ("role", "foot", "age", "trait", "exact player"):
        if field in hits:
            print(
                f"  {field:14s} {np.mean(hits[field]):.3f}   (asked in {len(hits[field])} queries)"
            )

    genders = sorted({g for _, g in by_gender})
    if len(genders) > 1:
        print("\nby gender of the player the query was built from:\n")
        header = "".join(f"{g:>14s}" for g in genders)
        print(f"  {'':14s}{header}")
        for field in ("role", "foot", "age", "trait", "exact player"):
            cells = ""
            for gender in genders:
                values = by_gender.get((field, gender))
                cells += f"{np.mean(values):>14.3f}" if values else f"{'-':>14s}"
            if any(c.strip() != "-" for c in cells.split()):
                print(f"  {field:14s}{cells}")
        counts = "".join(f"{len(by_gender.get(('role', g), [])):>14d}" for g in genders)
        print(f"  {'queries':14s}{counts}")


if __name__ == "__main__":
    main()
