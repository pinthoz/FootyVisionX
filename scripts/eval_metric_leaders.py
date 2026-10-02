"""Does "who makes the most X" retrieve the players who make the most X?

Nothing else in the evaluation can tell. RAGAS faithfulness checks that the answer sticks
to the retrieved profiles, and context precision that it uses them; both are satisfied by
an answer that names the best of the wrong six players. A leaders question has a right
answer in the database, so it can be checked against it directly.

For every metric question in the RAGAS bank this computes the true top six — highest
per-90 value among players of the role the question names, after the same hard filters
the assistant applies — and reports how many of them the retrieval returned:

    leaders@6   share of the true top six among the six returned
    top-1       whether the actual leader was returned at all
    role        share of the returned players who play the role asked about

at several sizes of the similarity-ranked candidate pool the metric reorders. Pool 0 is
the old behaviour: similarity alone, no reordering. Retrieval only, no model: offline,
free, deterministic.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragas_questions import all_questions  # noqa: E402

from footyvision.config import get_settings  # noqa: E402
from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.ml.features import load_feature_frame  # noqa: E402
from footyvision.rag.assistant import ScoutAssistant  # noqa: E402
from footyvision.rag.constraints import parse_constraints  # noqa: E402
from footyvision.rag.metrics import asks_for_leaders, metrics_in  # noqa: E402
from footyvision.rag.service import attach_roles  # noqa: E402
from footyvision.rag.store import VectorStore  # noqa: E402

POOLS = (0, 40, 160, None)
K = 6

# Role phrases, most specific first, mapped to `position_role`. The assistant's own filters
# stop at the four position groups, which is exactly why the role has to be recovered here
# to know what a correct answer would contain.
ROLES: tuple[tuple[str, str], ...] = (
    ("centre forward", "Centre Forward"),
    ("center forward", "Centre Forward"),
    ("ponta de lanca", "Centre Forward"),
    ("pontas de lanca", "Centre Forward"),
    ("striker", "Centre Forward"),
    ("wing-back", "Wing Back"),
    ("wing back", "Wing Back"),
    ("full-back", "Full Back"),
    ("full back", "Full Back"),
    ("laterais", "Full Back"),
    ("lateral", "Full Back"),
    ("centre back", "Centre Back"),
    ("centre-back", "Centre Back"),
    ("center back", "Centre Back"),
    ("defesas centrais", "Centre Back"),
    ("defesa central", "Centre Back"),
    ("attacking midfield", "Attacking Midfield"),
    ("medios ofensivos", "Attacking Midfield"),
    ("medio ofensivo", "Attacking Midfield"),
    ("defensive midfield", "Defensive Midfield"),
    ("medios defensivos", "Defensive Midfield"),
    ("medio defensivo", "Defensive Midfield"),
    ("goalkeeper", "Goalkeeper"),
    ("guarda-redes", "Goalkeeper"),
    ("winger", "Winger"),
    ("extremo", "Winger"),
    # A wing-back, as rag/constraints reads it: "medio ala" is a midfielder and "ala"
    # alone is the wide defender.
    ("alas", "Wing Back"),
)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def role_of(question: str) -> str | None:
    text = _fold(question)
    for phrase, role in ROLES:
        if re.search(rf"\b{re.escape(phrase)}", text):
            return role
    return None


def main() -> None:
    with SessionLocal() as session:
        loaded = VectorStore.load_db(session)
        frame = load_feature_frame(session, get_settings().min_minutes)
    if loaded is None:
        sys.exit("No index in the database: run `footyvision index` first.")
    store, _ = loaded
    with SessionLocal() as session:
        attach_roles(session, store)
    # One role per player: the season they played most of, which is the profile indexed.
    frame = frame.sort_values("minutes", ascending=False).drop_duplicates("player_id")
    roles = dict(zip(frame["player_id"].astype(int), frame["position_role"], strict=True))

    assistant = ScoutAssistant(store)
    questions = [
        q.text
        for q in all_questions()
        if q.kind == "metric" and metrics_in(q.text) and asks_for_leaders(q.text)
    ]
    print(f"{len(questions)} leaders questions from the RAGAS bank\n")

    results: dict[int, list[tuple[float, float, float]]] = {pool: [] for pool in POOLS}
    for question in questions:
        metric = metrics_in(question)[0]
        role = role_of(question)
        constraints = parse_constraints(question, store.countries)
        mask = store.matching(constraints) if constraints else [True] * len(store)
        eligible = [
            i
            for i in range(len(store))
            if mask[i]
            and store.metrics[i]
            and metric in store.metrics[i]
            and (role is None or roles.get(int(store.ids[i])) == role)
        ]
        eligible.sort(key=lambda i: store.metrics[i][metric], reverse=True)
        truth = [int(store.ids[i]) for i in eligible[:K]]
        if not truth:
            continue

        for pool in POOLS:
            found = [h.player_id for h in assistant.retrieve(question, K, ranking_pool=pool).hits]
            overlap = len(set(found) & set(truth)) / len(truth)
            top1 = float(truth[0] in found)
            in_role = mean(role is None or roles.get(p) == role for p in found) if found else 0.0
            results[pool].append((overlap, top1, in_role))

    print(f"{'candidate pool':>16} {'leaders@6':>10} {'top-1':>7} {'role':>7}")
    for pool in POOLS:
        rows = results[pool]
        name = {0: "0 (similarity)", None: "all filtered"}.get(pool, str(pool))
        print(
            f"{name:>16} {mean(r[0] for r in rows):10.3f} {mean(r[1] for r in rows):7.3f} "
            f"{mean(r[2] for r in rows):7.3f}"
        )


if __name__ == "__main__":
    main()
