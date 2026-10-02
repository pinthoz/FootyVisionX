"""A tiny in-memory vector store (cosine retrieval) over player-profile embeddings.

For ~400 players a numpy matrix is more than enough — no external vector DB needed. Vectors
are persisted to a .npz so the API doesn't re-embed on every start. pgvector/FAISS would be
the swap-in at much larger scale.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footyvision.llm.client import LLMClient
from footyvision.rag.constraints import Constraints


@dataclass
class Hit:
    player_id: int
    name: str
    text: str
    score: float
    # Where this profile sits in the store. A player who changed league mid-season has
    # two rows under one id, so the id alone cannot say which profile was retrieved.
    row: int = -1


def _name_tokens(text: str) -> list[str]:
    """Lowercase, accent-stripped word tokens — so "Bakambu" matches "Cédric Bakambu"."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", folded)


# Ordinary words that also happen to be surnames. Without this, "a **young** winger"
# pins Ashley Young, and because the assistant then queries with the centroid of the
# pinned players instead of the question, one accidental collision hijacks the whole
# retrieval. Filtering the *question* side is safe: someone actually asking about Ashley
# Young still writes "Ashley", which is distinctive and pins him on its own.
_COMMON_WORDS: frozenset[str] = frozenset(
    """
    young older elder best good great strong quick fast pace power powerful
    back backs wing wings wide side sides ball balls foot feet left right
    play plays player players playing style forward forwards striker
    winger wingers defender defenders midfield midfielder keeper goalkeeper
    find give need want show tell like similar compare best most more less than
    that this they them with from what which have take takes make makes
    someone somebody anyone scoring passing tackling dribbling pressing
    creative defensive attacking complete season minutes team teams club
    """.split()
)


# Above this many players sharing a name part, the part names a family rather than a
# person. Chosen from the pool's own distribution: 94% of tokens belong to three players
# or fewer, and every genuinely identifying name — messi, neymar, kante, putellas —
# belongs to exactly one.
MAX_PLAYERS_PER_TOKEN = 3


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, ids, names, texts, vectors: np.ndarray, attrs: dict | None = None) -> None:
        self.ids = np.asarray(ids)
        self.names = np.asarray(names, dtype=object)
        self.texts = np.asarray(texts, dtype=object)
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        # Hard attributes kept beside the vectors so retrieval can narrow the pool before
        # ranking it. An index built before these existed simply has none, and every
        # filter then matches everything rather than failing.
        attrs = attrs or {}
        n = len(self.ids)
        self.foot = np.asarray(attrs.get("foot", [None] * n), dtype=object)
        self.age = np.asarray(attrs.get("age", [np.nan] * n), dtype=np.float32)
        self.position_group = np.asarray(attrs.get("position_group", [None] * n), dtype=object)
        self.nationality = np.asarray(attrs.get("nationality", [None] * n), dtype=object)
        # Not stored with the vectors: it is derived from each player's primary position
        # when the index is loaded (see `rag.service.attach_roles`), so adding it needed
        # neither a migration nor re-embedding 2,562 profiles.
        self.position_role = np.asarray(attrs.get("position_role", [None] * n), dtype=object)
        # Per-90 values per row, as plain dicts. Not a numpy matrix: an index built
        # before these existed has none, and a ragged column of Nones is exactly the
        # "this dimension is unknown" case the filters already handle.
        self.metrics = list(attrs.get("metrics") or [None] * n)
        # Built on first use: one pass over every name, and only questions that name a
        # player ever need it.
        self._common_names: frozenset[str] | None = None

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def countries(self) -> list[str]:
        """The nationalities actually present, so the parser only recognises real ones.

        Reading the vocabulary off the index rather than hardcoding it means a question
        naming a country nobody in the pool comes from yields no filter and falls back to
        ranking, instead of narrowing the pool to nothing and answering "no such player".
        """
        return sorted({str(v) for v in self.nationality if v})

    @classmethod
    def build(cls, docs: list[dict], client: LLMClient, batch_size: int = 32) -> VectorStore:
        texts = [d["text"] for d in docs]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(client.embed(texts[i : i + batch_size]))
        return cls(
            [d["player_id"] for d in docs],
            [d["name"] for d in docs],
            texts,
            np.asarray(vectors, dtype=np.float32),
            attrs={
                "foot": [d.get("foot") for d in docs],
                "age": [d.get("age", np.nan) for d in docs],
                "position_group": [d.get("position_group") for d in docs],
                "nationality": [d.get("nationality") for d in docs],
                "metrics": [d.get("metrics") for d in docs],
            },
        )

    def _shared_tokens(self) -> frozenset[str]:
        """Name parts too many players share to identify any one of them.

        Read off the index rather than listed by hand, because which names are common is a
        property of who was loaded. In this pool 81% of name tokens belong to exactly one
        player and 94% to three or fewer; past that the tail is Iberian given names and
        patronymics — silva (43 players), maria (35), jose (35), suarez (8).

        Without this, "compare Neymar and Luis Suarez" pinned nineteen players, every one
        of them carrying "Luis" or "Suarez" somewhere in their name. The assistant then
        queried with the centroid of all nineteen and answered about none of them.
        """
        if self._common_names is None:
            counts: dict[str, int] = {}
            for name in self.names:
                for token in {t for t in _name_tokens(str(name)) if len(t) >= 4}:
                    counts[token] = counts.get(token, 0) + 1
            self._common_names = frozenset(
                token for token, n in counts.items() if n > MAX_PLAYERS_PER_TOKEN
            )
        return self._common_names

    def mentioned(self, question: str) -> list[Hit]:
        """Players named in the question, matched on a distinctive part of their name.

        Pure embedding search does not reliably retrieve the players a question names —
        "like Bale and Bakambu" can miss both. Pinning them is the lexical half of a
        hybrid retrieval: cheap, exact, and it makes comparisons actually answerable.
        """
        asked = set(_name_tokens(question)) - _COMMON_WORDS - self._shared_tokens()
        hits: list[Hit] = []
        for i, name in enumerate(self.names):
            # Skip very short tokens ("de", "da") — only distinctive ones identify a player.
            if any(tok in asked for tok in _name_tokens(str(name)) if len(tok) >= 4):
                hits.append(Hit(int(self.ids[i]), str(name), str(self.texts[i]), 1.0, i))
        return hits

    def style_centroid(self, player_ids: list[int]) -> np.ndarray | None:
        """The mean profile vector of the given players — a "players like these" query.

        Embedding the question text instead makes proper nouns dominate the vector, which
        retrieves players with similar-*looking names* rather than a similar playing style.
        """
        mask = np.isin(self.ids, player_ids)
        if not mask.any():
            return None
        centroid = self.vectors[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else centroid

    def matching(self, constraints: Constraints) -> np.ndarray:
        """Boolean mask of the players satisfying every hard constraint.

        Two rules, and the difference between them matters:

        * A *player* whose attribute is unknown fails the constraint. Saying "he is
          left-footed" of someone whose foot was never recorded is exactly the invention
          the assistant exists to avoid.
        * A *dimension* the index knows nothing about is not a filter at all. An index
          built before these attributes existed would otherwise exclude everybody and
          leave the assistant answering every positional question with silence.
        """
        mask = np.ones(len(self.ids), dtype=bool)
        if constraints.foot is not None and self._knows(self.foot):
            mask &= self.foot == constraints.foot
        if constraints.position_group is not None and self._knows(self.position_group):
            mask &= self.position_group == constraints.position_group
        if constraints.position_role is not None and self._knows(self.position_role):
            mask &= self.position_role == constraints.position_role
        if constraints.nationality is not None and self._knows(self.nationality):
            mask &= self.nationality == constraints.nationality
        if self._knows(self.age):
            if constraints.max_age is not None:
                mask &= np.nan_to_num(self.age, nan=np.inf) <= constraints.max_age
            if constraints.min_age is not None:
                mask &= np.nan_to_num(self.age, nan=-np.inf) >= constraints.min_age
        return mask

    def metric_notes(self, hits: list[Hit], columns: list[str]) -> list[str]:
        """One line per retrieved player giving the metrics the question asked about.

        The profile prose names a player's four best and one worst metric, so a question
        about anything else arrives at a model whose context does not hold the number.
        Asked which midfielders make the most tackles, the assistant answered that it had
        no tackle data — true of its context, false of the database. These lines close
        that gap without lengthening every profile with seventeen figures nobody asked for.

        Returns nothing when the index predates the stored metrics, which leaves the
        assistant exactly as it was rather than breaking it.
        """
        if not columns:
            return []
        from footyvision.rag.metrics import label

        notes: list[str] = []
        for hit in hits:
            values = self.metrics[hit.row] if 0 <= hit.row < len(self.metrics) else None
            if not values:
                continue
            parts = [
                f"{label(column)} {float(values[column]):.2f} per 90"
                for column in columns
                if column in values
            ]
            if parts:
                notes.append(f"{hit.name}: {', '.join(parts)}.")
        return notes

    def ranked_by(self, hits: list[Hit], column: str) -> list[Hit]:
        """The same players, highest per-90 value of `column` first.

        A player with no recorded value sorts last rather than being dropped: the question
        still deserves six names, and an unknown is not a zero.
        """

        def value(hit: Hit) -> float:
            values = self.metrics[hit.row] if 0 <= hit.row < len(self.metrics) else None
            v = (values or {}).get(column)
            return float(v) if v is not None else float("-inf")

        return sorted(hits, key=value, reverse=True)

    def unknown_dropped(self, constraints: Constraints) -> dict[str, int]:
        """How many players each constraint excluded for want of the attribute, not for
        failing it.

        Failing a filter and being unfilterable look identical in the mask, and they are
        not the same thing. Date of birth and preferred foot come from a men's football
        export, so an age or foot requirement silently removes every player in the
        women's competitions — a real answer to a question the user did not ask. Counting
        the difference lets the answer say so.
        """
        out: dict[str, int] = {}
        if constraints.foot is not None and self._knows(self.foot):
            # Falsy, not just None: the index stores an unrecorded foot as "" in places,
            # and the rest of this class already treats both as unknown.
            out["foot"] = int(sum(1 for v in self.foot if not v))
        if constraints.nationality is not None and self._knows(self.nationality):
            out["nationality"] = int(sum(1 for v in self.nationality if not v))
        if constraints.max_age is not None or constraints.min_age is not None:
            if self._knows(self.age):
                out["age"] = int((~np.isfinite(self.age)).sum())
        return {key: count for key, count in out.items() if count}

    @staticmethod
    def _knows(values: np.ndarray) -> bool:
        """Whether the index recorded this attribute for anyone at all."""
        if values.dtype == object:
            return any(v is not None for v in values)
        return bool(np.isfinite(values).any())

    def search(
        self,
        query_vector: list[float],
        k: int = 6,
        mask: np.ndarray | None = None,
    ) -> list[Hit]:
        """Rank by cosine similarity, optionally within a pre-filtered subset.

        Filtering before ranking rather than after is the point: an embedding cannot
        enforce "left-footed", so asking it to rank the whole pool and hoping the
        constraint survives the top-k does not work.
        """
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        candidates = np.arange(len(self.ids)) if mask is None else np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(scores[candidates])[::-1][:k]]
        return [
            Hit(
                int(self.ids[i]),
                str(self.names[i]),
                str(self.texts[i]),
                float(scores[i]),
                int(i),
            )
            for i in order
        ]

    def save_db(self, session, embed_model: str) -> int:
        """Replace the stored index with this one, in a single transaction."""
        from footyvision.db.models import PlayerVector

        session.query(PlayerVector).delete()
        session.bulk_save_objects(
            [
                PlayerVector(
                    player_id=int(self.ids[i]),
                    name=str(self.names[i]),
                    text=str(self.texts[i]),
                    vector=self.vectors[i].astype(np.float32).tobytes(),
                    dim=int(self.vectors.shape[1]),
                    embed_model=embed_model,
                    foot=self.foot[i] if self.foot[i] else None,
                    age=None if not np.isfinite(self.age[i]) else float(self.age[i]),
                    position_group=self.position_group[i] if self.position_group[i] else None,
                    nationality=self.nationality[i] if self.nationality[i] else None,
                    metrics=json.dumps(self.metrics[i]) if self.metrics[i] else None,
                )
                for i in range(len(self.ids))
            ]
        )
        session.commit()
        return len(self.ids)

    @classmethod
    def load_db(cls, session) -> tuple[VectorStore, str] | None:
        """Read the stored index back, with the name of the model that built it."""
        from footyvision.db.models import PlayerVector

        rows = session.query(PlayerVector).all()
        if not rows:
            return None
        vectors = np.vstack(
            [np.frombuffer(r.vector, dtype=np.float32).reshape(1, r.dim) for r in rows]
        )
        store = cls(
            [r.player_id for r in rows],
            [r.name for r in rows],
            [r.text for r in rows],
            vectors,
            attrs={
                "foot": [r.foot for r in rows],
                "age": [np.nan if r.age is None else r.age for r in rows],
                "position_group": [r.position_group for r in rows],
                "nationality": [r.nationality for r in rows],
                "metrics": [json.loads(r.metrics) if r.metrics else None for r in rows],
            },
        )
        return store, rows[0].embed_model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ids=self.ids,
            names=self.names,
            texts=self.texts,
            vectors=self.vectors,
            foot=self.foot,
            age=self.age,
            position_group=self.position_group,
            nationality=self.nationality,
            metrics=np.asarray([json.dumps(m) if m else "" for m in self.metrics]),
        )

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        data = np.load(path, allow_pickle=True)
        # Indexes written before the attributes existed have none; degrade, do not crash.
        attrs = {
            k: data[k] for k in ("foot", "age", "position_group", "nationality") if k in data.files
        }
        if "metrics" in data.files:
            # Stored as JSON strings because numpy has no dtype for a dict; "" is how an
            # absent one was written.
            attrs["metrics"] = [json.loads(m) if m else None for m in data["metrics"]]
        return cls(data["ids"], data["names"], data["texts"], data["vectors"], attrs)
