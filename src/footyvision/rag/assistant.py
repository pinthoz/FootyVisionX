"""RAG scouting assistant: retrieve relevant player profiles, then answer grounded in them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from footyvision.llm.client import LLMClient
from footyvision.rag.constraints import Constraints, parse_constraints
from footyvision.rag.metrics import asks_for_leaders, label, metrics_in
from footyvision.rag.store import Hit, VectorStore

# How many similarity-ranked candidates a "who leads in X" question is ordered from; None
# means every player the filters admit. Measured on the 42 leaders questions of the RAGAS
# bank with scripts/eval_metric_leaders.py — share of the true top six returned, and how
# often the actual leader is among them:
#
#     similarity alone          0.19   leader returned 29% of the time
#     40 candidates             0.58   64%
#     160 candidates            0.83   83%
#     every filtered player     0.99   100%
#
# A cap only ever helped while the filters could not express a role, so that similarity
# had to find the wing-backs; with the role now a filter, it only loses leaders.
RANKING_POOL: int | None = None


@dataclass
class Retrieval:
    hits: list[Hit]
    constraints: Constraints
    unfilterable: dict[str, int]
    # The per-90 columns the question names, first-mentioned first.
    asked: list[str]
    # The metric the shortlist was ordered by, when the question asked for leaders in one.
    ranked_by: str | None


def build_prompt(
    question: str,
    contexts: list[str],
    constraints: Constraints | None = None,
    unfilterable: dict[str, int] | None = None,
    ranked_by: str | None = None,
) -> tuple[str, str]:
    system = (
        "You are a football scouting assistant. Answer in English, concisely, even when "
        "the question is asked in another language. Base your answer EXCLUSIVELY on the "
        "retrieved players below — never invent players or statistics. Name the players "
        "you use, and if none of them fit, say so honestly. Every profile carries per-90 "
        "values and percentiles: when the question compares players or asks who is best "
        "at some metric, compare those numbers and justify your pick with them. If the "
        "metric asked about is not in the profiles, say you do not have it rather than "
        "guessing."
    )
    if constraints:
        # The pool was narrowed before ranking, so the model must not re-litigate the
        # filter — every candidate shown already satisfies it, and none of them should be
        # rejected for failing a requirement they were selected on.
        system += (
            " The retrieved players have ALREADY been filtered to those who are "
            f"{constraints.describe()}, so treat that requirement as met by all of them "
            "and judge only the rest of the question. If the list is empty, say plainly "
            "that no player in the dataset meets the requirement."
        )
    if unfilterable:
        # Stated as a limit on the search, not as a fact about the players: the shortlist
        # is drawn from a smaller pool than the reader assumes, and that changes what
        # "the best available" means.
        missing = ", ".join(
            f"{count} players have no recorded {field}" for field, count in unfilterable.items()
        )
        # A prohibition, not a request for a sentence. Asking the model to state the
        # limitation made it append one to every constrained answer — sometimes quoting
        # this instruction back verbatim — and a claim about the search is grounded in no
        # profile, so faithfulness counted it as invention on the largest category in the
        # evaluation. The disclosure reaches the reader from `not_considered` instead,
        # where it is exact and cannot be paraphrased into something false.
        system += (
            f" Note that {missing}, so they could not be checked against this requirement "
            "and were left out of the search. Do not mention this in your answer; it is "
            "reported to the reader separately."
            # Without this the model reads its own shortlist as a census. Asked for young
            # players in Liga F it answered "no players in the dataset play in Liga F" —
            # false, there are 275 of them; the age filter had removed every one, because
            # no woman in this data has a recorded date of birth. The retrieved six were
            # men, and it described them accurately and concluded something untrue.
            " The players shown are what survived that filter, NOT the whole database. "
            "Never say a competition, country or group is absent from the dataset merely "
            "because none of the retrieved players belongs to it — say instead that the "
            "requirement could not be checked for them."
        )
    if ranked_by:
        # Without this the model treats the six as a sample and hedges — "among the
        # retrieved players" — when they are in fact the top of the whole filtered pool,
        # in order. It is a fact about how the list was built, so it is stated as one.
        system += (
            f" The retrieved players are the highest in {label(ranked_by)} per 90 of every "
            "player in the dataset who meets the question's requirements, listed from "
            "highest to lowest."
        )
    # One entry per player, each already carrying whatever metric the question asked for.
    # Listing the numbers separately made the model see them twice, and made the evaluation
    # count twice as many contexts — half of them naming players no answer mentions, which
    # is most of why context precision read so low on metric questions.
    joined = "\n".join(f"- {c}" for c in contexts) or "(no player matches the requirement)"
    user = f"Question: {question}\n\nRetrieved players (context):\n{joined}\n\nAnswer:"
    return system, user


class ScoutAssistant:
    def __init__(self, store: VectorStore, client: LLMClient | None = None) -> None:
        self.store = store
        self.client = client or LLMClient()

    def retrieve(
        self, question: str, k: int = 6, ranking_pool: int | None = RANKING_POOL
    ) -> Retrieval:
        """Choose the players an answer will be written from.

        Separate from `answer` so it can be measured without a model in the loop: whether
        the right players reach the prompt is decided entirely here, and nothing the
        evaluation of the written answer checks can see it.
        """
        # Hybrid retrieval: players named in the question are pinned, the rest of the
        # budget is filled by embedding similarity.
        # Hard requirements ("left-footed", "under 23") narrow the pool before ranking:
        # an embedding cannot enforce them, it can only prefer them, and preference is
        # not enough when the constraint is absolute.
        # The index supplies the nationality vocabulary, so only countries someone in the
        # pool actually comes from can become a filter.
        constraints = parse_constraints(question, self.store.countries)
        mask = self.store.matching(constraints) if constraints else None
        # Players the filter removed because the attribute is missing for them, not
        # because they failed it. Reported rather than swallowed: age and foot are only
        # known for the men's leagues, so those two requirements quietly take the women's
        # competitions out of the running, and a silent omission reads as an answer.
        unfilterable = self.store.unknown_dropped(constraints) if constraints else {}

        pinned = self.store.mentioned(question)
        # When the question names players, fill the remaining slots with players whose
        # *style* resembles theirs. Embedding the raw question makes the proper nouns
        # dominate, which retrieves similar-sounding names instead of similar players.
        centroid = self.store.style_centroid([h.player_id for h in pinned]) if pinned else None
        query_vector = (
            centroid if centroid is not None else self.client.embed([question], kind="query")[0]
        )
        asked = metrics_in(question)

        # "Who makes the most tackles?" is a question about an ordering, and similarity
        # cannot answer it: the six profiles that *read* most like the question are not
        # the six with the most tackles, and the model, told to stay inside its context,
        # then names the best of the wrong six — faithfully. So when the question asks for
        # leaders in a metric, the metric orders the players the filters admit, and
        # similarity is kept only as the score shown beside each source. `ranking_pool`
        # 0 switches it off, which is what the evaluation compares against.
        if asked and not pinned and ranking_pool != 0 and asks_for_leaders(question):
            pool = len(self.store) if ranking_pool is None else ranking_pool
            candidates = self.store.search(query_vector, k=pool, mask=mask)
            leaders: list[Hit] = []
            for hit in self.store.ranked_by(candidates, asked[0]):
                # A player who changed league holds two rows; one name, one slot.
                if hit.player_id not in {h.player_id for h in leaders}:
                    leaders.append(hit)
                if len(leaders) == k:
                    break
            return Retrieval(leaders, constraints, unfilterable, asked, asked[0])

        # Under a hard constraint the pinned players are the *seed*, not answers: a named
        # player need not satisfy the requirement ("left-footed wingers like Bale" — Bale
        # is right-footed), and showing him would contradict the promise the prompt makes
        # that every listed player already passes the filter.
        hits = [] if constraints else list(pinned)
        seen = {h.player_id for h in hits}
        for hit in self.store.search(query_vector, k=k + len(hits), mask=mask):
            if len(hits) >= max(k, len(hits)):
                break
            if hit.player_id not in seen:
                hits.append(hit)
                seen.add(hit.player_id)
        return Retrieval(hits, constraints, unfilterable, asked, None)

    def answer(self, question: str, k: int = 6) -> dict[str, Any]:
        found = self.retrieve(question, k)
        hits, constraints, unfilterable = found.hits, found.constraints, found.unfilterable
        # A profile names a player's four best metrics and their weakest. Asked about any
        # other one the model used to answer that it had no such data — true of what it
        # had been given, false of what the database holds. The metric named in the
        # question is looked up and appended to each player's own paragraph: a context is
        # everything known about one player, and listing the numbers separately doubled
        # the count while leaving half of them citing nobody the answer names.
        #
        # Looked up per player rather than zipped by position: `metric_notes` skips a
        # player with no recorded values, so pairing its output with the hits by index
        # would hand one player's figures to the next.
        contexts = []
        for hit in hits:
            note = self.store.metric_notes([hit], found.asked)
            contexts.append(
                f"{hit.text} Asked about: {note[0].split(': ', 1)[-1]}" if note else hit.text
            )

        system, user = build_prompt(
            question, contexts, constraints, unfilterable, ranked_by=found.ranked_by
        )
        # 1400 was not enough. The configured cloud model is a reasoning one, and its
        # thinking is billed against the same ceiling as the words it writes, so answers
        # were being cut mid-sentence with most of the budget already spent before the
        # first one. `last_truncated` says when it happened anyway.
        answer = self.client.chat(system, user, max_tokens=3000)
        truncated = getattr(self.client, "last_truncated", False)
        return {
            "answer": answer,
            # Trimmed back to its last full sentence by the client, so it reads as prose,
            # but it is still a partial reply and the reader is entitled to know.
            "truncated": truncated,
            "filters": constraints.describe() or None,
            # The metric the shortlist is ordered by, as a reader would name it. Set only
            # when the question asked for leaders, so the reader knows the list is the top
            # of the filtered pool in order rather than six profiles that read alike.
            "ranked_by": label(found.ranked_by) if found.ranked_by else None,
            "not_considered": unfilterable or None,
            # The profiles the answer was written from. Absent from `AssistantResponse`,
            # so the HTTP payload is unchanged — this is here because an evaluation of
            # faithfulness needs the exact text the model was grounded in, and rebuilding
            # it afterwards from player ids would be measuring a different retrieval.
            "contexts": contexts,
            "sources": [
                {"player_id": h.player_id, "name": h.name, "score": round(h.score, 3)} for h in hits
            ],
        }
