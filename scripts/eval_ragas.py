"""Score the scouting assistant with RAGAS.

The evaluation already in this repo (`eval_retrieval_constraints.py`) measures retrieval
against ground truth we can compute: a question asking for left-footed wingers either got
left-footed wingers back or it did not. That says nothing about the half of the pipeline
that follows — whether the written answer is actually supported by the profiles retrieved,
or whether it quietly invents a number.

That is what RAGAS judges, with an LLM reading the answer against its sources. Three
metrics, none of which needs a hand-written reference answer:

  faithfulness        every claim in the answer traced back to a retrieved profile. The
                      one that matters most here: the system prompt promises never to
                      invent players or statistics, and this is the only thing that checks.
  context precision   whether the profiles retrieved were relevant to the question, judged
                      without a reference, so it measures retrieval rather than agreement
                      with an answer key someone wrote.
  answer relevancy    whether the answer addresses what was asked, rather than something
                      adjacent it happened to retrieve.

Read the numbers knowing who produced them. The judge is Gemini and so is the assistant,
and a model scoring its own family's output is a known bias in LLM-as-judge evaluation —
it tends to be generous. The scores are useful as a *relative* signal across runs and for
catching outright ungrounded answers; they are not an absolute grade.

Treat context precision with particular suspicion on a small judge model. In the first
run it returned exactly 0.00 for seven of eleven questions, including one whose answer
quotes three players and six figures straight out of the retrieved profiles — while the
deterministic evaluation puts role precision at 0.969 on the same retriever. A metric
disagreeing that hard with a measurement of the same thing is reporting on itself, not on
the pipeline. Faithfulness and answer relevancy held up under inspection; this one did
not, and it should be re-run against a stronger judge before anyone believes it.

Budget before you start. The bank is 241 questions, each costing two LLM calls to answer
and three to score — 1,200 calls for the full set, which is hours of local inference and
far past any free cloud allowance. `--sample 60` runs a seeded subset that still carries
every category and is the same subset every time.

Google meters the free tier *per model per day*, and the allowances vary wildly: 20 a day
for gemini-2.5-flash and gemini-3.6-flash, 500 for gemini-3.1-flash-lite. A run therefore
stops partway through with a 429 that looks like a per-minute limit and is not.

Both stages resume, so a stopped run is not a lost one — but finishing it on a *different*
model makes the panel mixed, and different judges calibrate differently. The snapshot
records every judge that scored a row for exactly that reason. For numbers worth comparing
between categories, grade the whole set with one judge: `--rescore` clears the previous
scores, and a local endpoint has no ceiling to run into.

Usage:
    python scripts/eval_ragas.py --generate     # run the assistant, cache the answers
    python scripts/eval_ragas.py --score        # score the cached run, no LLM generation
    python scripts/eval_ragas.py                # both, resuming whatever is cached

    # A fresh daily allowance, without changing what production is configured to use:
    python scripts/eval_ragas.py --answer-model gemini-3.1-flash-lite \
                                 --judge-model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragas_questions import Question, all_questions  # noqa: E402
from ragas_questions import is_refusal as _is_refusal  # noqa: E402

from footyvision.config import get_settings  # noqa: E402

CACHE = Path(".eval_cache/ragas/run.json")
# Committed, unlike the cache: this is the evidence the app shows for how the
# assistant was evaluated, and it has to survive a clean checkout.
SNAPSHOT = Path("src/footyvision/eval/ragas.json")

# 241 questions: 58 written by hand to probe specific behaviour, the rest generated from
# templates crossed with vocabularies read off the database. See ragas_questions.py.
QUESTIONS: tuple[Question, ...] = all_questions()


# Server-side and temporary: the request was fine and the service could not take it just
# then. Gemini answers 503 when a model is overloaded, and a run of 241 questions meets
# that often enough that stopping on it meant restarting by hand every few minutes.
_TRANSIENT = ("429", "rate limit", "500 internal", "502 bad gateway", "503 service", "504 gateway")


def _is_rate_limit(error: Exception) -> bool:
    """Whether waiting and asking again is the right response to this error."""
    text = str(error).lower()
    return any(marker in text for marker in _TRANSIENT) or "overloaded" in text


def _reason(error: Exception) -> str:
    text = str(error).lower()
    return "rate limited" if "429" in text or "rate limit" in text else "service unavailable"


def with_backoff(call, tries: int = 6, base: float = 30.0):
    """Retry through a rate limit or a temporarily unavailable server, and give up on
    anything else immediately.

    The free Gemini tier allows a handful of requests a minute, and a run of this length
    will meet that ceiling; it also meets 503s when the model is overloaded. Waiting is
    the correct response to both, and to nothing else — retrying a malformed request just
    spends the quota faster.
    """
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:
            if not _is_rate_limit(error) or attempt == tries - 1:
                raise
            wait = base * (attempt + 1)
            print(f"      {_reason(error)}, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def save(rows: list[dict], cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def generate(
    limit: int | None, pause: float, cache: Path, rows: list[dict], model: str | None
) -> list[dict]:
    """Run the real assistant over the questions not already answered.

    Saved after every question, and existing answers are kept. A rate limit two thirds of
    the way through a run used to discard the whole thing; now the same command picks up
    where it stopped.
    """
    from footyvision.db.base import SessionLocal
    from footyvision.llm.client import LLMClient
    from footyvision.rag.assistant import ScoutAssistant
    from footyvision.rag.service import get_store

    questions = QUESTIONS[:limit] if limit else QUESTIONS
    done = {row["question"] for row in rows}
    todo = [q for q in questions if q.text not in done]
    if not todo:
        print(f"  all {len(questions)} answers already cached")
        return rows

    # The session is closed before the walk begins. The index is read once and the loop
    # then spends minutes asleep between questions, which is long enough for the database
    # to hang up — a connection held open across all of that buys nothing and dies loudly.
    with SessionLocal() as session:
        store = get_store(session)

    assistant = ScoutAssistant(store, client=LLMClient(cloud_model=model))
    for i, question in enumerate(todo, 1):
        try:
            result = with_backoff(lambda q=question: assistant.answer(q.text))
        except Exception as error:
            print(f"\n  stopped at {question.text!r}: {error}")
            print(f"  {len(rows)} answers kept; re-run to continue.")
            break
        rows.append(
            {
                "question": question.text,
                "kind": question.kind,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "filters": result.get("filters"),
                "not_considered": result.get("not_considered"),
                # Which model wrote it. Read from settings at publish time, this reported
                # whatever production was configured with rather than what actually
                # answered — and the two diverge the moment a quota forces a switch.
                "answered_by": assistant.client.last_chat_model
                or model
                or get_settings().cloud_llm_model,
            }
        )
        save(rows, cache)
        print(
            f"  [{i}/{len(todo)}] {question.text[:56]:58s} {len(result['contexts'])} contexts",
            flush=True,
        )
        time.sleep(pause)
    return rows


def judge_name(model: str | None, base_url: str | None = None) -> str:
    """The model that will actually grade, resolved the same way `_judge` resolves it.

    This existed in two places and they disagreed. `_judge` falls back to the *local*
    model setting when a base URL is given; the provenance fields fell back to the *cloud*
    one, so a run graded on LM Studio was published as having been graded by Gemini —
    which is precisely the confusion `judged_by` was added to prevent.

    With a local endpoint the requested name is not the last word either: the server
    serves whatever it has loaded, whatever it was asked for. So it is asked.
    """
    settings = get_settings()
    if not base_url:
        return model or settings.cloud_llm_model
    try:
        import httpx

        loaded = httpx.get(f"{base_url}/models", timeout=5).json().get("data", [])
        # Only when it is unambiguous. Several loaded models mean the name in the request
        # decides, and guessing between them would be the same mistake in a new place.
        if len(loaded) == 1 and loaded[0].get("id"):
            return str(loaded[0]["id"])
    except Exception:
        pass
    return model or settings.llm_model


def _judge(model: str | None, base_url: str | None = None):
    """A RAGAS LLM and embedder, cloud by default and local when a base URL is given.

    A local endpoint is the only way to grade the whole set with one judge: the cloud free
    tier runs out partway through and finishing on a second model makes the panel mixed.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    settings = get_settings()
    if base_url:
        import instructor
        from ragas.llms.base import InstructorLLM

        # LM Studio wants a key it never checks; the OpenAI client refuses to send none.
        client = AsyncOpenAI(base_url=base_url, api_key="local")
        # Built by hand rather than through `llm_factory`, which hardcodes
        # instructor's JSON mode and so asks for `response_format: json_object`. LM Studio
        # accepts only `json_schema` or `text` and rejects the request outright, so the
        # mode is the one thing that has to differ for a local judge.
        llm = InstructorLLM(
            client=instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA),
            model=model or settings.llm_model,
            provider="openai",
            max_tokens=4096,
            temperature=0.0,
        )
        return llm, OpenAIEmbeddings(client=client, model=settings.llm_embed_model)

    if not settings.active_cloud_api_key:
        raise SystemExit("No cloud API key configured; RAGAS needs a judge model.")
    # Gemini speaks the OpenAI protocol at this endpoint, which is what RAGAS expects.
    # Async, not sync: every metric's synchronous `score()` is a wrapper that calls the
    # async path underneath, and a sync client makes it fail rather than fall back.
    client = AsyncOpenAI(
        base_url=settings.cloud_llm_base_url, api_key=settings.active_cloud_api_key
    )
    llm = llm_factory(
        model or settings.cloud_llm_model,
        provider="openai",
        client=client,
        # Gemini 2.5 Flash reasons before it answers, and that reasoning is billed against
        # the same budget as the JSON it has to emit. At the default the structured output
        # is cut off mid-object and instructor raises rather than returning a partial score.
        max_tokens=4096,
        # A judge that changes its mind between runs cannot be compared against itself.
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(client=client, model=settings.cloud_llm_embed_model)
    return llm, embeddings


def contexts_for(row: dict) -> list[str]:
    """Everything the answer was legitimately grounded in, not only the profiles.

    The assistant is also told which filter it applied and how many players could not be
    checked against it, and it is instructed to say so — "1161 players have no recorded
    foot, so they were left out of the search". That sentence is true and comes from the
    pipeline, but it appears in no profile, so a faithfulness judge shown only the
    profiles marks it as invented. Scoring it that way would push the assistant towards
    dropping the disclosure, which is the opposite of what it should do. The fix is to
    show the judge the same grounding the model had.
    """
    notes: list[str] = []
    if row.get("filters"):
        notes.append(f"Retrieval note: the pool was filtered to players who are {row['filters']}.")
    for field, count in (row.get("not_considered") or {}).items():
        notes.append(
            f"Retrieval note: {count} players have no recorded {field}, so they could not be "
            f"checked against this requirement and were left out of the search."
        )
    return list(row["contexts"]) + notes


def score(
    rows: list[dict],
    pause: float,
    cache: Path,
    model: str | None,
    base_url: str | None = None,
) -> dict[str, list[float]]:
    """Score every cached answer that has not been scored yet, saving as it goes."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    llm, embeddings = _judge(model, base_url)
    # Resolved once, from the same place `_judge` resolves it, so the label on every row
    # names the thing that graded it.
    judge = judge_name(model, base_url)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }

    pending = [row for row in rows if "scores" not in row]
    for i, row in enumerate(pending, 1):
        question, answer = row["question"], row["answer"]
        contexts = contexts_for(row)
        try:
            values = {}
            for name, metric in metrics.items():
                # Answer relevancy asks whether the response fits the question, so it is
                # the one metric that takes no contexts.
                kwargs = {"user_input": question, "response": answer}
                if name != "answer_relevancy":
                    kwargs["retrieved_contexts"] = contexts
                values[name] = float(with_backoff(lambda m=metric, k=kwargs: m.score(**k)).value)
                time.sleep(pause)
        except Exception as error:
            print(f"\n  stopped scoring at {question!r}: {error}")
            print(f"  {len(rows) - len(pending) + i - 1} rows scored; re-run to continue.")
            break
        row["scores"] = values
        # Which judge produced them. A run interrupted by a daily quota gets finished by a
        # different model, and the snapshot's single `judge_model` would then describe only
        # the last one — quietly presenting a mixed panel as one grader.
        row["judged_by"] = judge
        save(rows, cache)
        line = "  ".join(f"{n.split('_')[0]} {v:.2f}" for n, v in values.items())
        print(f"  [{i}/{len(pending)}] {row['kind']:20s} {line}", flush=True)

    scored: dict[str, list[float]] = {name: [] for name in metrics}
    for row in rows:
        for name, value in row.get("scores", {}).items():
            scored[name].append(value)
    return scored


def _grouped(rows: list[dict]) -> dict[str, list[dict]]:
    """Scored rows by question kind, in the order the kinds first appear."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["kind"], []).append(row)
    return out


def write_snapshot(rows: list[dict], path: Path, judge: str, answerer: str) -> None:
    """Publish the run as a small, dated artifact the API can serve.

    The cache holds every answer and every retrieved profile, which is what debugging
    needs and far more than a reader does. This keeps the shape of the result: the means,
    the per-question scores, and the two caveats that a bare average hides. Written by the
    script rather than typed by hand, so the numbers on the page are the numbers that were
    measured.
    """
    scored = [r for r in rows if "scores" in r]
    # A partial run must never replace a complete one. This snapshot is what the dashboard
    # shows as its evidence, and a rate limit two thirds of the way through would otherwise
    # quietly downgrade "58 questions, faithfulness 0.96" to whatever the first fourteen
    # happened to score — with nothing on the page to say so.
    if len(scored) < len(QUESTIONS):
        print(
            f"\nnot published: {len(scored)} of {len(QUESTIONS)} questions scored. "
            f"Re-run to finish; {path} keeps the last complete run."
        )
        return

    answerable = [r for r in scored if not _is_refusal(r["kind"])]
    refusals = [r for r in scored if _is_refusal(r["kind"])]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    payload = {
        "measured_on": datetime.now(UTC).date().isoformat(),
        "answer_model": answerer,
        # Every model that wrote an answer, read off the rows rather than assumed.
        "answer_models": sorted({r.get("answered_by", answerer) for r in scored}),
        "judge_model": judge,
        # Every model that scored a row. More than one means the panel was mixed, which
        # makes comparisons *between* categories unreliable: different judges calibrate
        # differently, and the difference is not separable from the signal.
        "judges": sorted({r.get("judged_by", judge) for r in scored}),
        "questions": len(scored),
        "metrics": {
            name: mean([r["scores"][name] for r in scored])
            for name in ("faithfulness", "context_precision", "answer_relevancy")
        },
        # The questions with no answer in the data are a separate test: there a perfect
        # faithfulness means the assistant declined rather than invented, and answer
        # relevancy is zero by the metric's design for exactly that reason.
        "unanswerable": {
            "count": len(refusals),
            "faithfulness": mean([r["scores"]["faithfulness"] for r in refusals]),
        },
        "answerable_relevancy": mean([r["scores"]["answer_relevancy"] for r in answerable]),
        # Per kind, because a single headline hides that the assistant is strong on
        # constraints and weakest where the data itself is partial.
        "by_kind": {
            kind: {
                "count": len(group),
                "faithfulness": mean([r["scores"]["faithfulness"] for r in group]),
                "answer_relevancy": (
                    None
                    if _is_refusal(kind)
                    else mean([r["scores"]["answer_relevancy"] for r in group])
                ),
            }
            for kind, group in _grouped(scored).items()
        },
        "rows": [
            {
                "question": r["question"],
                "kind": r["kind"],
                **{k: round(v, 3) for k, v in r["scores"].items()},
            }
            for r in scored
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsnapshot written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Only run the assistant.")
    parser.add_argument("--score", action="store_true", help="Only score the cached run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pause",
        type=float,
        default=5.0,
        help=(
            "Seconds between calls. The free tier meters requests per minute per model — "
            "measured at 15 for gemini-3.1-flash-lite — and answering one question costs "
            "two of them, the question embedded and then the answer written. Five seconds "
            "leaves margin; four sat exactly on the limit and tripped it."
        ),
    )
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Evaluate a random but seeded subset. The full bank is 241 questions, which "
            "is 1,200 LLM calls and hours of local inference; a sample of 60 still carries "
            "every category and is the same 60 on every run."
        ),
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Drop existing scores so the whole set is graded by one judge.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SNAPSHOT,
        help="Where to publish the dated summary the API serves.",
    )
    # Google meters the free tier per model per day, and for some models that budget
    # is twenty requests -- less than one run of this script. Overriding the model is
    # how you get a fresh allowance without touching what production is configured to
    # use, and pointing both at a local endpoint removes the ceiling entirely.
    parser.add_argument("--answer-model", default=None, help="Model that writes the answers.")
    parser.add_argument("--judge-model", default=None, help="Model that scores them.")
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Point the judge at a local endpoint, e.g. http://localhost:1234/v1.",
    )
    args = parser.parse_args()

    do_generate = args.generate or not args.score
    do_score = args.score or not args.generate

    # Both stages resume from the cache, so a rate limit costs the questions still to do
    # rather than the ones already paid for.
    if args.sample:
        # Seeded, so the sample is the same set every time and two runs are comparable.
        # Taken from the shuffled bank, which already mixes the categories together.
        global QUESTIONS
        QUESTIONS = tuple(
            random.Random(7).sample(list(QUESTIONS), min(args.sample, len(QUESTIONS)))
        )

    rows: list[dict] = []
    if args.cache.exists():
        rows = json.loads(args.cache.read_text(encoding="utf-8"))

    if do_generate:
        print("Running the assistant over the question set...")
        rows = generate(args.limit, args.pause, args.cache, rows, args.answer_model)
        print(f"\n{len(rows)} answers cached in {args.cache}\n")

    if not do_score:
        return
    if args.limit:
        rows = rows[: args.limit]

    if args.rescore:
        for row in rows:
            row.pop("scores", None)
            row.pop("judged_by", None)
        save(rows, args.cache)
        print(f"cleared previous scores on {len(rows)} rows\n")

    print("Scoring with RAGAS...")
    scores = score(rows, args.pause, args.cache, args.judge_model, args.judge_base_url)

    report(rows, scores)
    write_snapshot(
        rows,
        args.snapshot,
        judge=judge_name(args.judge_model, args.judge_base_url),
        answerer=args.answer_model or get_settings().cloud_llm_model,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def report(rows: list[dict], scores: dict[str, list[float]]) -> None:
    """Print the means, then the two things a bare mean gets wrong here.

    Answer relevancy detects a refusal and scores it zero on purpose, so the questions
    written specifically to have no answer drag the average down for behaving correctly.
    Averaging them in would reward an assistant that made something up.
    """
    print("\n" + "-" * 58)
    for name, values in scores.items():
        print(f"  {name:20s} {_mean(values):.3f}   (n={len(values)})")

    scored = [r for r in rows if "scores" in r]
    answerable = [r for r in scored if not _is_refusal(r["kind"])]
    refusals = [r for r in scored if _is_refusal(r["kind"])]

    if refusals:
        print(
            f"\n  On the {len(refusals)} questions the data cannot answer, faithfulness is "
            f"{_mean([r['scores']['faithfulness'] for r in refusals]):.3f}\n"
            "  — the assistant declined rather than inventing. Their answer relevancy is 0 "
            "by\n  design: the metric scores a refusal as irrelevant, which is right for a "
            "question\n  that had an answer and wrong for one that never did."
        )
    if answerable:
        print(
            f"\n  Answer relevancy on the {len(answerable)} answerable questions only: "
            f"{_mean([r['scores']['answer_relevancy'] for r in answerable]):.3f}"
        )

    print("\n  by kind:\n")
    for kind, group in _grouped(scored).items():
        faith = _mean([r["scores"]["faithfulness"] for r in group])
        # Relevancy is meaningless where a refusal is the right answer, so it is withheld
        # rather than printed as a zero somebody will read as a failure.
        relevancy = (
            "    n/a"
            if _is_refusal(kind)
            else f"{_mean([r['scores']['answer_relevancy'] for r in group]):7.3f}"
        )
        print(f"    {kind:22s} n={len(group):3d}   faithful {faith:.3f}   relevancy {relevancy}")

    print("\n  per question:")
    for row in scored:
        s = row["scores"]
        print(
            f"    {row['kind']:20s} f {s['faithfulness']:.2f}  cp {s['context_precision']:.2f}"
            f"  ar {s['answer_relevancy']:.2f}   {row['question'][:44]}"
        )
    print("-" * 58)


if __name__ == "__main__":
    main()
