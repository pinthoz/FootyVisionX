"use client";

import { useEffect, useState } from "react";
import { RagasEvaluation, api } from "../lib/api";

/** Header pill reporting how the scouting assistant scored, with the run behind a click.

    A retrieval-augmented answer reads exactly as confident whether or not it is grounded
    in anything, so a dashboard that shows one owes the reader some evidence about the
    second. These are RAGAS scores from a dated run, served from a snapshot the evaluation
    script writes — not recomputed on a page load, which would cost dozens of LLM calls
    and produce a slightly different number every time. */
export default function AssistantEvaluation() {
  const [data, setData] = useState<RagasEvaluation | null>(null);
  const [failed, setFailed] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    let stale = false;
    api
      .assistantEvaluation()
      .then((d) => !stale && setData(d))
      .catch(() => !stale && setFailed(true));
    return () => {
      stale = true;
    };
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setIsOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen]);

  // Nothing published means nothing to claim. A pill reading 0.00 would be read as a
  // failing score rather than as an absence.
  if (failed) return null;

  const faithfulness = data?.metrics.faithfulness ?? null;

  return (
    <>
      <button
        className="coverage-trigger"
        onClick={() => setIsOpen(true)}
        title="How the scouting assistant was evaluated"
      >
        <span className="coverage-trigger-label">Assistant:</span>
        {data && faithfulness !== null ? (
          <span className="coverage-trigger-value">
            {pct(faithfulness)} faithful · {data.questions} questions
          </span>
        ) : (
          <span className="coverage-trigger-value dim">…</span>
        )}
      </button>

      {isOpen && data && (
        <div className="modal-backdrop" onClick={() => setIsOpen(false)}>
          <div className="coverage-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="picker-modal-header">
              <div>
                <div className="coverage-header-pill">RAGAS · {data.measured_on}</div>
                <h3 className="picker-title">Assistant Evaluation</h3>
              </div>
              <button className="picker-close-btn" onClick={() => setIsOpen(false)}>
                ✕
              </button>
            </div>

            <div className="coverage-modal-body">
              <div className="coverage-totals">
                <span>
                  <strong>{pct(data.metrics.faithfulness)}</strong> faithfulness
                </span>
                <span>
                  <strong>{pct(data.answerable_relevancy)}</strong> answer relevancy
                </span>
                <span>
                  <strong>{data.questions}</strong> questions
                </span>
              </div>

              <p className="why-note">
                <strong>Faithfulness</strong> is the one that matters here: it checks that
                every claim in an answer traces back to a profile the search actually
                returned. The assistant is told never to invent a player or a statistic,
                and this is the only thing that verifies it did not.
              </p>

              <p className="why-note">
                {data.unanswerable.count} of the {data.questions} questions have no answer
                in this dataset — some ask for a metric we do not hold, others for players
                from leagues never loaded. On those the assistant scored{" "}
                <strong>{pct(data.unanswerable.faithfulness)}</strong> faithfulness by
                declining rather than inventing. Their <em>answer relevancy</em> is 0 by
                the metric&apos;s design, which scores a refusal as irrelevant — right for
                a question that had an answer, wrong for one that never did. The headline
                relevancy above therefore excludes them; including them would reward a
                model that made something up.
              </p>

              <ByKind data={data} />

              <details className="eval-questions">
                <summary>All {data.questions} questions</summary>
                <table className="coverage-table">
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th className="num">Faithful</th>
                      <th className="num">Relevancy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.question}>
                        <td>
                          <span className="coverage-comp">{r.question}</span>
                          <span className="coverage-country">{r.kind.replace("-", " ")}</span>
                        </td>
                        <td className="num">{r.faithfulness.toFixed(2)}</td>
                        <td className="num">
                          {r.kind.startsWith("unanswerable")
                            ? "n/a"
                            : r.answer_relevancy.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>

              <p className="why-note dim">
                Judged by {(data.judges ?? [data.judge_model]).join(", ")}, answering with{" "}
                {(data.answer_models ?? [data.answer_model]).join(", ")}.
                {(data.judges?.length ?? 1) > 1 && (
                  <>
                    {" "}
                    <strong>More than one judge</strong> means a quota ran out mid-run and a
                    second model finished it; different judges calibrate differently, so
                    comparing categories against each other is not safe here.
                  </>
                )}{" "}
                A model
                scoring its own family&apos;s output tends to be generous, so read these as
                a signal across runs rather than an absolute grade. Context precision is
                measured too and deliberately not shown: it has scored 0.00 on questions
                whose answers quote figures straight out of the retrieved profiles, and
                swings by a factor of three between judges, while a deterministic check
                puts retrieval precision at 0.97 on the same index. It is reporting on
                itself rather than on the pipeline.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** The breakdown that a single headline hides.

    One mean over fifty-eight questions says the assistant is broadly grounded and nothing
    about where it is weakest. Split by what was asked, the shape appears: strongest when a
    question names a metric, weakest when it names nothing at all. */
function ByKind({ data }: { data: RagasEvaluation }) {
  const kinds = Object.entries(data.by_kind ?? {});
  if (kinds.length === 0) return null;

  const best = Math.max(...kinds.map(([, k]) => k.faithfulness ?? 0), 0.0001);

  return (
    <>
      <h4 className="eval-subhead">By what was asked</h4>
      <table className="coverage-table">
        <thead>
          <tr>
            <th>Kind of question</th>
            <th className="num">Asked</th>
            <th>Faithfulness</th>
            <th className="num">Relevancy</th>
          </tr>
        </thead>
        <tbody>
          {kinds.map(([kind, k]) => (
            <tr key={kind}>
              <td>
                <span className="coverage-comp">{kind.replace(/-/g, " ")}</span>
                <span className="coverage-country">{DESCRIPTIONS[kind] ?? ""}</span>
              </td>
              <td className="num">{k.count}</td>
              <td>
                <span className="why-shap">
                  <span className="why-bar">
                    <span
                      style={{
                        width: `${((k.faithfulness ?? 0) / best) * 100}%`,
                        background: "var(--accent)",
                      }}
                    />
                  </span>
                  <span className="why-shap-value">{pct(k.faithfulness)}</span>
                </span>
              </td>
              {/* Withheld rather than shown as zero, which reads as a failure when it is
                  the metric declining to grade a correct refusal. */}
              <td className="num">{k.answer_relevancy === null ? "—" : pct(k.answer_relevancy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

const DESCRIPTIONS: Record<string, string> = {
  constraint: "foot, position or age",
  nationality: "the one attribute known for everyone",
  comparison: "named players",
  metric: "a specific per-90 number",
  "coverage-gap": "answerable only in part",
  "unanswerable-metric": "a number we do not hold",
  "unanswerable-player": "someone never loaded",
  vague: "no requirement at all",
};

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}
