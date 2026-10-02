"""Published evaluation results.

`ragas.json` is written by `scripts/eval_ragas.py`, not by hand: the numbers the app shows
for how the assistant was evaluated are the numbers that were measured. It is committed
rather than regenerated at runtime because scoring costs dozens of LLM calls and the
result belongs to a moment — a dated statement about a particular index and a particular
model, not something to recompute on a page load.
"""
