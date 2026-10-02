"""Does knowing where a player acts fix what the position classifiers get wrong?

Two failures were traced to the features having no location at all:

  * Wing Back had 0.00 recall. Ten of fourteen held-out wing-backs were predicted as
    full-backs; the difference between the two is how high they play, which no per-90
    count records.
  * 42% of the exact-position model's errors were pure left/right swaps. Preferred foot
    was the only feature with a side in it.

`footyvision spatial` reduced every match's events to a pitch grid per player-season, and
`etl.spatial.features` summarises it: mean height, mean width (which carries the side),
spread, and the share of actions in the final third, own third, each flank and the wide
channels. This trains each classifier twice on identical splits — per-90 features as in
production, then the same plus those zone features — over several seeds, so a difference
has to survive more than one lucky split to count.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.config import get_settings  # noqa: E402
from footyvision.db.base import SessionLocal  # noqa: E402
from footyvision.etl import spatial  # noqa: E402
from footyvision.ml.features import load_feature_frame  # noqa: E402
from footyvision.ml.talent import MIN_CLASS_SIZE, classifier_features  # noqa: E402

SEEDS = (42, 7, 13, 101, 2024)
TARGETS = ("position_group", "position_role", "primary_position")
# The zone features that say nothing about which side: for the targets that ignore the
# side, the rest may be noise, as preferred foot turned out to be.
SIDELESS = (
    "zone_final_third",
    "zone_mean_x",
    "zone_own_third",
    "zone_spread_x",
    "zone_spread_y",
    "zone_wide",
)


def side(position: str) -> str:
    return position.replace("Left ", "").replace("Right ", "").strip()


def main() -> None:
    payload = spatial.load()
    if payload is None:
        sys.exit("No models/spatial/spatial.json: run `footyvision spatial` first.")
    with SessionLocal() as session:
        frame = load_feature_frame(session, get_settings().min_minutes)

    zones = {}
    for key, season in payload["players"].items():
        zones[key] = spatial.features(season["grid"])
    keys = (
        frame["player_id"].astype(int).astype(str)
        + "-"
        + frame["competition_id"].astype(int).astype(str)
        + "-"
        + frame["sb_season_id"].astype(int).astype(str)
    )
    zone_cols = sorted(next(iter(z for z in zones.values() if z)).keys())
    for col in zone_cols:
        frame[col] = [zones.get(k, {}).get(col, np.nan) for k in keys]
    covered = frame[zone_cols[0]].notna().mean()
    print(f"{len(frame)} player-seasons, {covered:.1%} with a pitch map\n")

    for target in TARGETS:
        data = frame[frame[target] != "Unknown"].copy()
        counts = data[target].value_counts()
        data = data[data[target].isin(counts[counts >= MIN_CLASS_SIZE].index)]
        classes = sorted(data[target].unique())
        code = {c: i for i, c in enumerate(classes)}
        y = data[target].map(code).to_numpy()
        base = classifier_features(data, target)

        print(f"### {target} ({len(classes)} classes)")
        variants = [("per-90 (production)", base), ("per-90 + zones", base + zone_cols)]
        if target != "primary_position":
            variants.append(("per-90 + sideless zones", base + list(SIDELESS)))
        for label, cols in variants:
            X = data[cols].to_numpy(dtype=float)
            runs = []
            for seed in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=seed, stratify=y
                )
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    num_class=len(classes),
                    random_state=seed,
                ).fit(Xtr, ytr)
                pred = model.predict(Xte)
                recall = dict(
                    zip(
                        classes,
                        recall_score(
                            yte, pred, labels=range(len(classes)), average=None, zero_division=0
                        ),
                        strict=True,
                    )
                )
                wrong = [(classes[a], classes[b]) for a, b in zip(yte, pred, strict=True) if a != b]
                swaps = sum(side(a) == side(b) for a, b in wrong)
                runs.append(
                    {
                        "acc": float((pred == yte).mean()),
                        "balanced": float(balanced_accuracy_score(yte, pred)),
                        "wing_back": recall.get("Wing Back", float("nan")),
                        "full_back": recall.get("Full Back", float("nan")),
                        "swap_share": swaps / len(wrong) if wrong else 0.0,
                        "dead": sum(1 for r in recall.values() if r == 0.0),
                    }
                )

            def fmt(k: str, runs: list[dict] = runs) -> str:
                return f"{mean(r[k] for r in runs):.3f}±{pstdev(r[k] for r in runs):.3f}"

            line = f"  {label:25} acc {fmt('acc')}  balanced {fmt('balanced')}"
            if target == "position_role":
                line += f"  wing-back recall {fmt('wing_back')}  full-back {fmt('full_back')}"
            else:
                line += f"  left/right swaps {fmt('swap_share')} of errors"
            line += f"  never-predicted {mean(r['dead'] for r in runs):.1f}"
            print(line)
        print()

    # Which zone features the model leans on, from one fit on everything.
    data = frame[frame["primary_position"] != "Unknown"]
    data = data[data["primary_position"].map(Counter(data["primary_position"])) >= MIN_CLASS_SIZE]
    cols = classifier_features(data, "primary_position") + zone_cols
    classes = sorted(data["primary_position"].unique())
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08, num_class=len(classes), random_state=42
    ).fit(
        data[cols].to_numpy(dtype=float),
        data["primary_position"].map({c: i for i, c in enumerate(classes)}),
    )
    ranked = sorted(zip(cols, model.feature_importances_, strict=True), key=lambda kv: -kv[1])[:8]
    print("top features (exact position):", ", ".join(f"{c} {v:.3f}" for c, v in ranked))


if __name__ == "__main__":
    main()
