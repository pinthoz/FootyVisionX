"""What the deployed API may load, and the artifacts it cannot run without.

The API used to import scikit-learn, xgboost and lightgbm and load three fitted models:
405MB warm and 505MB at peak against a 512MB container, which is how it came to be
restarted under load. It now serves their outputs from files under models/ and imports
none of them. Nothing but these tests stops a single misplaced import from quietly
putting the 250MB back.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from footyvision.ml import precompute

# The libraries the `train` extra carries. A deployment installs none of them.
TRAINING_ONLY = ("sklearn", "xgboost", "lightgbm", "shap", "rapidfuzz")


def _loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and report which training modules it loaded.

    A subprocess because this one has already imported all of them for other tests.
    """
    probe = f"{code}\nimport sys\nprint(','.join(m for m in {TRAINING_ONLY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {m for m in out.splitlines()[-1].split(",") if m} if out else set()


def test_importing_the_api_loads_no_training_library():
    assert _loaded_after("import footyvision.api.main") == set()


def test_loading_sofifa_data_needs_no_model_library():
    """Its value parser used to live in ml/value.py, so reading a CSV imported lightgbm."""
    assert _loaded_after("import footyvision.etl.sofifa") == set()


def test_the_value_artifact_is_plain_json():
    """It was a pickle, and a pickle carries its writer's libraries into every reader. The
    first held a fitted LGBMRegressor, so reading eight scalars imported lightgbm; the
    second held a pandas 3 frame whose text columns were pyarrow arrays, so it would not
    load at all where pyarrow is absent — which is CI and Render. Parsing it with nothing
    but the standard library is the guarantee that neither can happen again."""
    import json

    from footyvision.api.routers.value import ARTIFACT

    assert ARTIFACT.is_file(), f"missing {ARTIFACT}: run `footyvision value-report`"
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert {"r2_eur", "baseline_mae_eur", "interval_coverage"} <= set(payload["metrics"])
    assert payload["priced"] and {"player_id", "predicted_low", "predicted_high"} <= set(
        payload["priced"][0]
    )


def test_the_value_endpoints_load_without_a_training_library():
    code = "from footyvision.api.routers.value import load_artifact; load_artifact()"
    assert _loaded_after(code) == set()


def test_published_predictions_carry_what_the_api_reads():
    payload = precompute.load()
    assert payload is not None, f"missing {precompute.ARTIFACT}: run `footyvision precompute`"
    assert set(payload["models"]) == set(precompute.TARGETS)
    for meta in payload["models"].values():
        assert {"classes", "test_accuracy", "balanced_accuracy", "per_class_recall"} <= set(meta)
    assert payload["players"] and payload["top_features"]
    assert {"db_matches", "played_in", "None"} <= set(payload.get("teams", {}))


def _stale_payload() -> dict:
    return {"built_on": "2026-01-01", "pool": 99_999, "players": {}, "models": {}}


def test_a_stale_file_is_served_and_marked_when_refitting_is_impossible(monkeypatch):
    """In production the training extra is absent. Refitting there would cost 250MB and
    restart the instance, so the old answers are served with a flag saying so."""
    from footyvision.api.routers import talent as router

    def cannot_train(frame):
        raise ImportError("No module named 'xgboost'")

    monkeypatch.setattr(router.precompute, "load", _stale_payload)
    monkeypatch.setattr(router.precompute, "build", cannot_train)
    monkeypatch.setattr(router, "_PREDICTIONS", {})

    served = router._predictions(pd.DataFrame({"player_id": [1, 2, 3]}))
    assert served["stale"] is True
    assert served["built_on"] == "2026-01-01"


def test_no_file_and_no_training_stack_is_a_503_not_a_crash(monkeypatch):
    from fastapi import HTTPException

    from footyvision.api.routers import talent as router

    def cannot_train(frame):
        raise ImportError("No module named 'xgboost'")

    monkeypatch.setattr(router.precompute, "load", lambda: None)
    monkeypatch.setattr(router.precompute, "build", cannot_train)
    monkeypatch.setattr(router, "_PREDICTIONS", {})

    with pytest.raises(HTTPException) as refused:
        router._predictions(pd.DataFrame({"player_id": [1]}))
    assert refused.value.status_code == 503
    assert "footyvision precompute" in refused.value.detail


def test_published_pitch_maps_serve_a_player(client):
    """The pitch maps are downloaded from a gigabyte of event JSON the deployment cannot
    fetch, so the published file is the only way the endpoint answers at all."""
    from footyvision.etl import spatial

    payload = spatial.load()
    assert payload is not None, f"missing {spatial.ARTIFACT}: run `footyvision spatial`"
    season = max(payload["players"].values(), key=lambda s: s["actions"])

    body = client.get(f"/players/{season['player_id']}/pitch").json()
    assert len(body["cells"]) == body["cols"] * body["rows"]
    assert sum(body["cells"]) == body["actions"] > 0
    assert {"zone_mean_x", "zone_mean_y"} <= set(body["zones"])
    assert client.get("/players/999999999/pitch").status_code == 404
