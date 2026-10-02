"""Unit tests for the market-value predictor — synthetic data, no scraping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES
from footyvision.ml.value import match_values, parse_value_eur, train_value_model


def test_parse_value_eur():
    assert parse_value_eur("€5M") == 5_000_000
    assert parse_value_eur("€900K") == 900_000
    assert parse_value_eur("€27.5M") == 27_500_000
    assert parse_value_eur(4_000_000) == 4_000_000
    assert parse_value_eur("") == 0.0
    assert parse_value_eur(None) == 0.0


def test_match_values_fuzzy():
    features = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Lionel Messi", "Luis Suárez", "Nobody Here"],
            "position_group": ["FWD", "FWD", "MID"],
        }
    )
    values = pd.DataFrame(
        {
            "name": ["Lionel Andrés Messi", "Luis Suarez", "Cristiano Ronaldo"],
            "value_eur": [120_000_000.0, 80_000_000.0, 100_000_000.0],
        }
    )
    merged = match_values(features, values, score_cutoff=80)
    by_name = dict(zip(merged["name"], merged["value_eur"], strict=False))
    assert by_name["Lionel Messi"] == 120_000_000.0
    assert by_name["Luis Suárez"] == 80_000_000.0
    assert "Nobody Here" not in by_name  # no confident match


def _merged(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feats = {f: float(rng.random()) for f in PER90_FEATURES}
        # Value driven mostly by xg + progressive passes, plus noise.
        value = 1e6 + 4e7 * feats["xg_per90"] + 2e7 * feats["progressive_passes_per90"]
        value *= 1 + rng.normal(0, 0.1)
        rows.append(
            {
                "player_id": i,
                "name": f"P{i}",
                "position_group": "FWD",
                "value_eur": max(1e5, value),
                **feats,
            }
        )
    return pd.DataFrame(rows)


def test_train_value_model_runs_and_reports_metrics():
    vm = train_value_model(_merged())
    assert vm.features == list(PER90_FEATURES)
    assert vm.n_train > 0 and vm.n_test > 0
    assert np.isfinite(vm.r2)
    assert vm.mae_eur >= 0
    # Both scales and the constant to beat, because `r2` alone is the flattering one:
    # it is computed on log(value), where the model looks several times better than it
    # does in euros. Dropping either of these would let the number be read as precision.
    assert np.isfinite(vm.r2_eur)
    assert vm.baseline_mae_eur >= 0


def test_match_values_rejects_substring_collisions():
    """A short value-source name buried inside a long one is not the same footballer.

    Every pair here scored 90 on `fuzz.WRatio` against the real data and was accepted
    before the token rule existed: Amanda Sampedro Bustos was given Pedro's value, and
    Charly Musonda was given Son's. The two genuine matches have the same *shape* — a
    brief name contained in a long one — so the test would pass trivially if the fix had
    simply raised the cutoff.
    """
    features = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "name": [
                "Amanda Sampedro Bustos",
                "Charly Musonda Junior",
                "Carlos Henrique Casimiro",
                "Lionel Andres Messi Cuccittini",
                "Neymar da Silva Santos Junior",
            ],
            "position_group": ["MID"] * 5,
        }
    )
    values = pd.DataFrame(
        {
            "name": ["Pedro", "Son", "Henrique", "Lionel Messi", "Neymar"],
            "value_eur": [23e6, 0.1e6, 1e6, 100e6, 100e6],
        }
    )
    # Casemiro is only rejected because `henrique` belongs to more than one player, which
    # is true of the real squad list and has to be made true here as well — the rule reads
    # the corpus it is given, so a five-row fixture makes every token look distinctive.
    features.loc[len(features)] = [6, "Henrique Adriano Buss", "DEF"]
    # 80, not the default 90: "Lionel Andres Messi Cuccittini" against "Lionel Messi"
    # only scores 85.5, and a cutoff that excludes him tests nothing about the token rule.
    pairs = match_values(features, values, score_cutoff=80)
    matched = dict(zip(pairs["name"], pairs["matched_name"], strict=False))

    assert "Amanda Sampedro Bustos" not in matched
    assert "Charly Musonda Junior" not in matched
    assert "Carlos Henrique Casimiro" not in matched
    # Two shared tokens settles it; one unambiguous token does too.
    assert matched["Lionel Andres Messi Cuccittini"] == "Lionel Messi"
    assert matched["Neymar da Silva Santos Junior"] == "Neymar"


def test_match_values_leaves_ambiguous_single_tokens_unmatched():
    """When one token is all there is and two players could own it, match neither.

    `roberto` is Diego Roberto Godin Leal's middle name and also a Transfermarkt player,
    and nothing in either name says which was meant. Dropping both is the trade the
    module makes: a missing value costs a row, a wrong one costs the label.
    """
    features = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["Diego Roberto Godin Leal", "Roberto Soldado Rillo"],
            "position_group": ["DEF", "FWD"],
        }
    )
    values = pd.DataFrame({"name": ["Roberto"], "value_eur": [7.5e6]})
    assert match_values(features, values, score_cutoff=80).empty
