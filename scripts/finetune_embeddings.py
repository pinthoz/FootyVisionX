"""LoRA fine-tune EmbeddingGemma on FootyVision's own retrieval task, and measure it.

Why this model rather than the chat one: the assistant's weakness is measured and
specific. `eval_embeddings.py` scores EmbeddingGemma at 0.97 on English queries and 0.73
on their Portuguese twins, because the profiles are written in English while the dashboard
is used in Portuguese. That is a retrieval problem, so the retriever is what gets trained.

Deliberately self-contained: it reads the database directly and builds its own profile
text and query pairs rather than importing `ml.features` or `rag.profiles`. Those modules
do not currently expose foot, age or a side-agnostic role, and this script has no business
changing them — the columns it needs are already in Postgres.

What is trained: `google/embeddinggemma-300m` in fp16 from the Hub (not the quantized QAT
copy LM Studio serves — quantized weights are not trainable), with LoRA adapters, because
LM Studio holds about 4GB of this 6GB card for the models the app is serving.

Usage:
    python scripts/finetune_embeddings.py --eval-only      # baseline, no training
    python scripts/finetune_embeddings.py --epochs 1
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.config import get_settings  # noqa: E402
from footyvision.db.base import engine  # noqa: E402

BASE_MODEL = "google/embeddinggemma-300m"
OUTPUT_DIR = Path("models/embeddinggemma-footyvision")

# The task prefixes the app uses (config.py). An asymmetric retrieval model measurably
# degrades when both sides are encoded the same way.
DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "

METRICS = [
    "goals",
    "assists",
    "shots",
    "xg",
    "passes",
    "passes_completed",
    "progressive_passes",
    "dribbles",
    "dribbles_completed",
    "carries",
    "progressive_carries",
    "tackles",
    "interceptions",
    "blocks",
    "clearances",
    "ball_recoveries",
    "pressures",
]
PER90 = [f"{m}_per90" for m in METRICS]

YOUNG_MAX_AGE = 23.0
VETERAN_MIN_AGE = 32.0


def position_group(position: str | None) -> str:
    if not position:
        return "Unknown"
    p = position.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    if "wing" in p or "forward" in p or "striker" in p:
        return "FWD"
    return "MID"


def position_role(position: str | None) -> str:
    """Side-agnostic role. The metrics are counts and carry no left/right, so the side
    is dropped rather than invited into the labels."""
    if not position:
        return "Unknown"
    p = position.lower()
    if "goalkeeper" in p:
        return "Goalkeeper"
    if "wing back" in p:
        return "Wing Back"
    if "center back" in p:
        return "Centre Back"
    if "back" in p:
        return "Full Back"
    if "defensive midfield" in p:
        return "Defensive Midfield"
    if "attacking midfield" in p:
        return "Attacking Midfield"
    if "center midfield" in p:
        return "Central Midfield"
    if "midfield" in p:
        return "Wide Midfield"
    if "wing" in p:
        return "Winger"
    if "forward" in p or "striker" in p:
        return "Centre Forward"
    return "Unknown"


ROLE_WORDS: dict[str, tuple[str, str]] = {
    "Goalkeeper": ("goalkeeper", "guarda-redes"),
    "Centre Back": ("centre-back", "defesa central"),
    "Full Back": ("full-back", "lateral"),
    "Wing Back": ("wing-back", "ala"),
    "Defensive Midfield": ("defensive midfielder", "médio defensivo"),
    "Central Midfield": ("central midfielder", "médio centro"),
    "Attacking Midfield": ("attacking midfielder", "médio ofensivo"),
    "Wide Midfield": ("wide midfielder", "médio ala"),
    "Winger": ("winger", "extremo"),
    "Centre Forward": ("centre-forward", "ponta de lança"),
}

FOOT_WORDS = {
    "left": ("left-footed", "canhoto"),
    "right": ("right-footed", "destro"),
    "both": ("two-footed", "ambidestro"),
}

# The words a scout types, deliberately NOT the words used inside the profiles: pairing
# English prose with an English paraphrase of itself teaches nothing.
TRAIT_WORDS = {
    "goals_per90": ("scores a lot of goals", "marca muitos golos"),
    "xg_per90": ("gets into dangerous positions", "aparece em zonas perigosas"),
    "shots_per90": ("shoots often", "remata muito"),
    "assists_per90": ("creates chances for others", "cria oportunidades"),
    "passes_per90": ("sees a lot of the ball", "mexe muito na bola"),
    "passes_completed_per90": ("keeps possession safely", "mantém a posse com segurança"),
    "progressive_passes_per90": ("moves the ball forward", "faz a bola progredir"),
    "dribbles_per90": ("takes opponents on", "vai para cima do adversário"),
    "dribbles_completed_per90": ("beats defenders", "passa pelos defesas"),
    "carries_per90": ("carries the ball", "conduz a bola"),
    "progressive_carries_per90": ("drives forward with the ball", "arranca com a bola"),
    "tackles_per90": ("wins tackles", "ganha desarmes"),
    "interceptions_per90": ("reads the game and intercepts", "lê o jogo e interceta"),
    "blocks_per90": ("blocks shots and passes", "corta remates e passes"),
    "clearances_per90": ("clears the danger", "alivia o perigo"),
    "ball_recoveries_per90": ("recovers loose balls", "recupera bolas perdidas"),
    "pressures_per90": ("presses high", "pressiona alto"),
}

STYLE_PHRASES = {
    "goals_per90": "scoring goals",
    "xg_per90": "getting into dangerous scoring positions",
    "shots_per90": "shooting frequently",
    "assists_per90": "creating assists for teammates",
    "passes_per90": "high passing volume",
    "passes_completed_per90": "retaining possession with accurate passing",
    "progressive_passes_per90": "progressing the ball forward with passes",
    "dribbles_per90": "taking opponents on",
    "dribbles_completed_per90": "beating defenders with dribbles",
    "carries_per90": "carrying the ball",
    "progressive_carries_per90": "driving forward with the ball",
    "tackles_per90": "winning tackles",
    "interceptions_per90": "reading the game and intercepting passes",
    "blocks_per90": "blocking shots and passes",
    "clearances_per90": "clearing danger from defence",
    "ball_recoveries_per90": "recovering loose balls",
    "pressures_per90": "pressing opponents high",
}


@dataclass(frozen=True)
class Pair:
    query: str
    profile: str
    player_id: int
    language: str


def load_frame(min_minutes: int) -> pd.DataFrame:
    """Player-seasons with the attributes the profiles and queries need.

    Age is measured at the midpoint of the season the row describes, not today: a
    2015/16 row is a 2015/16 player.
    """
    columns = ", ".join(f"s.{c}" for c in PER90)
    sql = f"""
        SELECT s.player_id, p.name, p.foot, p.date_of_birth, p.height_cm,
               s.primary_position, s.minutes, s.competition_id, s.sb_season_id,
               c.name AS competition, c.gender, {columns}
        FROM player_season_stats s
        JOIN players p ON p.id = s.player_id
        JOIN competitions c ON c.id = s.competition_id
        WHERE s.minutes >= :mm
    """
    frame = pd.read_sql(text(sql), engine, params={"mm": min_minutes})

    spans = pd.read_sql(
        text(
            "SELECT competition_id, sb_season_id, MIN(match_date) AS first, "
            "MAX(match_date) AS last FROM matches GROUP BY 1, 2"
        ),
        engine,
    )
    spans["midpoint"] = (
        pd.to_datetime(spans["first"])
        + (pd.to_datetime(spans["last"]) - pd.to_datetime(spans["first"])) / 2
    )
    merged = frame[["competition_id", "sb_season_id"]].merge(
        spans[["competition_id", "sb_season_id", "midpoint"]],
        on=["competition_id", "sb_season_id"],
        how="left",
    )
    dob = pd.to_datetime(frame["date_of_birth"], errors="coerce")
    years = (merged["midpoint"].to_numpy() - dob.to_numpy()) / np.timedelta64(365, "D")
    frame["age"] = np.round(years.astype(float), 1)

    frame["position_group"] = frame["primary_position"].map(position_group)
    frame["position_role"] = frame["primary_position"].map(position_role)
    return frame


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_profiles(frame: pd.DataFrame) -> tuple[dict[int, str], dict]:
    """One English scouting paragraph per player, plus the within-group percentiles."""
    profiles: dict[int, str] = {}
    ranks: dict = {}
    for group, sub in frame.groupby("position_group"):
        percentiles = sub[PER90].rank(pct=True) * 100.0
        for idx, row in sub.iterrows():
            ranked = percentiles.loc[idx].sort_values(ascending=False)
            ranks[int(row["player_id"])] = ranked
            strengths = ", ".join(
                f"{STYLE_PHRASES[f]} ({row[f]:.2f} per 90, {_ordinal(round(ranked[f]))} percentile)"
                for f in ranked.index[:4]
            )
            bio = []
            if pd.notna(row["age"]):
                bio.append(f"{int(round(row['age']))} years old")
            if isinstance(row["foot"], str) and row["foot"]:
                bio.append("two-footed" if row["foot"] == "both" else f"{row['foot']}-footed")
            if pd.notna(row["height_cm"]):
                bio.append(f"{float(row['height_cm']) / 100:.2f}m tall")
            biography = ", " + ", ".join(bio) if bio else ""
            profiles[int(row["player_id"])] = (
                f"{row['name']} is a {row['primary_position']} ({group}) in "
                f"{row['competition']}{biography}. Playing style: excels at {strengths}. "
                f"Played {int(row['minutes'])} minutes."
            )
    return profiles, ranks


def build_pairs(frame: pd.DataFrame, seed: int = 42, per_player: int = 3) -> list[Pair]:
    """Portuguese and English versions of the same query, paired with the same profile.

    Emitting both languages against one profile is what teaches the model that "extremo
    canhoto" and "left-footed winger" point at the same place.
    """
    rng = random.Random(seed)
    profiles, ranks = build_profiles(frame)
    pairs: list[Pair] = []

    for _, row in frame.iterrows():
        player_id = int(row["player_id"])
        role = row["position_role"]
        if role not in ROLE_WORDS or player_id not in profiles:
            continue
        profile = profiles[player_id]
        role_en, role_pt = ROLE_WORDS[role]

        ranked = ranks[player_id]
        trait = next((f for f in ranked.index if f in TRAIT_WORDS and ranked[f] >= 70), None)
        foot = row["foot"] if isinstance(row["foot"], str) else None

        shapes = [(role_en, role_pt)]
        if foot in FOOT_WORDS:
            foot_en, foot_pt = FOOT_WORDS[foot]
            shapes.append((f"{foot_en} {role_en}", f"{role_pt} {foot_pt}"))
        if pd.notna(row["age"]):
            if float(row["age"]) <= YOUNG_MAX_AGE:
                shapes.append((f"young {role_en}", f"{role_pt} jovem"))
            elif float(row["age"]) >= VETERAN_MIN_AGE:
                shapes.append((f"experienced {role_en}", f"{role_pt} experiente"))
        if trait:
            trait_en, trait_pt = TRAIT_WORDS[trait]
            shapes.append((f"{role_en} who {trait_en}", f"{role_pt} que {trait_pt}"))
            if foot in FOOT_WORDS:
                foot_en, foot_pt = FOOT_WORDS[foot]
                shapes.append(
                    (f"{foot_en} {role_en} who {trait_en}", f"{role_pt} {foot_pt} que {trait_pt}")
                )

        rng.shuffle(shapes)
        for query_en, query_pt in shapes[:per_player]:
            pairs.append(Pair(query_pt, profile, player_id, "pt"))
            pairs.append(Pair(query_en, profile, player_id, "en"))
    return pairs


def split_by_player(
    pairs: list[Pair], holdout_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Hold out whole players. Splitting pairs at random would put a player's Portuguese
    query in training and his English one in test, scoring the model on profiles it was
    explicitly taught to retrieve."""
    rng = random.Random(seed)
    players = sorted({p.player_id for p in pairs})
    rng.shuffle(players)
    held = set(players[: int(len(players) * holdout_fraction)])
    return (
        [p for p in pairs if p.player_id not in held],
        [p for p in pairs if p.player_id in held],
    )


def recall_at_k(model, pairs: list[Pair], k: int = 5) -> dict[str, float]:
    """Fraction of queries whose own player's profile lands in the top k.

    A harsh metric on purpose-built short queries — "left-footed winger" legitimately
    describes dozens of players, so absolute values stay low. It is used to measure
    *movement*, and the PT/EN split is the point rather than the average.
    """
    profiles: dict[int, str] = {}
    for pair in pairs:
        profiles.setdefault(pair.player_id, pair.profile)
    player_ids = list(profiles)
    index = {pid: i for i, pid in enumerate(player_ids)}

    corpus = model.encode(
        [DOC_PREFIX + profiles[pid] for pid in player_ids],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    out: dict[str, float] = {}
    for language in ("pt", "en"):
        subset = [p for p in pairs if p.language == language]
        if not subset:
            continue
        queries = model.encode(
            [QUERY_PREFIX + p.query for p in subset],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        top = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
        out[language] = float(
            np.mean([index[p.player_id] in row for p, row in zip(subset, top, strict=True)])
        )
    out["gap"] = out.get("en", 0.0) - out.get("pt", 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--full", action="store_true", help="Full fine-tune, needs ~4GB free.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=192,
        help="Profiles run to roughly 130 tokens; padding past that is wasted compute.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=2400,
        help="Cap on training pairs. LM Studio holds most of this card, so the binding "
        "constraint is time, not data, and the shapes per player are near-duplicates.",
    )
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"gpu: {torch.cuda.get_device_name(0)} ({free / 1e9:.1f}GB free of {total / 1e9:.1f}GB)"
        )

    frame = load_frame(get_settings().min_minutes)
    pairs = build_pairs(frame)
    train_pairs, test_pairs = split_by_player(pairs)
    print(
        f"{len(pairs)} pairs | train {len(train_pairs)} "
        f"({len({p.player_id for p in train_pairs})} players) | "
        f"test {len(test_pairs)} ({len({p.player_id for p in test_pairs})} players)\n"
    )

    model = SentenceTransformer(BASE_MODEL, device=device)
    model.max_seq_length = args.max_seq_length

    if not args.full:
        from peft import LoraConfig

        model.add_adapter(
            LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA: training {trainable:,} of {total_p:,} parameters "
            f"({100 * trainable / total_p:.2f}%)"
        )

    before = recall_at_k(model, test_pairs, args.k)
    print(
        f"BEFORE  recall@{args.k}:  PT {before['pt']:.3f}   EN {before['en']:.3f}   "
        f"gap {before['gap']:+.3f}",
        flush=True,
    )

    if args.eval_only:
        return

    if args.max_train and len(train_pairs) > args.max_train:
        train_pairs = random.Random(42).sample(train_pairs, args.max_train)
        print(
            f"training on {len(train_pairs)} pairs "
            f"({len({p.player_id for p in train_pairs})} players)",
            flush=True,
        )

    dataset = Dataset.from_dict(
        {
            "anchor": [QUERY_PREFIX + p.query for p in train_pairs],
            "positive": [DOC_PREFIX + p.profile for p in train_pairs],
        }
    )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=(device == "cuda"),
            dataloader_num_workers=0,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=dataset,
        # In-batch negatives: every other profile in the batch is a negative for this
        # query, which is why batch size matters more here than the learning rate.
        loss=MultipleNegativesRankingLoss(model),
    ).train()

    model.save(str(OUTPUT_DIR))
    print(f"\nsaved to {OUTPUT_DIR}")

    after = recall_at_k(model, test_pairs, args.k)
    print(
        f"AFTER   recall@{args.k}:  PT {after['pt']:.3f}   EN {after['en']:.3f}   "
        f"gap {after['gap']:+.3f}"
    )
    print(
        f"\ndelta:  PT {after['pt'] - before['pt']:+.3f}   "
        f"EN {after['en'] - before['en']:+.3f}   "
        f"gap {abs(after['gap']) - abs(before['gap']):+.3f} (negative = narrower)"
    )


if __name__ == "__main__":
    main()
