"""FootyVision command line: initialise the schema and run the ETL.

Examples
--------
    footyvision init-db
    footyvision competitions
    footyvision load --competition 43 --season 3 --limit 5
    footyvision aggregate --competition 43 --season 3
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from footyvision.config import get_settings
from footyvision.db.base import Base, SessionLocal, engine
from footyvision.etl import enrich, load, statsbomb

app = typer.Typer(add_completion=False, help="FootyVision ETL & admin CLI.")
console = Console()


@app.command("init-db")
def init_db() -> None:
    """Create all tables (quick start; use Alembic migrations for production)."""
    import footyvision.db.models  # noqa: F401  (register mappers)

    Base.metadata.create_all(engine)
    console.print("[green]Schema created.[/green]")


@app.command("competitions")
def competitions() -> None:
    """List the competition/season pairs available in StatsBomb Open Data."""
    df = statsbomb.competitions()
    cols = ["competition_id", "season_id", "country_name", "competition_name", "season_name"]
    table = Table(title="StatsBomb Open Data — competitions")
    for col in cols:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(*(str(r[c]) for c in cols))
    console.print(table)


@app.command("load")
def load_cmd(
    competition: int = typer.Option(..., "--competition", "-c"),
    season: int = typer.Option(..., "--season", "-s"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Only load first N matches."),
    aggregate: bool = typer.Option(True, help="Rebuild season aggregates after loading."),
) -> None:
    """Extract + aggregate + load a competition season into Postgres."""
    settings = get_settings()
    with SessionLocal() as session:

        def _progress(match_row, i, total):
            console.print(f"  [{i}/{total}] match {match_row['match_id']} loaded")

        console.print(f"[cyan]Loading competition={competition} season={season}...[/cyan]")
        n = load.load_competition_season(session, competition, season, limit, on_match=_progress)
        console.print(f"[green]{n} matches loaded.[/green]")

        if aggregate:
            written = load.rebuild_season_aggregates(
                session, competition, season, settings.min_minutes
            )
            console.print(
                f"[green]{written} player-season rows aggregated "
                f"(min {settings.min_minutes} minutes).[/green]"
            )


@app.command("aggregate")
def aggregate_cmd(
    competition: int = typer.Option(..., "--competition", "-c"),
    season: int = typer.Option(..., "--season", "-s"),
) -> None:
    """Recompute season aggregates from already-loaded match stats."""
    settings = get_settings()
    with SessionLocal() as session:
        written = load.rebuild_season_aggregates(session, competition, season, settings.min_minutes)
        console.print(f"[green]{written} player-season rows aggregated.[/green]")


@app.command("enrich")
def enrich_cmd(
    birthdates: bool = typer.Option(True, help="Also match Transfermarkt attributes."),
) -> None:
    """Fill nickname/country from lineups, and date of birth, foot and height from Transfermarkt."""
    with SessionLocal() as session:
        console.print("[cyan]Reading lineups for nickname and country...[/cyan]")

        def _progress(match_id, seen, pending):
            if seen % 25 == 0:
                console.print(f"  {seen} matches read, {pending} players still missing detail")

        stats = enrich.enrich_from_lineups(session, on_match=_progress)
        console.print(
            f"[green]{stats['nicknames']} nicknames and {stats['countries']} countries "
            f"filled from {stats['matches_read']} matches "
            f"({stats['still_missing']} players still without either, "
            f"{stats['unavailable']} lineups unavailable).[/green]"
        )

        if birthdates:
            console.print("[cyan]Matching dates of birth, foot and height...[/cyan]")
            dob = enrich.backfill_birthdates(session)
            console.print(
                f"[green]{dob['matched']} dates of birth matched; {dob['unmatched']} of "
                f"{dob['players']} still without one.[/green]"
            )
            phys = enrich.backfill_physical(session)
            console.print(
                f"[green]{phys['feet']} preferred feet and {phys['heights']} heights "
                f"matched; {phys['without_foot']} players still without a foot.[/green]"
            )


@app.command("talent-report")
def talent_report(
    min_minutes: int | None = typer.Option(None, "--min-minutes"),
) -> None:
    """Train the XGBoost position classifier and print its honest evaluation + SHAP."""
    # Imported lazily so the heavy ML stack only loads for this command.
    from footyvision.ml.features import load_feature_frame
    from footyvision.ml.talent import role_mismatches, shap_importance, train_position_classifier

    settings = get_settings()
    mm = settings.min_minutes if min_minutes is None else min_minutes
    with SessionLocal() as session:
        frame = load_feature_frame(session, mm)
        console.print(f"[cyan]Training position classifier on {len(frame)} players...[/cyan]")
        from footyvision.etl.spatial import attach_features

        frame = attach_features(frame)
        tm = train_position_classifier(frame)
        console.print(
            f"[green]Held-out accuracy: {tm.test_accuracy:.1%}[/green] "
            f"(train={tm.n_train}, test={tm.n_test}, classes={tm.classes})"
        )

        imp = Table(title="Top features by mean |SHAP| — which metrics define position")
        imp.add_column("feature")
        imp.add_column("mean |SHAP|", justify="right")
        for row in shap_importance(tm, frame, top_n=10):
            imp.add_row(row["feature"], f"{row['mean_abs_shap']:.3f}")
        console.print(imp)

        mism = Table(title="Role mismatches — stats resemble a different position")
        for col in ["player", "listed", "plays like", "confidence"]:
            mism.add_column(col)
        for r in role_mismatches(tm, frame, top_n=10):
            mism.add_row(r["name"], r["listed"], r["plays_like"], f"{r['confidence']:.2f}")
        console.print(mism)


@app.command("precompute")
def precompute_predictions() -> None:
    """Write the model outputs the API serves, so it need not load the models itself.

    Loading three XGBoost classifiers costs 250MB, which is most of a 512MB container and
    was enough to have the deployment restarted under load. Everything they are asked at
    request time depends only on a player's own feature row, so it is computed here.
    """
    from footyvision.ml import precompute
    from footyvision.ml.features import load_feature_frame

    settings = get_settings()
    with SessionLocal() as session:
        frame = load_feature_frame(session, settings.min_minutes)

    console.print(f"[cyan]Fitting the three classifiers over {len(frame)} rows...[/cyan]")
    payload = precompute.build(frame)
    console.print("[cyan]Fitting team attack/defence per competition...[/cyan]")
    with SessionLocal() as session:
        payload["teams"] = precompute.build_team_strengths(session)
    path = precompute.save(payload)
    size = path.stat().st_size / 1e6
    console.print(
        f"[green]{len(payload['players'])} players written to {path} ({size:.1f} MB).[/green]"
    )
    for name, meta in payload["models"].items():
        dead = [c for c, r in meta["per_class_recall"].items() if r == 0.0]
        console.print(
            f"  {name:18} {len(meta['classes']):2d} classes  acc {meta['test_accuracy']:.3f}"
            f"  balanced {meta['balanced_accuracy']:.3f}"
            + (f"  [yellow]never predicted: {', '.join(dead)}[/yellow]" if dead else "")
        )


@app.command("spatial")
def spatial_maps(workers: int = typer.Option(8, "--workers")) -> None:
    """Download every loaded match's events and publish per-player pitch maps.

    Reads the raw event JSON from the StatsBomb open-data repository, keeps only where each
    player acted and where they shot, and writes models/spatial/spatial.json for the API.
    Each match's reduction is cached under data/spatial_cache/, so a rerun resumes.
    """
    from sqlalchemy import select

    from footyvision.db.models import Match
    from footyvision.etl import spatial

    with SessionLocal() as session:
        matches = [
            (int(m), int(c), int(s))
            for m, c, s in session.execute(
                select(Match.id, Match.competition_id, Match.sb_season_id)
            )
        ]
    console.print(f"[cyan]Reducing the events of {len(matches)} matches...[/cyan]")

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            console.print(f"  {done}/{total}")

    payload = spatial.build(matches, workers=workers, progress=progress)
    path = spatial.save(payload)
    size = path.stat().st_size / 1e6
    console.print(
        f"[green]{len(payload['players'])} player-seasons from {payload['matches']} matches "
        f"written to {path} ({size:.1f} MB).[/green]"
    )
    if payload["failed"]:
        console.print(
            f"[yellow]{len(payload['failed'])} matches failed; rerun to retry them.[/yellow]"
        )

    # The expected table reads the same cached reductions, so it is built here rather than
    # by a command that would have to be remembered after this one.
    from footyvision.ml import expected_table

    with SessionLocal() as session:
        table = expected_table.build(session)
    path = expected_table.save(table)
    console.print(
        f"[green]Expected table for {len(table['seasons'])} seasons from {table['matches']} "
        f"matches written to {path}.[/green]"
    )


@app.command("value-report")
def value_report(
    min_minutes: int | None = typer.Option(None, "--min-minutes"),
) -> None:
    """Match Transfermarkt 2015/16 values to our players, train LightGBM, print metrics."""
    from footyvision.etl.transfermarkt import read_market_values_2016
    from footyvision.ml.features import PER90_FEATURES, load_feature_frame
    from footyvision.ml.value import (
        bargains,
        match_values,
        predict_values,
        shap_importance,
        train_value_model,
    )

    settings = get_settings()
    mm = settings.min_minutes if min_minutes is None else min_minutes
    console.print("[cyan]Loading Transfermarkt 2015/16 values (ES1/GB1/IT1/FR1)...[/cyan]")
    values = read_market_values_2016()
    console.print(f"  {len(values)} players with market values.")

    with SessionLocal() as session:
        features = load_feature_frame(session, mm)
    # The value source is four men's leagues, so the women's competitions cannot match
    # anything in it — every pair the fuzzy matcher found for them was a collision with a
    # man of a similar name. Dropping them here states that, rather than relying on the
    # token rule to catch it after the fact.
    if "gender" in features.columns:
        features = features[features["gender"] == "male"]
    # Age comes from our own players table now, so only the label is taken from the join.
    merged = match_values(features, values, keep_cols=("value_eur",))
    console.print(f"[green]Matched {len(merged)}/{len(features)} players to a value.[/green]")

    vm = train_value_model(merged, feature_cols=[*PER90_FEATURES, "age"])
    # Both scales, and the constant to beat. The log-scale R² is the flattering one and
    # was the only number printed here; on its own it reads as a model that explains a
    # fifth of player value, when in euros it explains almost none of it.
    console.print(
        f"[green]LightGBM value model — R²={vm.r2:.2f} on log(value), "
        f"{vm.r2_eur:.2f} in euros[/green] (train={vm.n_train}, test={vm.n_test})"
    )
    # Written so the API can answer without the Kaggle CSVs, which are gitignored and so
    # absent from every deployment. Without this the /value/* endpoints are 503 in
    # production no matter how well the model scores here.
    from footyvision.api.routers.value import ARTIFACT, save_artifact

    save_artifact(vm, predict_values(vm, merged))
    console.print(f"[cyan]Model written to {ARTIFACT}[/cyan]")

    edge = vm.baseline_mae_eur - vm.mae_eur
    console.print(
        f"[green]MAE €{vm.mae_eur:,.0f} against €{vm.baseline_mae_eur:,.0f} for predicting "
        f"the median for everyone — an edge of €{edge:,.0f}.[/green]"
    )
    if vm.interval_coverage is not None:
        # What the band actually catches, not what it was asked for. The label is the easy
        # part, and the two have never agreed here.
        console.print(
            f"[green]5th-95th percentile band contains "
            f"{vm.interval_coverage:.0%} of held-out players[/green] "
            f"(nominally 90% — the shortfall is the model being more uncertain than its "
            f"own quantiles admit)."
        )

    imp = Table(title="Top features by mean |SHAP| — what drives value")
    imp.add_column("feature")
    imp.add_column("mean |SHAP|", justify="right")
    for row in shap_importance(vm, merged, top_n=10):
        imp.add_row(row["feature"], f"{row['mean_abs_shap']:.3f}")
    console.print(imp)

    barg = Table(title="Bargains — performance implies more value than the market price")
    for col in ["player", "pos", "market value", "model value", "10th-90th", "upside"]:
        barg.add_column(col)
    priced = predict_values(vm, merged).set_index("name")
    for r in bargains(vm, merged, top_n=10):
        row = priced.loc[r["name"]]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        band = (
            f"€{row['predicted_low'] / 1e6:.1f}-{row['predicted_high'] / 1e6:.1f}M"
            if "predicted_low" in row
            else "—"
        )
        barg.add_row(
            r["name"],
            r["position_group"],
            f"€{r['actual_value']:,.0f}",
            f"€{r['predicted_value']:,.0f}",
            band,
            f"€{r['upside']:,.0f}",
        )
    console.print(barg)


@app.command("index")
def index_cmd(
    to_file: bool = typer.Option(
        False, "--to-file", help="Also write the legacy .npz alongside the database copy."
    ),
) -> None:
    """Embed all player profiles and persist the RAG index to Postgres."""
    from footyvision.llm.client import LLMClient
    from footyvision.rag.service import STORE_PATH, build_store

    client = LLMClient()
    with SessionLocal() as session:
        console.print("[cyan]Embedding player profiles...[/cyan]")
        store = build_store(session, client)
        if to_file:
            store.save(STORE_PATH)
    console.print(
        f"[green]Indexed {len(store)} profiles into Postgres "
        f"(embedded by {client.last_embed_model}).[/green]"
    )
    if to_file:
        console.print(f"[green]Also written to {STORE_PATH}.[/green]")


if __name__ == "__main__":
    app()
