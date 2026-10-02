"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    country: str | None = None
    gender: str | None = None


class SeasonStatsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    competition_id: int
    sb_season_id: int
    primary_position: str | None
    matches_played: int
    minutes: float

    goals_per90: float
    assists_per90: float
    xg_per90: float
    progressive_passes_per90: float
    tackles_per90: float
    interceptions_per90: float


class TargetOut(BaseModel):
    player_id: int
    name: str
    primary_position: str | None
    position_group: str
    minutes: float


class SimilarPlayerOut(BaseModel):
    player_id: int
    name: str
    primary_position: str | None
    position_group: str
    competition_id: int
    sb_season_id: int
    minutes: float
    similarity: float
    # a little context so results are readable without a second call
    xg_per90: float
    progressive_passes_per90: float
    tackles_per90: float
    dribbles_per90: float


class SimilarResponse(BaseModel):
    target: TargetOut
    count: int
    results: list[SimilarPlayerOut]


class RadarMetric(BaseModel):
    value: float
    percentile: float


class RadarResponse(BaseModel):
    player_id: int
    name: str
    position_group: str
    minutes: float
    metrics: dict[str, RadarMetric]


class ReportResponse(BaseModel):
    player_id: int
    name: str
    report: str
    context: dict


class ReportContextResponse(BaseModel):
    player_id: int
    context: dict


# Bounds on what a caller may send to an endpoint that spends LLM tokens. Unbounded, one
# request could carry a megabyte of text, or ask the assistant for every player in the
# pool — 2,562 profiles, several hundred thousand tokens, in a single call that the rate
# limiter counts as one. The longest question in the 241-question evaluation bank is 66
# characters and the dashboard asks for six players, so neither limit is close to use.
MAX_QUESTION_CHARS = 500
MAX_ASSISTANT_K = 12


class NLSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class SearchResultRow(BaseModel):
    player_id: int
    name: str
    competition: str | None
    primary_position: str | None
    position_group: str
    gender: str | None = None
    nationality: str | None = None
    stats: dict[str, float]


class SearchResponse(BaseModel):
    interpreted: dict
    count: int
    results: list[SearchResultRow]


class ScoreResponse(BaseModel):
    player_id: int
    name: str
    position_group: str
    performance_score: float
    breakdown: list[dict]
    style_profile: dict[str, float]
    # The finer read: which of the ten side-agnostic roles the numbers look like.
    predicted_role: str | None = None
    role_confidence: float | None = None
    role_profile: dict[str, float] = {}
    # The exact-position model's three best guesses, and deliberately not its first.
    #
    # Chosen when its top-1 was right 47% of the time, 42% of its errors being pure
    # left/right swaps, because nothing it saw had a side. The pitch-map features took that
    # to 71% and the swaps to a quarter of the errors, which makes a single label far more
    # defensible than it was — but still wrong three times in ten, where the shortlist of
    # three contains the answer 88% of the time. It stays a shortlist.
    position_shortlist: list[str] = []


class RankingRow(BaseModel):
    player_id: int
    name: str
    competition: str | None
    position_group: str
    primary_position: str | None
    performance_score: float
    gender: str | None = None


class RankingsResponse(BaseModel):
    count: int
    results: list[RankingRow]


class FeatureImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class RoleModelInfo(BaseModel):
    """A finer-grained classifier sitting beside the four-group one."""

    classes: list[str]
    test_accuracy: float
    n_train: int
    n_test: int
    # Only meaningful where there are enough classes for a single-label score to
    # understate the model — reported for the 21-class exact position, not for four groups.
    top3_accuracy: float | None = None
    # Mean recall over classes, each counting equally. `test_accuracy` counts players, so
    # the crowded classes decide it; these targets are lopsided enough that the two numbers
    # tell different stories, and the smaller one is the one a scout needs.
    balanced_accuracy: float | None = None
    # Recall per class. The only field that reveals a class the model has stopped
    # predicting altogether, which no aggregate score will show.
    per_class_recall: dict[str, float] = {}


class ModelInfoResponse(BaseModel):
    task: str
    classes: list[str]
    test_accuracy: float
    n_train: int
    n_test: int
    balanced_accuracy: float | None = None
    per_class_recall: dict[str, float] = {}
    # Everything the classifier is allowed to see, and which of it actually decides.
    features: list[str] = []
    top_features: list[FeatureImportance] = []
    role_model: RoleModelInfo | None = None
    exact_model: RoleModelInfo | None = None
    # When the served predictions were computed, and whether they describe a different
    # pool from the database in front of them. A season imported without re-running
    # `footyvision precompute` leaves these answering for a pool that has since changed.
    predictions_built_on: str | None = None
    predictions_stale: bool = False


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(6, ge=1, le=MAX_ASSISTANT_K)


class AssistantSource(BaseModel):
    player_id: int
    name: str
    score: float


class AssistantResponse(BaseModel):
    answer: str
    sources: list[AssistantSource]
    # The hard requirements read out of the question and applied before ranking, so the
    # caller can see why a pool was narrowed — null when the question stated none.
    filters: str | None = None
    # Players a filter removed for want of the attribute rather than for failing it,
    # counted per requirement. Date of birth and preferred foot come from a men's football
    # source, so an age or foot question drops the women's competitions without ever
    # comparing them, and a caller that cannot see that reads a partial pool as the whole.
    not_considered: dict[str, int] | None = None
    # Set when the question asked who leads in a metric: the sources are then the top of
    # the filtered pool by that metric, in order, rather than the closest matches in style.
    ranked_by: str | None = None
    # Set when the model stopped because it ran out of tokens rather than because it had
    # finished. The text has been trimmed back to its last complete sentence so it does not
    # end mid-word, but it is still a partial answer, and presenting one as whole is the
    # kind of quiet wrongness this API tries not to ship.
    truncated: bool = False


class DistributionPoint(BaseModel):
    # Unique per row, which the player id is not: eleven players changed league mid-season
    # and hold one player-season in each, so a chart keyed on player_id drops one of them.
    id: str
    player_id: int
    name: str
    competition: str | None = None
    value: float


class DistributionResponse(BaseModel):
    """One metric's value for every player in the pool, for plotting a distribution."""

    metric: str
    position_group: str | None
    count: int
    values: list[DistributionPoint]


class CoverageSeason(BaseModel):
    competition_id: int
    competition: str
    country: str | None
    season_id: int
    season: str
    matches: int
    teams: int
    players: int
    coverage: float
    complete: bool
    # Set when the export is one club's season rather than a league's. The Bundesliga
    # 2015/16 here is Bayer Leverkusen's 34 matches, which against a league baseline reads
    # as 11% of a season and is in fact all of one.
    focus_team: str | None = None


class CatalogueEntry(BaseModel):
    competition_id: int
    season_id: int
    competition: str
    country: str | None
    season: str
    matches: int
    teams: int
    gender: str
    kind: str
    complete: bool
    loaded: bool


class CoverageResponse(BaseModel):
    competitions: int
    matches: int
    players: int
    seasons: list[CoverageSeason]
    catalogue: list[CatalogueEntry]
    catalogue_verified: str


class RagasRow(BaseModel):
    question: str
    kind: str
    faithfulness: float
    context_precision: float
    answer_relevancy: float


class RagasUnanswerable(BaseModel):
    count: int
    faithfulness: float | None = None


class RagasKind(BaseModel):
    count: int
    faithfulness: float | None = None
    # Null where a refusal is the right answer: RAGAS scores one as irrelevant by design,
    # so an average there would grade the assistant down for behaving correctly.
    answer_relevancy: float | None = None


class RagasEvaluation(BaseModel):
    """A dated RAGAS run over the scouting assistant, as published by scripts/eval_ragas.py.

    `answerable_relevancy` is reported separately from the headline mean because RAGAS
    scores a refusal as irrelevant by design: the questions written to have no answer in
    the data drag the average down for behaving correctly, and averaging them in would
    reward an assistant that made something up instead.
    """

    measured_on: str
    answer_model: str
    judge_model: str
    questions: int
    metrics: dict[str, float | None]
    unanswerable: RagasUnanswerable
    answerable_relevancy: float | None = None
    # Every model that answered and every model that judged, read off the rows. More than
    # one of either means the run was finished on a different model after a quota ran out,
    # which makes comparisons between categories unreliable.
    answer_models: list[str] = []
    judges: list[str] = []
    by_kind: dict[str, RagasKind] = {}
    rows: list[RagasRow] = []


class TeamStrengthOut(BaseModel):
    team_id: int
    name: str
    competition: str | None = None
    # Log-scale Poisson coefficients: 0 is an average side, a positive attack scores more
    # than average, and a negative defence concedes less. The defence sign catches people
    # out, so it is stated wherever these are rendered.
    attack: float
    defence: float
    matches: int


class ExpectedTableRow(BaseModel):
    team_id: int
    name: str
    played: int
    points: int
    # Points the chances deserved: each match's win/draw/loss probabilities from both sides'
    # shot xG, treated as independent chances. `luck` is points minus xPts.
    xpts: float
    luck: float
    gf: int
    ga: int
    xgf: float
    xga: float


class ExpectedTableSeason(BaseModel):
    competition_id: int
    season_id: int
    competition: str | None = None
    teams: list[ExpectedTableRow] = []


class ExpectedTable(BaseModel):
    """The league tables the chances deserved, next to the real ones.

    Measured in scripts/eval_expected_table.py: over 143 team-seasons, expected points per
    match in the first half of a season predict points per match in the second half better
    than the points actually won (r 0.72 against 0.67) — finishing and luck do not persist,
    chances do.
    """

    built_on: str
    matches: int
    seasons: list[ExpectedTableSeason] = []


class TeamStrengthResponse(BaseModel):
    # Also on the log scale: exp() of it is the factor applied to a side's goal rate at
    # home, which comes out around 1.26 on this data.
    home_advantage: float
    matches: int
    teams: list[TeamStrengthOut] = []


class ValueModelInfo(BaseModel):
    """How well the value model does, on both scales and against doing nothing.

    `r2_log` is computed on log(value), where the model is fitted; `r2_eur` is the same
    model judged in euros, and it is the smaller number by some distance. Both are given
    because the log figure on its own reads as far more precision than this model has.

    `mae_eur` means nothing without `baseline_mae_eur` beside it — the error from predicting
    the training median for every player. The gap between them is the model's entire
    contribution.
    """

    features: list[str]
    r2_log: float
    r2_eur: float
    mae_eur: float
    baseline_mae_eur: float
    n_train: int
    n_test: int
    # What share of held-out players actually fell inside the band, against the 90% it was
    # fitted for. These have never agreed here, so the measured figure travels with every
    # prediction rather than living in a footnote.
    interval_coverage: float | None = None
    nominal_coverage: float = 0.9


class PlayerValue(BaseModel):
    """A price range for one player, never a price.

    The band is the answer. `predicted_eur` is the median model's point and is included
    only because a range with no centre is awkward to render — it is not a valuation, and
    `interval_coverage` on the model says how often the band around it was even right.
    """

    player_id: int
    name: str
    position_group: str
    predicted_eur: float
    predicted_low_eur: float
    predicted_high_eur: float
    # The Transfermarkt 2015/16 figure, where this player could be matched to one. Absent
    # for most: it is the training label, not a lookup table, and only 1,019 of the men in
    # the database have one.
    market_value_eur: float | None = None
    # market value minus prediction. Negative means the model prices him above the market.
    residual_eur: float | None = None
    model: ValueModelInfo


class ValueBargain(BaseModel):
    player_id: int
    name: str
    position_group: str
    market_value_eur: float
    predicted_eur: float
    predicted_low_eur: float
    predicted_high_eur: float
    upside_eur: float
    # True when the market value sits below the bottom of the predicted band, so the
    # gap is larger than the model's own uncertainty about the player.
    clears_band: bool


class ValueBargains(BaseModel):
    """Players the model prices above what the market paid.

    `clears_band` is the only column that separates a signal from arithmetic: it is true
    when the market value falls below the *bottom* of the predicted range, so the gap
    survives the model's own uncertainty. Rows where it is false are inside the noise.
    """

    results: list[ValueBargain] = []
    model: ValueModelInfo


class PitchShot(BaseModel):
    x: float
    y: float
    xg: float
    # goal / saved / blocked / off-target / hit-woodwork / other
    outcome: str
    # Kept apart because one spot kick is worth ~0.76 xG and would otherwise dominate the
    # shot map of any player who took a few.
    penalty: bool
    body_part: str


class PitchMap(BaseModel):
    """Where one player acted over one season, in StatsBomb coordinates.

    `cells` is a row-major `cols` x `rows` grid of action counts over a 120 x 80 pitch,
    x towards the opponent's goal and y = 0 on the player's own attacking left. `zones`
    are the summaries the position classifiers read — how high, how wide, which side.
    """

    player_id: int
    competition_id: int
    season_id: int
    cols: int
    rows: int
    actions: int
    cells: list[int]
    shots: list[PitchShot] = []
    zones: dict[str, float] = {}
    # Expected Threat added by completed open-play passes and carries: how much, summed over
    # the season, they raised the chance that possession ends in a goal. Validated against
    # goals in scripts/eval_expected_threat.py — it tracks a team's goals in a match better
    # than its completed-move count (r 0.37 against 0.27) and predicts its later goals as
    # well as xG does. Per 90 and as a percentile among the same peer group as the radar,
    # because conversion differs between leagues and a pooled ranking would compare them.
    xt_added: float | None = None
    xt_per90: float | None = None
    xt_percentile: float | None = None
    moves_completed: int | None = None
