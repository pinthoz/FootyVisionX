// Thin typed client for the FootyVision FastAPI backend.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

// What a 503 from an LLM endpoint means to a visitor. The API answers 503 whenever the
// language model fails — a local LM Studio that is not running, but on a deployment far
// more often a cloud model out of its free quota — so the message names neither.
export const LLM_UNAVAILABLE =
  "The language model is unavailable right now (out of quota or overloaded). Try again in a minute.";

export type Player = { id: number; name: string; country: string | null; gender?: string | null };

export type RadarMetric = { value: number; percentile: number };
export type Radar = {
  player_id: number;
  name: string;
  position_group: string;
  minutes: number;
  metrics: Record<string, RadarMetric>;
};

export type Similar = {
  player_id: number;
  name: string;
  primary_position: string | null;
  position_group?: string;
  similarity: number;
  xg_per90?: number;
  progressive_passes_per90?: number;
  tackles_per90?: number;
  dribbles_per90?: number;
};

export type Score = {
  player_id?: number;
  name: string;
  position_group: string;
  performance_score: number;
  style_profile: Record<string, number>;
  breakdown: { metric: string; weight: number; percentile: number; contribution: number }[];
  predicted_role: string | null;
  role_confidence: number | null;
  role_profile: Record<string, number>;
  /** The exact-position model's three best guesses, and deliberately not its first: its
      top pick is right 71% of the time, the three together 88%. */
  position_shortlist?: string[];
};

export type RankingRow = {
  player_id: number;
  name: string;
  competition?: string | null;
  position_group: string;
  primary_position: string | null;
  performance_score: number;
  gender?: string | null;
};

export type TopFeature = {
  feature: string;
  mean_abs_shap: number;
};

export type ModelInfoResponse = {
  task: string;
  classes: string[];
  test_accuracy: number;
  balanced_accuracy?: number | null;
  per_class_recall?: Record<string, number>;
  n_train: number;
  n_test: number;
  features?: string[];
  top_features?: TopFeature[];
  role_model: PositionModel | null;
  exact_model?: PositionModel | null;
  /** When the served predictions were computed, and whether the pool has changed since. */
  predictions_built_on?: string | null;
  predictions_stale?: boolean;
};

export type PositionModel = {
  classes: string[];
  test_accuracy: number;
  /** Mean recall over classes, each counting equally. Accuracy counts players, so the
      crowded classes decide it; these targets run to 116:1 and the two come apart. */
  balanced_accuracy?: number | null;
  /** Per class. The only field that reveals one the model never predicts at all. */
  per_class_recall?: Record<string, number>;
  n_train: number;
  n_test: number;
  /** Only reported where a single-label score understates the model: 23 exact positions. */
  top3_accuracy?: number | null;
};

export type DistributionPoint = {
  /** Unique per player-season; `player_id` is not, for the eleven who changed league. */
  id: string;
  player_id: number;
  name: string;
  competition?: string | null;
  value: number;
};

export type Distribution = {
  metric: string;
  position_group: string | null;
  count: number;
  values: DistributionPoint[];
};

export type SearchRow = {
  player_id: number;
  name: string;
  competition: string | null;
  primary_position: string | null;
  position_group?: string;
  gender?: string | null;
  nationality?: string | null;
  stats: Record<string, number>;
};

export type AssistantResult = {
  answer: string;
  sources: { player_id: number; name: string; score?: number }[];
  filters?: string | null;
  /** Players a filter removed for want of the attribute, counted per requirement. */
  not_considered?: Record<string, number> | null;
  /** The metric the sources are ordered by, when the question asked who leads in one. */
  ranked_by?: string | null;
  /** The model ran out of tokens. The text is trimmed to its last full sentence, but
      it is a partial reply and saying so is the whole point of the flag. */
  truncated?: boolean;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function post(path: string, body: unknown): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  searchPlayers: (q: string, gender?: "all" | "female" | "male") => {
    const term = q.trim();
    const params = new URLSearchParams({ with_stats: "true", limit: "200" });
    if (term) params.set("search", term);
    if (gender && gender !== "all") params.set("gender", gender);
    return get<Player[]>(`/players?${params.toString()}`);
  },
  radar: (id: number) => get<Radar>(`/players/${id}/radar`),
  similar: (id: number) =>
    get<{ results: Similar[] }>(`/players/${id}/similar?top_n=10`).then((r) => r.results),
  score: (id: number) => get<Score>(`/players/${id}/score`),
  distribution: (metric: string, positionGroup?: string) =>
    get<Distribution>(
      `/metrics/${metric}/distribution` +
        (positionGroup ? `?position_group=${positionGroup}` : "")
    ),
  rankings: (positionGroup?: string, topN = 20, gender?: "all" | "female" | "male") => {
    const params = new URLSearchParams({ top_n: String(topN) });
    if (positionGroup) params.set("position_group", positionGroup);
    if (gender && gender !== "all") params.set("gender", gender);
    return get<{ results: RankingRow[] }>(`/rankings?${params.toString()}`).then((r) => r.results);
  },
  modelInfo: () => get<ModelInfoResponse>("/talent/model-info"),
  nlSearch: (query: string) =>
    post("/search", { query }).then(async (r) => ({
      status: r.status,
      data: r.ok ? await r.json() : null,
    })),
  report: (id: number) =>
    post(`/players/${id}/report`, {}).then(async (r) => ({
      status: r.status,
      data: r.ok ? await r.json() : null,
    })),
  assistant: (question: string) =>
    post("/assistant", { question, k: 6 }).then(async (r) => ({
      status: r.status,
      data: r.ok ? ((await r.json()) as AssistantResult) : null,
    })),
  coverage: () => get<Coverage>("/coverage"),
  assistantEvaluation: () => get<RagasEvaluation>("/eval/assistant"),
  teamStrength: () => get<TeamStrength>("/teams/strength"),
  pitch: (id: number) => get<PitchMap>(`/players/${id}/pitch`),
  expectedTable: () => get<ExpectedTable>("/teams/expected-table"),
};

export type CoverageSeason = {
  competition_id: number;
  competition: string;
  country: string | null;
  season_id: number;
  season: string;
  matches: number;
  teams: number;
  players: number;
  coverage: number;
  complete: boolean;
  /** Set when the export covers one club rather than a league. */
  focus_team?: string | null;
};

export type CatalogueEntry = {
  competition_id: number;
  season_id: number;
  competition: string;
  country: string | null;
  season: string;
  matches: number;
  teams: number;
  gender: string;
  kind: string;
  complete: boolean;
  loaded: boolean;
};

export type Coverage = {
  competitions: number;
  matches: number;
  players: number;
  seasons: CoverageSeason[];
  catalogue: CatalogueEntry[];
  catalogue_verified: string;
};

export type MetricCategory = "all" | "attack" | "passing" | "defense";

export type MetricMeta = {
  key: string;
  label: string;
  category: "attack" | "passing" | "defense";
  description: string;
};

export const ALL_METRICS: MetricMeta[] = [
  // Attacking
  { key: "goals_per90", label: "Goals", category: "attack", description: "Goals scored per 90 mins" },
  { key: "xg_per90", label: "xG", category: "attack", description: "Expected goals per 90 mins" },
  { key: "shots_per90", label: "Shots", category: "attack", description: "Total shots attempted per 90 mins" },
  { key: "assists_per90", label: "Assists", category: "attack", description: "Goal assists per 90 mins" },
  { key: "dribbles_per90", label: "Dribbles", category: "attack", description: "Take-ons attempted per 90 mins" },
  { key: "dribbles_completed_per90", label: "Dribbles Done", category: "attack", description: "Successful take-ons per 90 mins" },

  // Passing & Ball Progression
  { key: "passes_per90", label: "Passes", category: "passing", description: "Passes attempted per 90 mins" },
  { key: "passes_completed_per90", label: "Passes Comp.", category: "passing", description: "Successful passes per 90 mins" },
  { key: "progressive_passes_per90", label: "Prog. Passes", category: "passing", description: "Forward progressive passes per 90 mins" },
  { key: "carries_per90", label: "Carries", category: "passing", description: "Ball carries per 90 mins" },
  { key: "progressive_carries_per90", label: "Prog. Carries", category: "passing", description: "Progressive ball carries per 90 mins" },

  // Defending & Workrate
  { key: "tackles_per90", label: "Tackles", category: "defense", description: "Tackles made per 90 mins" },
  { key: "interceptions_per90", label: "Interceptions", category: "defense", description: "Interceptions won per 90 mins" },
  { key: "blocks_per90", label: "Blocks", category: "defense", description: "Pass/shot blocks per 90 mins" },
  { key: "clearances_per90", label: "Clearances", category: "defense", description: "Clearances per 90 mins" },
  { key: "ball_recoveries_per90", label: "Recoveries", category: "defense", description: "Loose ball recoveries per 90 mins" },
  { key: "pressures_per90", label: "Pressures", category: "defense", description: "Defensive pressing events per 90 mins" },
];

export const RADAR_PRESETS: { id: string; label: string; axes: [string, string][] }[] = [
  {
    id: "curated",
    label: "Scout Core (9)",
    axes: [
      ["xg_per90", "xG"],
      ["shots_per90", "Shots"],
      ["assists_per90", "Assists"],
      ["progressive_passes_per90", "Prog Passes"],
      ["passes_per90", "Passes"],
      ["dribbles_per90", "Dribbles"],
      ["ball_recoveries_per90", "Recoveries"],
      ["tackles_per90", "Tackles"],
      ["interceptions_per90", "Interceptions"],
    ],
  },
  {
    id: "all",
    label: "All Metrics (17)",
    axes: ALL_METRICS.map((m) => [m.key, m.label]),
  },
  {
    id: "attack",
    label: "Attack & Finishing",
    axes: ALL_METRICS.filter((m) => m.category === "attack").map((m) => [m.key, m.label]),
  },
  {
    id: "passing",
    label: "Distribution & Progression",
    axes: ALL_METRICS.filter((m) => m.category === "passing").map((m) => [m.key, m.label]),
  },
  {
    id: "defense",
    label: "Defending & Pressing",
    axes: ALL_METRICS.filter((m) => m.category === "defense").map((m) => [m.key, m.label]),
  },
];

export const RADAR_AXES: [string, string][] = RADAR_PRESETS[0].axes;

/** The country as a scout would write it, not as the source stores it.

    StatsBomb spells a handful of these out in full — "Venezuela (Bolivarian Republic)",
    "Macedonia, Republic of" — and one of them carries a non-breaking space. The
    parenthetical and the appended qualifier are bureaucratic form, never the part a
    reader needs, so both are dropped; only the United States is short enough to be worth
    a special case. */
export function formatCountry(country: string | null): string | null {
  if (!country) return null;
  const trimmed = country.split("(")[0].split(",")[0].replace(/\s+/g, " ").trim();
  return trimmed === "United States of America" ? "USA" : trimmed || null;
}

export function getMetricLabel(metricKey: string): string {
  const found = ALL_METRICS.find((m) => m.key === metricKey);
  if (found) return found.label;
  return metricKey.replace(/_per90$/, "").replace(/_/g, " ");
}

export type RagasRow = {
  question: string;
  kind: string;
  faithfulness: number;
  context_precision: number;
  answer_relevancy: number;
};

export type RagasKind = {
  count: number;
  faithfulness: number | null;
  /** Null where a refusal is the right answer — RAGAS scores one as irrelevant by design. */
  answer_relevancy: number | null;
};

export type RagasEvaluation = {
  measured_on: string;
  answer_model: string;
  judge_model: string;
  questions: number;
  metrics: Record<string, number | null>;
  unanswerable: { count: number; faithfulness: number | null };
  answerable_relevancy: number | null;
  /** More than one of either means the run was finished on a second model. */
  answer_models?: string[];
  judges?: string[];
  by_kind?: Record<string, RagasKind>;
  rows: RagasRow[];
};

export type TeamRating = {
  team_id: number;
  name: string;
  competition: string | null;
  /** Log-scale Poisson coefficients: 0 is average, and a *negative* defence is good. */
  attack: number;
  defence: number;
  matches: number;
};

export type TeamStrength = {
  home_advantage: number;
  matches: number;
  teams: TeamRating[];
};

/** One shot, in StatsBomb coordinates (x 0-120 towards goal, y 0-80, own left at 0). */
export type PitchShot = {
  x: number;
  y: number;
  xg: number;
  outcome: "goal" | "saved" | "blocked" | "off-target" | "hit-woodwork" | "other";
  penalty: boolean;
  body_part: "left-foot" | "right-foot" | "head" | "other";
};

/** Where a player acted over their busiest season: a row-major grid of action counts. */
export type PitchMap = {
  player_id: number;
  competition_id: number;
  season_id: number;
  cols: number;
  rows: number;
  actions: number;
  cells: number[];
  shots: PitchShot[];
  zones: Record<string, number>;
  /** Expected Threat added by completed open-play passes and carries, over the season. */
  xt_added?: number | null;
  xt_per90?: number | null;
  /** Among the same peer group the radar ranks against. */
  xt_percentile?: number | null;
  moves_completed?: number | null;
};

export type ExpectedTableRow = {
  team_id: number;
  name: string;
  played: number;
  points: number;
  /** Points the chances deserved, from every shot's xG. */
  xpts: number;
  /** points - xpts: finishing, goalkeeping and luck. */
  luck: number;
  gf: number;
  ga: number;
  xgf: number;
  xga: number;
};

export type ExpectedTable = {
  built_on: string;
  matches: number;
  seasons: {
    competition_id: number;
    season_id: number;
    competition: string | null;
    teams: ExpectedTableRow[];
  }[];
};
