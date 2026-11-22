// Typed access to the FastAPI backend. All physics results come from the server.

export type Outcome = "survived" | "fell" | "numerical";

export interface CompactTrial {
  trial: number;
  start_substep: number;
  start_s: number;
  stance: string;
  nominal_outcome: Outcome;
  outcome: Outcome;
  reason: string | null;
  fall_time_s: number | null;
  numerical_time_s: number | null;
  time_after_start_s: number | null;
  first_nonfoot_body: string | null;
  events: Record<string, number>;
  forward_progress_m: number;
  nominal_progress_m: number;
  pairing_ok: boolean;
  has_trajectories: boolean;
}

export interface CaseAggregate {
  total_trials: number;
  survived: number;
  physical_falls: number;
  numerical_failures: number;
  valid_trials: number;
  fall_rate: number | null;
  wilson_95?: [number, number] | null;
  distinct_trajectories: number;
  time_after_start_s: { n: number; min: number | null; median: number | null; max: number | null };
}

export interface StanceRow {
  stance: string;
  tested: number;
  fell: number;
}

export interface Provenance {
  git: { commit: string; dirty: boolean | null };
  versions: Record<string, string>;
  created_utc: string;
  policy_sha256: string;
  scene: string;
}

export interface ReportSummary {
  name: string;
  config: SuiteConfig;
  config_sha256: string;
  results_sha256: string;
  sampling: { method: string; kind: string; seed: number; seed_used: boolean; interval_s: [number, number]; trials: number };
  aggregates: { nominal: CaseAggregate; attacked: CaseAggregate; paired: { mean_progress_difference_m: number | null; pairing_ok_all: boolean } };
  stance_table: StanceRow[];
  interpretation: string[];
  provenance: Provenance;
  trials: CompactTrial[];
}

export interface SuiteConfig {
  name: string;
  command: [number, number, number];
  horizon_s: number;
  trials: number;
  seed: number;
  scene: string;
  attack: { force_n: number; direction: [number, number, number]; duration_s: number; start: { method: string; low_s: number; high_s: number } };
}

export interface HiddenPhase extends ReportSummary {
  preset: { id: string; version: number; title: string; source: Record<string, string>; result_origin: string };
  verified_failure_substep: number;
  verified_failure_note: string;
  failing_substeps: number[];
  stance_finding: { stance: string; fell: number; tested: number; all_failures_in_stance: boolean } | null;
}

export interface RunListEntry {
  run_id: string;
  seed: number;
  requested_utc: string;
  status: string;
  physical_falls?: number;
  valid_trials?: number;
  numerical_failures?: number;
}

export interface LiveStress {
  preset: { id: string; version: number; title: string; source: Record<string, string> };
  config: SuiteConfig;
  trials: number;
  reference: { result_origin: string; name: string; aggregates: ReportSummary["aggregates"]; stance_table: StanceRow[]; provenance: Provenance };
  runs: RunListEntry[];
}

export interface RunStatus {
  run_id: string;
  status: "queued" | "running" | "done" | "failed" | "incomplete";
  seed: number;
  seed_source: string;
  requested_utc: string;
  reproduce: string;
  completed?: number;
  total?: number;
  error?: string | null;
  trials?: { trial: number; start_s: number; attacked: { outcome: Outcome; reason: string | null; fall_time_s: number | null } }[];
  summary?: ReportSummary;
  timing?: { wall_s: number; rollouts: number };
}

export interface ModelGeom {
  body: string;
  mesh: string;
  rgba: number[];
}

export interface ModelPayload {
  geoms: ModelGeom[];
  meshes: Record<string, { vertices: string; faces: string; n_vertices: number; n_faces: number }>;
}

export interface ReplayCase {
  frames: number;
  pos: string;
  quat: string;
  up_z: number[];
  trunk_z: number[];
  nonfoot: boolean[];
  feet: boolean[][];
  outcome: {
    outcome: Outcome;
    reason: string | null;
    fall_time_s: number | null;
    numerical_time_s: number | null;
    time_after_start_s: number | null;
    events: Record<string, number>;
    first_nonfoot_body: string | null;
    forward_progress_m: number;
  };
}

export interface Verification {
  all_match: boolean;
  recorded_commit: string;
  cases: Record<string, { digest_match: boolean; outcome_match: boolean; recorded_outcome: string; resimulated_outcome: string }>;
}

export interface PayloadBoxFrames {
  half_size_m: [number, number, number];
  mass_kg: number;
  pos: string;
  quat: string;
}

export interface GhostInfo {
  kind: "matched" | "original";
  label: string;
  parameters: { ground_friction_mu: number; payload_kg: number; payload_offset_xyz_m: number[]; push: null };
}

export interface Replay {
  fps: number;
  frames: number;
  time: number[];
  geom_count: number;
  foot_order: string[];
  cases: { nominal: ReplayCase & { payload_box?: PayloadBoxFrames | null }; attacked: ReplayCase & { payload_box?: PayloadBoxFrames | null } };
  push: { start_s: number; end_s: number; world_force_n: [number, number, number]; force_n: number } | null;
  source: "re-simulated" | "recorded";
  verification: Verification | null;
  start_s: number | null;
  run_id?: string;
  trial?: number;
  start_substep?: number;
  which?: "final" | "search";
  // M5 scenario replays
  mode?: "single" | "suite";
  ghost?: GhostInfo;
  labels?: { attacked: string; nominal: string };
  environment?: { mu: number; ghost_mu: number; nominal_mu: number };
}

// ---------------------------------------------------------------- M5 scenario designer

export interface ScenarioPush {
  force_n: number;
  direction_deg: number;
  start_s: number;
  duration_s: number;
}

export interface ScenarioDoc {
  schema_version: number;
  kind: string;
  name: string;
  command: [number, number, number];
  horizon_s: number;
  scene: string;
  environment: { ground_friction_mu: number | null };
  payload: { mass_kg: number; offset_xyz_m: [number, number, number] };
  push: ScenarioPush | null;
}

export interface FieldError {
  field: string;
  reason: string;
}

export interface Adjustment {
  field: string;
  requested: unknown;
  applied: unknown;
  reason: string;
}

export interface FrictionReport {
  requested_mu: number | null;
  nominal_mu: number;
  configured_mu: number;
  configured_floor_friction: number[];
  resolved_foot_floor: Record<string, number[]>;
  observed_at_reset: { label: string; contacts: number; sliding_mu_values: number[]; dims: number[] };
  priorities: { floor: number; feet: number[] };
  condim: { floor: number; feet: number[] };
  rule: string;
  applies_to: string;
}

export interface MassReport {
  total_mass_kg: number;
  payload: null | {
    mass_kg: number;
    trunk_mass_kg: number;
    composite_mass_kg: number;
    composite_com_trunk_frame_m: number[];
    composite_inertia_eigenvalues: number[];
    positive_definite: boolean;
  };
}

export interface Changed {
  parameter: string;
  nominal: unknown;
  value: unknown;
}

export interface ScenarioPreview {
  valid: true;
  errors: FieldError[];
  scenario: ScenarioDoc;
  adjustments: Adjustment[];
  scenario_sha256: string;
  changed_from_original: Changed[];
  effective_mu: number;
  friction: FrictionReport;
  mass: MassReport;
  model: { reused_base_model: boolean; cache: { size: number; builds: number; hits: number } };
  push: (ScenarioPush & { direction_vector: number[]; impulse_ns: number; frame: string }) | null;
  runnable: { single: boolean; suite: boolean; suite_reason: string | null };
  primary_baseline: "matched" | "original";
}

export type ValidateResponse = ScenarioPreview | { valid: false; errors: FieldError[] };

export interface LlmMeta {
  requested_model: string;
  model: string | null;
  upstream_provider: string | null;
  usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | null;
  latency_s: number;
  prompt_version: number;
}

export interface CompileResponse {
  compile_id: string;
  valid: boolean;
  errors: FieldError[];
  proposal: { name?: string; rationale?: string; not_representable?: string } & Record<string, unknown>;
  scenario_document: ScenarioDoc;
  unexpected_fields: string[];
  llm: LlmMeta;
  scenario?: ScenarioDoc;
  adjustments?: Adjustment[];
  preview?: ScenarioPreview;
}

export interface CompilerStatus {
  configured: boolean;
  provider: string;
  model: string;
  model_source: string;
  limits: { max_prompt_chars: number; timeout_s: number; max_tokens: number; rate: { per_client_per_min: number; total_per_min: number } | null };
  prompt_version: number;
}

export interface ScenarioRunEntry {
  run_id: string;
  mode: "single" | "suite";
  name: string;
  origin: string;
  requested_utc: string;
  status: string;
  outcome?: Outcome;
  reason?: string | null;
  physical_falls?: number;
  valid_trials?: number;
  numerical_failures?: number;
}

export interface ScenarioConfig {
  bounds: Record<string, [number, number]>;
  nominal_mu: number;
  presets: { id: string; file: string; note: string; scenario: ScenarioDoc; sha256: string }[];
  examples: string[];
  compiler: CompilerStatus;
  suite_defaults: { trials: number; interval_s: [number, number]; method: string; max_trials: number };
  semantics: Record<string, string>;
  runs: ScenarioRunEntry[];
}

export interface CaseSummary {
  outcome: Outcome;
  reason: string | null;
  fall_time_s: number | null;
  numerical_time_s: number | null;
  time_after_start_s: number | null;
  forward_progress_m: number;
  max_tilt_deg: number;
  mean_forward_speed_mps: number;
  slip: { mean_slip_speed_mps: number | null; total_slip_distance_m: number };
  physics: { ground_friction_mu: number; friction_source: string; payload_kg: number; payload_offset_xyz_m: number[]; push: ScenarioPush | null };
}

export interface ScenarioResult {
  mode: "single" | "suite";
  scenario: ScenarioDoc;
  adjustments: Adjustment[];
  changed_from_original: Changed[];
  trial_semantics: string;
  primary_baseline: "matched" | "original";
  baselines: Record<string, string | null>;
  cases?: { scenario: CaseSummary; matched?: CaseSummary; original: CaseSummary };
  stance_at_push?: Record<string, boolean> | null;
  results_sha256?: string;
  experiment_results_sha256?: string;
  aggregates?: ReportSummary["aggregates"];
  original?: CaseSummary;
  matched_physics?: CaseSummary["physics"];
  model: { reused_base_model: boolean; friction: FrictionReport; mass: MassReport };
  provenance: { git: { commit: string; dirty: boolean | null } };
  wall_s: number;
}

export interface ScenarioStatus {
  run_id: string;
  status: "queued" | "running" | "done" | "failed" | "incomplete";
  request: {
    mode: "single" | "suite";
    origin: { kind: string; compile_id?: string; model?: string; edited_after_compile?: boolean };
    reproduce: string;
    suite: { trials: number; seed: number; seed_source: string; interval_s: [number, number] } | null;
    scenario: ScenarioDoc;
  };
  completed?: number;
  total?: number;
  trials?: { trial: number; start_s: number; attacked: { outcome: Outcome; reason: string | null; fall_time_s: number | null } }[];
  error?: string | null;
  result?: ScenarioResult;
  summary?: ReportSummary;
}

export interface ScenarioRunRequest {
  scenario: ScenarioDoc;
  mode: "single" | "suite";
  trials?: number;
  seed?: number;
  compile_id?: string;
}

// ---------------------------------------------------------------- M4 failure discovery

export interface SearchCandidate {
  force_n: number;
  direction_deg: number;
  start_substep: number;
  duration_substeps: number;
  start_s: number;
  duration_s: number;
  impulse_ns: number;
}

export interface Evaluation {
  index: number;
  generation: number | null;
  candidate: SearchCandidate;
  outcome: Outcome;
  reason: string | null;
  fall_time_s: number | null;
  time_after_start_s: number | null;
  objective: number;
  stance_at_start: string | null;
  first_nonfoot_body: string | null;
}

export interface PairedTrialRecord {
  stance_at_start: Record<string, boolean> | null;
  attacked: { outcome: Outcome; reason: string | null; fall_time_s: number | null; time_after_start_s: number | null; first_nonfoot_body: string | null };
}

export interface ShrinkTest {
  phase: string;
  candidate: SearchCandidate;
  outcome: Outcome;
  status: string;
}

export interface DiscoveryRecord {
  config: { algorithm: string; budget: number; seed: number; shrink_budget: number; timeout_s: number };
  search: {
    stop_reason: string;
    found_failure: boolean;
    candidate_failures: number;
    numerical: number;
    evaluations_to_first_failure: number | null;
    best_candidate_index: number | null;
    best_candidate_impulse_ns: number | null;
    wall_s: number;
    library?: string;
  };
  evaluations: Evaluation[];
  confirmation: { attempts: { evaluation: number; confirmed: boolean }[]; confirmed: { evaluation: number; candidate: SearchCandidate; trial: PairedTrialRecord } | null };
  shrink: {
    improved: boolean;
    start: SearchCandidate;
    final: SearchCandidate;
    tested: ShrinkTest[];
    surviving_below_final: ShrinkTest[];
    untested: string[];
    rollouts: number;
    confirmation_rollouts: number;
    budget_exhausted: boolean;
    statement: string;
  } | null;
  final: { source: "search" | "shrunk"; candidate: SearchCandidate; trial: PairedTrialRecord } | null;
  accounting: Record<string, number>;
  provenance: { git: { commit: string; dirty: boolean | null } };
  results_sha256: string;
  statement: string;
  wall_s: number;
  request?: { reproduce: string; seed_source: string };
}

export interface DiscoveryStatus {
  run_id: string;
  status: "queued" | "running" | "done" | "failed" | "incomplete";
  stage: string;
  request: { algorithm: string; budget: number; seed: number; seed_source: string; shrink_budget: number; reproduce: string };
  evaluations?: Evaluation[];
  completed?: number;
  budget?: number;
  cancel_requested?: boolean;
  error?: string | null;
  record?: DiscoveryRecord;
}

export interface ComparisonAggregate {
  runs: number;
  found_failure: number;
  median_evaluations_to_first_failure: number | null;
  median_search_confirmed_impulse_ns: number | null;
  median_final_impulse_ns: number | null;
  min_final_impulse_ns: number | null;
  shrink_improved: number;
  total_physics_rollouts: number;
  median_wall_s: number;
}

export interface PairedTest {
  pairs: number;
  cmaes_better: number;
  random_better: number;
  ties: number;
  sign_test_p_two_sided: number | null;
}

export interface DiscoveryInfo {
  bounds: Record<string, unknown>;
  defaults: { budget: number; shrink_budget: number };
  limits: { budget: { min: number; max: number; multiple_of: number }; shrink_budget: { min: number; max: number }; timeout_s: number };
  comparison: {
    result_origin: string;
    protocol: { seeds: number[]; budget: number; shrink_budget: number };
    aggregate: { random: ComparisonAggregate; cmaes: ComparisonAggregate };
    paired: Record<string, PairedTest>;
    found_failure_fisher_p_two_sided: number;
    git: { commit: string };
  };
  runs: { run_id: string; algorithm: string; budget: number; seed: number; status: string; found_failure?: boolean; final_impulse_ns?: number | null; final_source?: string | null; stop_reason?: string }[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let text = response.statusText;
    let detail: unknown = null;
    try {
      const body = await response.json();
      detail = body.detail;
      if (typeof detail === "string") text = detail;
      else if (detail && typeof detail === "object" && "message" in detail) text = String((detail as { message: unknown }).message);
      else text = JSON.stringify(detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(`${response.status}: ${text}`, response.status, detail);
  }
  return (await response.json()) as T;
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  model: () => request<ModelPayload>("/api/model"),
  hiddenPhase: () => request<HiddenPhase>("/api/presets/hidden-phase"),
  hiddenReplay: (substep: number) => request<Replay>(`/api/presets/hidden-phase/replay/${substep}`),
  liveStress: () => request<LiveStress>("/api/presets/live-stress"),
  startRun: (seed: number | null) =>
    request<{ run_id: string; seed: number; total: number }>("/api/presets/live-stress/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(seed === null ? {} : { seed }),
    }),
  run: (runId: string) => request<RunStatus>(`/api/runs/${encodeURIComponent(runId)}`),
  runReplay: (runId: string, trial: number) =>
    request<Replay>(`/api/runs/${encodeURIComponent(runId)}/trials/${trial}/replay`),
  discoveryInfo: () => request<DiscoveryInfo>("/api/discovery"),
  startDiscovery: (body: { algorithm: string; budget: number; seed: number | null; shrink_budget: number }) =>
    request<{ run_id: string }>("/api/discovery/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body.seed === null ? { algorithm: body.algorithm, budget: body.budget, shrink_budget: body.shrink_budget } : body),
    }),
  discoveryStatus: (runId: string) => request<DiscoveryStatus>(`/api/discovery/runs/${encodeURIComponent(runId)}`),
  cancelDiscovery: (runId: string) =>
    request<{ cancel_requested: boolean }>(`/api/discovery/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }),
  discoveryReplay: (runId: string, which: "final" | "search") =>
    request<Replay>(`/api/discovery/runs/${encodeURIComponent(runId)}/replay?which=${which}`),
  scenarioConfig: () => request<ScenarioConfig>("/api/scenarios/config"),
  validateScenario: (scenario: ScenarioDoc) => request<ValidateResponse>("/api/scenarios/validate", postJson({ scenario })),
  compileScenario: (prompt: string) => request<CompileResponse>("/api/scenarios/compile", postJson({ prompt })),
  startScenario: (body: ScenarioRunRequest) => request<{ run_id: string; request: ScenarioStatus["request"] }>("/api/scenarios/runs", postJson(body)),
  scenarioStatus: (runId: string) => request<ScenarioStatus>(`/api/scenarios/runs/${encodeURIComponent(runId)}`),
  scenarioReplay: (runId: string, trial: number, ghost: "primary" | "matched" | "original") =>
    request<Replay>(`/api/scenarios/runs/${encodeURIComponent(runId)}/replay?trial=${trial}&ghost=${ghost}`),
};

export function decodeFloat32(b64: string): Float32Array {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Float32Array(bytes.buffer);
}

export function decodeInt32(b64: string): Int32Array {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Int32Array(bytes.buffer);
}

export const COLORS: Record<Outcome | "none", string> = {
  survived: "#2e9d5b",
  fell: "#d64545",
  numerical: "#7b5cd6",
  none: "#c9ced6",
};

export function fmt(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

export function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(100 * value).toFixed(1)}%`;
}
