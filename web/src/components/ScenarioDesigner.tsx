// Scenario Designer (M5): compose friction, payload and push; optionally compile a description with
// OpenRouter. Every value shown as "validated" comes back from the server's validator; the browser
// never clamps or edits values on its own, and all outcomes come from server-side MuJoCo runs.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  COLORS,
  fmt,
  pct,
  type CaseSummary,
  type CompileResponse,
  type FieldError,
  type ScenarioConfig,
  type ScenarioDoc,
  type ScenarioPreview,
  type ScenarioPush,
  type ScenarioStatus,
  type ValidateResponse,
} from "../api";
import { Advanced, Details, Evidence, EvidenceLink, ModeIntro, useJobReport } from "./Workspace";

type Ghost = "primary" | "matched" | "original";

interface Props {
  config: ScenarioConfig;
  onReplay: (runId: string, trial: number, ghost: Ghost, title: string) => void;
  onRefresh: () => void;
  replaying: string | null;
}

const DEFAULT_PUSH: ScenarioPush = { force_n: 100, direction_deg: 270, start_s: 2.0, duration_s: 0.1 };
const DIRECTIONS: { label: string; deg: number }[] = [
  { label: "from the left (pushes right)", deg: 270 },
  { label: "from the right (pushes left)", deg: 90 },
  { label: "from behind (pushes forward)", deg: 0 },
  { label: "from the front (pushes back)", deg: 180 },
];

const clone = (doc: ScenarioDoc): ScenarioDoc => JSON.parse(JSON.stringify(doc)) as ScenarioDoc;

export function ScenarioDesigner({ config, onReplay, onRefresh, replaying }: Props) {
  const initial = config.presets.find((p) => p.id === "icy-heavy-shove") ?? config.presets[0];
  const [doc, setDoc] = useState<ScenarioDoc>(() => clone(initial.scenario));
  const [presetId, setPresetId] = useState<string | null>(initial.id);
  const lastPush = useRef<ScenarioPush>(initial.scenario.push ?? DEFAULT_PUSH);
  const [preview, setPreview] = useState<{ key: string; data: ValidateResponse } | null>(null);
  const [validating, setValidating] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [compiling, setCompiling] = useState(false);
  const [compiled, setCompiled] = useState<CompileResponse | null>(null);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [compileId, setCompileId] = useState<string | null>(null);
  const [status, setStatus] = useState<ScenarioStatus | null>(null);
  const [seedText, setSeedText] = useState("");
  const [ghost, setGhost] = useState<"matched" | "original">("matched");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const autoReplay = useRef<string | null>(null); // replay the result of a run started here
  const key = JSON.stringify(doc);
  const compiler = config.compiler;

  // Server-side validation of the current form, debounced.
  useEffect(() => {
    setValidating(true);
    const handle = window.setTimeout(() => {
      api
        .validateScenario(JSON.parse(key) as ScenarioDoc)
        .then((data) => setPreview({ key, data }))
        .catch((e) => setPreview({ key, data: { valid: false, errors: [{ field: "request", reason: String(e) }] } }))
        .finally(() => setValidating(false));
    }, 350);
    return () => window.clearTimeout(handle);
  }, [key]);

  useEffect(() => {
    const runId = new URLSearchParams(window.location.search).get("run"); // deep link to a run
    if (runId && /^sc-\d{8}T\d{6}Z-[0-9a-f]{8}$/.test(runId)) poll(runId);
    return () => { if (timer.current) window.clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const update = (change: (d: ScenarioDoc) => void) => {
    setDoc((current) => {
      const next = clone(current);
      change(next);
      return next;
    });
    setPresetId(null);
  };
  const loadPreset = (id: string) => {
    const preset = config.presets.find((p) => p.id === id);
    if (!preset) return;
    setDoc(clone(preset.scenario));
    if (preset.scenario.push) lastPush.current = preset.scenario.push;
    setPresetId(id);
    setCompileId(null);
    setCompiled(null);
  };

  const compile = async (text: string) => {
    setCompiling(true);
    setCompileError(null);
    setCompiled(null);
    try {
      const result = await api.compileScenario(text);
      setCompiled(result);
      if (result.valid && result.scenario) {
        setDoc(clone(result.scenario));
        if (result.scenario.push) lastPush.current = result.scenario.push;
        setPresetId(null);
        setCompileId(result.compile_id);
      }
    } catch (e) {
      setCompileError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setCompiling(false);
    }
  };

  const poll = useCallback(
    (runId: string) => {
      const tick = async () => {
        try {
          const s = await api.scenarioStatus(runId);
          setStatus(s);
          if (s.status === "queued" || s.status === "running") timer.current = window.setTimeout(tick, 300);
          else {
            onRefresh();
            if (autoReplay.current === runId && s.status === "done" && s.result) {
              autoReplay.current = null;
              const name = s.result.scenario.name;
              if (s.result.mode === "single") onReplay(runId, 0, "primary", `Scenario ${name} vs ${s.result.primary_baseline === "matched" ? "matched baseline" : "original nominal"}`);
              else {
                const t = s.summary?.trials.find((x) => x.outcome === "fell") ?? s.summary?.trials[0];
                if (t) onReplay(runId, t.trial, "matched", `Scenario ${name} · trial ${t.trial} · push at ${t.start_s.toFixed(3)} s`);
              }
            }
          }
        } catch (e) {
          setError(String(e));
        }
      };
      void tick();
    },
    [onRefresh],
  );

  const run = async (mode: "single" | "suite") => {
    setError(null);
    const seed = seedText.trim() === "" ? undefined : Number(seedText);
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0)) {
      setError("Seed must be a non-negative integer, or blank for a fresh random seed.");
      return;
    }
    try {
      const started = await api.startScenario({
        scenario: doc,
        mode,
        ...(mode === "suite" ? { trials: config.suite_defaults.trials, ...(seed !== undefined ? { seed } : {}) } : {}),
        ...(compileId ? { compile_id: compileId } : {}),
      });
      setStatus({ run_id: started.run_id, status: "queued", request: started.request, completed: 0, total: mode === "suite" ? config.suite_defaults.trials : 1 });
      setGhost("matched");
      autoReplay.current = started.run_id;
      poll(started.run_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const current = preview && preview.key === key ? preview.data : null;
  const valid = current?.valid ? (current as ScenarioPreview) : null;
  const running = status?.status === "queued" || status?.status === "running";
  const canRun = valid !== null && !validating && !running;
  const mu = doc.environment.ground_friction_mu;
  const push = doc.push;

  const result = status?.status === "done" ? status.result : undefined;
  useJobReport(running, `Scenario run ${status?.completed ?? 0}/${status?.total ?? 1}`);
  return (
    <>
      <div className="mode">
        <ModeIntro eyebrow="Design scenario · friction × payload × shove" title="Test the robot against a disturbance">
          <p>Pick a sample or describe one. The server validates every value and rejects, never clamps, anything out of range. Controller: frozen baseline v0.</p>
        </ModeIntro>

        <div className="group">
          <div className="group-label" id="samples-label">Samples</div>
          <div className="chip-row" role="group" aria-labelledby="samples-label">
            {config.presets.map((p) => (
              <button key={p.id} className="chip-btn" aria-pressed={presetId === p.id} onClick={() => loadPreset(p.id)} title={p.note}>{p.id}</button>
            ))}
          </div>
        </div>

        <div className="summary-box" aria-live="polite">
          <div><span className="muted">Scenario</span> <strong>{doc.name}</strong> · {describeDoc(doc, config.nominal_mu)}</div>
          <ValidationLine data={current} validating={validating} />
          {compileId && <div className="xsmall muted">From OpenRouter compile {compileId}; edits are recorded as "edited after compile".</div>}
        </div>

        <div className="cta">
          <button className="primary big wide" onClick={() => void run("single")} disabled={!canRun}>
            {running ? `Running on the server… ${status?.completed ?? 0}/${status?.total ?? 1}` : "Run scenario"}
          </button>
          <div className="inline-status">
            {!running && !result && <span>One paired run: the scenario, its matched baseline and the original nominal.</span>}
            {running && <div className="progress"><div style={{ width: `${(100 * (status?.completed ?? 0)) / (status?.total ?? 1)}%` }} /></div>}
            {error && <span className="error">{error}</span>}
            {status?.status === "failed" && <span className="error">Run failed: {status.error}</span>}
            {status && result && <ResultLine status={status} ghost={ghost} onReplay={onReplay} replaying={replaying} />}
          </div>
        </div>

        <div className="composer">
          <label className="group-label" htmlFor="scenario-prompt">Or describe it in words</label>
          <textarea id="scenario-prompt" className="prompt" maxLength={compiler.limits.max_prompt_chars}
            placeholder={compiler.configured ? "e.g. sideways push from the left while walking on wet tile" : "OpenRouter is not configured on this server — use a sample or Advanced parameters"}
            value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={!compiler.configured || compiling} />
          <div className="composer-actions">
            <button onClick={() => void compile(prompt)} disabled={!compiler.configured || compiling || prompt.trim() === ""}>
              {compiling ? "Compiling…" : "Compile to parameters"}
            </button>
            <span className="xsmall muted">{compiler.configured ? `OpenRouter · ${compiler.model} proposes values; the server validates them` : "No OPENROUTER_API_KEY on the server"}</span>
          </div>
          {compiler.configured && (
            <div className="chip-row" aria-label="Example descriptions">
              {config.examples.map((ex) => (
                <button key={ex} className="chip-btn" onClick={() => setPrompt(ex)} disabled={compiling} title={ex}>{shortExample(ex)}</button>
              ))}
            </div>
          )}
          {compileError && <p className="error small">Compile failed: {compileError}. Nothing was changed.</p>}
          {compiled && <CompileResult result={compiled} />}
        </div>

        <Advanced summary="Advanced · edit parameters, run a 16-trial suite">
          <fieldset>
            <legend>Ground friction μ</legend>
            <div className="param-row">
              <label className="radio"><input type="checkbox" checked={mu === null} onChange={(e) => update((d) => { d.environment.ground_friction_mu = e.target.checked ? null : config.nominal_mu; })} /> trained floor ({config.nominal_mu})</label>
            </div>
            {mu !== null && (
              <NumberSlider value={mu} bounds={config.bounds["environment.ground_friction_mu"]} step={0.01} unit="" onChange={(v) => update((d) => { d.environment.ground_friction_mu = v; })} />
            )}
            <div className="muted small">Sets the floor geom's sliding friction; MuJoCo resolves every floor contact with the floor's value (higher priority).</div>
          </fieldset>

          <fieldset>
            <legend>Payload (welded box on the trunk)</legend>
            <NumberSlider label="mass" value={doc.payload.mass_kg} bounds={config.bounds["payload.mass_kg"]} step={0.1} unit="kg" onChange={(v) => update((d) => { d.payload.mass_kg = v; })} />
            {(["x forward", "y left", "z up"] as const).map((axis, i) => (
              <NumberSlider key={axis} label={`CoM ${axis}`} value={doc.payload.offset_xyz_m[i]} bounds={config.bounds[`payload.offset_xyz_m[${i}]`]} step={0.005} unit="m" onChange={(v) => update((d) => { d.payload.offset_xyz_m[i] = v; })} disabled={doc.payload.mass_kg === 0} />
            ))}
            <div className="muted small">Offset is relative to the trunk centre of mass, in trunk axes. At 0 kg no body is added (the original model is used).</div>
          </fieldset>

          <fieldset>
            <legend>Push (trunk force, initial-heading frame)</legend>
            <div className="param-row">
              <label className="radio">
                <input type="checkbox" checked={push !== null} onChange={(e) => update((d) => { if (e.target.checked) d.push = { ...lastPush.current }; else { if (d.push) lastPush.current = d.push; d.push = null; } })} /> apply a push
              </label>
            </div>
            {push && (
              <>
                <NumberSlider label="force" value={push.force_n} bounds={config.bounds["push.force_n"]} step={1} unit="N" onChange={(v) => update((d) => { d.push!.force_n = v; })} />
                <div className="param-row">
                  <span>direction</span>
                  <input type="number" value={push.direction_deg} step={15} onChange={(e) => update((d) => { d.push!.direction_deg = Number(e.target.value); })} /> °
                  <select value={DIRECTIONS.find((x) => x.deg === push.direction_deg)?.deg ?? ""} onChange={(e) => update((d) => { d.push!.direction_deg = Number(e.target.value); })}>
                    {!DIRECTIONS.some((x) => x.deg === push.direction_deg) && <option value="">custom</option>}
                    {DIRECTIONS.map((x) => <option key={x.deg} value={x.deg}>{x.label}</option>)}
                  </select>
                </div>
                <NumberSlider label="start" value={push.start_s} bounds={[0, doc.horizon_s]} step={0.004} unit="s" onChange={(v) => update((d) => { d.push!.start_s = v; })} />
                <NumberSlider label="duration" value={push.duration_s} bounds={config.bounds["push.duration_s"]} step={0.004} unit="s" onChange={(v) => update((d) => { d.push!.duration_s = v; })} />
                <div className="muted small">0° pushes forward, 90° pushes toward the robot's left, 270° toward its right.</div>
              </>
            )}
          </fieldset>

          <fieldset>
            <legend>Task</legend>
            <div className="param-row">
              <span>command vx</span>
              <input type="number" step={0.1} value={doc.command[0]} onChange={(e) => update((d) => { d.command[0] = Number(e.target.value); })} /> m/s
              <span>vy</span>
              <input type="number" step={0.1} value={doc.command[1]} onChange={(e) => update((d) => { d.command[1] = Number(e.target.value); })} />
              <span>yaw</span>
              <input type="number" step={0.1} value={doc.command[2]} onChange={(e) => update((d) => { d.command[2] = Number(e.target.value); })} /> rad/s
            </div>
            <div className="param-row">
              <span>horizon</span>
              <input type="number" step={0.5} value={doc.horizon_s} onChange={(e) => update((d) => { d.horizon_s = Number(e.target.value); })} /> s
              <span>name</span>
              <input type="text" value={doc.name} maxLength={64} onChange={(e) => update((d) => { d.name = e.target.value; })} />
            </div>
          </fieldset>
          <fieldset>
            <legend>16-trial suite (varies only the shove start)</legend>
            <div className="row wrap">
              <input className="seed" aria-label="Suite seed" placeholder="seed (blank = new)" value={seedText} onChange={(e) => setSeedText(e.target.value)} disabled={running || !valid?.runnable.suite} />
              <button onClick={() => void run("suite")} disabled={!canRun || !valid?.runnable.suite}>Run {config.suite_defaults.trials}-trial suite</button>
            </div>
            {valid && !valid.runnable.suite && <span className="muted small">Unavailable: {valid.runnable.suite_reason}.</span>}
          </fieldset>
        </Advanced>
        <EvidenceLink />
      </div>
      <Details>
        {status && status.status === "done" && status.result && (
          <ScenarioResultCard status={status} ghost={ghost} setGhost={setGhost} onReplay={onReplay} replaying={replaying} />
        )}
        <Evidence title="Validated preview" summary="server validation and compiled-model facts" open={!status?.result}>
          <PreviewCard data={current} validating={validating} nominalMu={config.nominal_mu} />
          <p className="muted small">Single: {config.semantics.single}. Suite: {config.semantics.suite}.</p>
        </Evidence>

        {config.runs.length > 0 && (
          <details className="card">
            <summary>Previous scenario runs ({config.runs.length})</summary>
            <table>
              <thead><tr><th>Run</th><th>Name</th><th>Mode</th><th>Origin</th><th>Result</th><th /></tr></thead>
              <tbody>
                {config.runs.map((r) => (
                  <tr key={r.run_id}>
                    <td>{r.run_id}</td><td>{r.name}</td><td>{r.mode}</td><td>{r.origin}</td>
                    <td>{r.status !== "done" ? r.status : r.mode === "single" ? `${r.outcome}${r.reason ? ` (${r.reason})` : ""}` : `${r.physical_falls}/${r.valid_trials} fell${r.numerical_failures ? ` (+${r.numerical_failures} numerical)` : ""}`}</td>
                    <td>{r.status === "done" && <button onClick={() => poll(r.run_id)} disabled={running}>Open</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}
      </Details>
    </>
  );
}

function NumberSlider({ label, value, bounds, step, unit, onChange, disabled }: {
  label?: string; value: number; bounds: [number, number]; step: number; unit: string; onChange: (v: number) => void; disabled?: boolean;
}) {
  // The number box accepts any value so the server's rejection is visible; the slider spans the bounds.
  return (
    <div className="param-row">
      {label && <span style={{ minWidth: 86 }}>{label}</span>}
      <input type="range" min={bounds[0]} max={bounds[1]} step={step} value={Math.min(bounds[1], Math.max(bounds[0], value))} onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} />
      <input type="number" step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} /> {unit}
      <span className="muted small">[{bounds[0]}, {bounds[1]}]</span>
    </div>
  );
}

function Errors({ errors }: { errors: FieldError[] }) {
  return (
    <ul className="errors">
      {errors.map((e, i) => <li key={i}><code>{e.field}</code> {e.reason}</li>)}
    </ul>
  );
}

function CompileResult({ result }: { result: CompileResponse }) {
  const usage = result.llm.usage;
  return (
    <div style={{ marginTop: 8 }}>
      <div className="small muted">
        {result.llm.model ?? result.llm.requested_model} via {result.llm.upstream_provider ?? "OpenRouter"} · {fmt(result.llm.latency_s, 2)} s
        {usage?.total_tokens ? ` · ${usage.total_tokens} tokens` : ""} · prompt v{result.llm.prompt_version} · {result.compile_id}
      </div>
      {result.proposal.rationale && <div className="quote">Model rationale (not a measurement): {result.proposal.rationale}</div>}
      {result.proposal.not_representable && <div className="quote">Not representable in this simulator: {result.proposal.not_representable}</div>}
      {result.valid ? (
        <p className="small"><span className="badge ok">valid</span> Loaded into the form below. Check the values (models can get directions and signs wrong) before running.</p>
      ) : (
        <>
          <p className="small"><span className="badge warn">rejected by the validator</span> The proposal was not loaded. Proposed values:</p>
          <code style={{ display: "block", whiteSpace: "pre-wrap" }}>{JSON.stringify(result.scenario_document, null, 1)}</code>
          <Errors errors={result.errors} />
        </>
      )}
    </div>
  );
}

function PreviewCard({ data, validating, nominalMu }: { data: ValidateResponse | null; validating: boolean; nominalMu: number }) {
  if (!data) return <div className="card"><h3>Validated preview</h3><p className="muted">Validating…</p></div>;
  if (!data.valid) {
    return (
      <div className="card">
        <div className="sweep-head"><h3>Validated preview</h3><span className="badge warn">invalid{validating ? " · rechecking…" : ""}</span></div>
        <Errors errors={data.errors} />
        <p className="muted small">Nothing can run until every field validates. Values are never clamped for you.</p>
      </div>
    );
  }
  const p = data;
  const f = p.friction;
  const payload = p.mass.payload;
  return (
    <div className="card">
      <div className="sweep-head">
        <h3>Validated preview</h3>
        <span className="badge ok">valid · {p.scenario_sha256.slice(0, 10)}{validating ? " · rechecking…" : ""}</span>
      </div>
      {p.adjustments.length > 0 && (
        <ul className="adjust">
          {p.adjustments.map((a, i) => <li key={i}><code>{a.field}</code> {JSON.stringify(a.requested)} → {JSON.stringify(a.applied)} ({a.reason})</li>)}
        </ul>
      )}
      <dl className="facts">
        <dt>Changed</dt>
        <dd>{p.changed_from_original.length === 0 ? "nothing: trained nominal conditions" : p.changed_from_original.map((c) => c.parameter).join(", ")}</dd>
        <dt>Friction</dt>
        <dd>
          requested {f.requested_mu ?? `none (trained ${nominalMu})`} · configured floor {f.configured_mu} · observed at reset: {f.observed_at_reset.contacts} floor contacts at μ{" "}
          {f.observed_at_reset.sliding_mu_values.join(", ") || "—"}
          <div className="muted small">Foot–floor resolution: {Object.entries(f.resolved_foot_floor).map(([foot, v]) => `${foot} ${v[0]}`).join(", ")} (floor priority {f.priorities.floor} &gt; feet {f.priorities.feet.join(",")}). Applies to {f.applies_to}.</div>
        </dd>
        <dt>Mass</dt>
        <dd>
          total {fmt(p.mass.total_mass_kg, 3)} kg
          {payload ? (
            <> · payload {payload.mass_kg} kg; trunk+payload {fmt(payload.composite_mass_kg, 3)} kg, CoM [{payload.composite_com_trunk_frame_m.map((x) => x.toFixed(3)).join(", ")}] m (trunk frame); inertia eigenvalues [{payload.composite_inertia_eigenvalues.map((x) => x.toFixed(4)).join(", ")}] kg·m², {payload.positive_definite ? "positive definite" : "NOT positive definite"}</>
          ) : " · no payload body"}
        </dd>
        <dt>Push</dt>
        <dd>{p.push ? `${p.push.force_n} N at ${p.push.direction_deg}° for ${p.push.duration_s} s from ${p.push.start_s} s (impulse ${fmt(p.push.impulse_ns, 2)} N·s); ${p.push.frame}` : "none"}</dd>
        <dt>Task</dt>
        <dd>command [{p.scenario.command.join(", ")}], horizon {p.scenario.horizon_s} s, scene {p.scenario.scene}</dd>
        <dt>Ghost</dt>
        <dd>{p.primary_baseline === "matched" ? "matched baseline (same friction and payload, push removed); original nominal also recorded" : "original nominal (nothing to remove, so no matched baseline)"}</dd>
        <dt>Model</dt>
        <dd>{p.model.reused_base_model ? "original compiled model reused (nominal physics)" : "scenario model compiled and cached"}</dd>
      </dl>
    </div>
  );
}

const CASE_LABEL: Record<string, string> = { scenario: "Scenario", matched: "Matched baseline", original: "Original nominal" };

function outcomeText(c: CaseSummary) {
  if (c.outcome === "fell") return `fell (${c.reason}) at ${fmt(c.fall_time_s, 3)} s`;
  if (c.outcome === "numerical") return `numerical failure (${c.reason})`;
  return "survived";
}

function CaseRow({ name, c }: { name: string; c: CaseSummary }) {
  const ph = c.physics;
  return (
    <tr>
      <td><span className="dot" style={{ background: COLORS[c.outcome] }} /> {CASE_LABEL[name] ?? name}</td>
      <td>{ph.ground_friction_mu}</td>
      <td>{ph.payload_kg > 0 ? `${ph.payload_kg} kg @ [${ph.payload_offset_xyz_m.join(", ")}]` : "—"}</td>
      <td>{ph.push ? `${ph.push.force_n} N ${ph.push.direction_deg}° ${ph.push.start_s}–${(ph.push.start_s + ph.push.duration_s).toFixed(3)} s` : "—"}</td>
      <td>{outcomeText(c)}</td>
      <td>{fmt(c.forward_progress_m, 3)}</td>
      <td>{fmt(c.mean_forward_speed_mps, 3)}</td>
      <td>{fmt(c.max_tilt_deg, 1)}</td>
      <td>{fmt(c.slip.mean_slip_speed_mps, 4)}</td>
    </tr>
  );
}

function ScenarioResultCard({ status, ghost, setGhost, onReplay, replaying }: {
  status: ScenarioStatus; ghost: "matched" | "original"; setGhost: (g: "matched" | "original") => void;
  onReplay: Props["onReplay"]; replaying: string | null;
}) {
  const r = status.result!;
  const origin = status.request.origin;
  const head = (
    <div className="sweep-head">
      <strong>Executed now · scenario run {status.run_id} · {r.scenario.name}</strong>
      <span className="badge live">new simulation · commit {r.provenance.git.commit.slice(0, 7)}{r.provenance.git.dirty ? " (dirty tree)" : ""} · {fmt(r.wall_s, 2)} s</span>
    </div>
  );
  const originLine = (
    <p className="muted small">
      Origin: {origin.kind === "openrouter" ? `OpenRouter compile ${origin.compile_id} (${origin.model})${origin.edited_after_compile ? ", edited after compile" : ", unedited"}` : "manual / preset"}.
      Reproduce: <code>{status.request.reproduce}</code>
    </p>
  );
  if (r.mode === "single" && r.cases) {
    const names = (["scenario", "matched", "original"] as const).filter((n) => r.cases![n]);
    return (
      <div className="card">
        {head}
        <table className="cases">
          <thead><tr><th>Case</th><th>μ</th><th>Payload</th><th>Push</th><th>Outcome</th><th>Progress m</th><th>Speed m/s</th><th>Max tilt °</th><th>Slip m/s</th></tr></thead>
          <tbody>{names.map((n) => <CaseRow key={n} name={n} c={r.cases![n]!} />)}</tbody>
        </table>
        <p className="small">{r.trial_semantics}. Slip = mean horizontal foot speed while that foot is in floor contact, before any fall.</p>
        <div className="row wrap">
          {r.cases.matched && (
            <button onClick={() => onReplay(status.run_id, 0, "matched", `Scenario ${r.scenario.name} vs matched baseline`)} disabled={replaying !== null}>
              {replaying === `scenario/${status.run_id}/0/matched` ? "Loading…" : "Replay vs matched baseline"}
            </button>
          )}
          <button onClick={() => onReplay(status.run_id, 0, "original", `Scenario ${r.scenario.name} vs original nominal`)} disabled={replaying !== null}>
            {replaying === `scenario/${status.run_id}/0/original` ? "Loading…" : "Replay vs original nominal"}
          </button>
        </div>
        {originLine}
      </div>
    );
  }
  const summary = status.summary;
  const att = r.aggregates?.attacked;
  const nom = r.aggregates?.nominal;
  return (
    <div className="card">
      {head}
      <p className="small">{r.trial_semantics}</p>
      <div className="chips">
        {summary?.trials.map((t) => (
          <button key={t.trial} className="chip" style={{ background: COLORS[t.outcome] }} disabled={replaying !== null}
            title={`trial ${t.trial}: ${t.outcome}${t.fall_time_s !== null ? ` @ ${fmt(t.fall_time_s)} s` : ""}`}
            onClick={() => onReplay(status.run_id, t.trial, ghost, `Scenario ${r.scenario.name} · trial ${t.trial} · push at ${t.start_s.toFixed(3)} s`)}>
            <span>{t.start_s.toFixed(3)} s</span>
            <small>{replaying === `scenario/${status.run_id}/${t.trial}/${ghost}` ? "loading" : t.outcome === "fell" ? `fell ${fmt(t.fall_time_s, 2)}` : t.outcome}</small>
          </button>
        ))}
      </div>
      {att && nom && (
        <p className="result-line">
          <strong>Pushed: {att.physical_falls} falls / {att.valid_trials} valid trials</strong> ({pct(att.fall_rate)})
          {att.wilson_95 && <> · Wilson 95% [{fmt(att.wilson_95[0])}, {fmt(att.wilson_95[1])}]</>}; numerical {att.numerical_failures}.
          {" "}Matched baseline (same friction and payload, no push): {nom.physical_falls} falls, {nom.distinct_trajectories} distinct trajectory.
          {r.original && <> Original nominal reference: {outcomeText(r.original)}, progress {fmt(r.original.forward_progress_m, 2)} m.</>}
        </p>
      )}
      <div className="row">
        <span className="small">Ghost in replays:</span>
        <label className="radio small"><input type="radio" checked={ghost === "matched"} onChange={() => setGhost("matched")} /> matched baseline</label>
        <label className="radio small"><input type="radio" checked={ghost === "original"} onChange={() => setGhost("original")} /> original nominal</label>
      </div>
      <p className="muted small">An estimate over the stated push-timing distribution only; click a trial to replay it.</p>
      {originLine}
    </div>
  );
}

function ValidationLine({ data, validating }: { data: ValidateResponse | null; validating: boolean }) {
  if (!data) return <p className="small muted">Validating on the server…</p>;
  if (!data.valid) {
    return (
      <div className="small">
        <span className="badge warn">invalid{validating ? " · rechecking…" : ""}</span> Nothing can run until every field validates.
        <Errors errors={data.errors} />
      </div>
    );
  }
  const f = data.friction;
  return (
    <div className="small">
      <span className="badge ok">valid · {data.scenario_sha256.slice(0, 8)}{validating ? " · rechecking…" : ""}</span>{" "}
      {data.changed_from_original.length ? `changed: ${data.changed_from_original.map((c) => c.parameter).join(", ")}` : "trained nominal conditions"} · floor μ
      observed at reset {f.observed_at_reset.sliding_mu_values.join(", ") || "—"} · total mass {fmt(data.mass.total_mass_kg, 2)} kg
      {data.adjustments.length > 0 && <> · {data.adjustments.length} adjustment(s): {data.adjustments.map((a) => `${a.field} → ${JSON.stringify(a.applied)}`).join(", ")}</>}
      <span className="muted"> · full preview below</span>
    </div>
  );
}

function ResultLine({ status, ghost, onReplay, replaying }: { status: ScenarioStatus; ghost: "matched" | "original"; onReplay: Props["onReplay"]; replaying: string | null }) {
  const r = status.result!;
  if (r.mode === "single" && r.cases) {
    const c = r.cases;
    return (
      <div className="small result-box">
        <div><span className="dot" style={{ background: COLORS[c.scenario.outcome] }} /> <strong>Scenario:</strong> {outcomeText(c.scenario)}, progress {fmt(c.scenario.forward_progress_m, 2)} m</div>
        {c.matched && <div><span className="dot" style={{ background: COLORS[c.matched.outcome] }} /> Matched baseline: {outcomeText(c.matched)}, {fmt(c.matched.forward_progress_m, 2)} m</div>}
        <div><span className="dot" style={{ background: COLORS[c.original.outcome] }} /> Original nominal: {outcomeText(c.original)}, {fmt(c.original.forward_progress_m, 2)} m</div>
        <div className="row wrap">
          {c.matched && <button onClick={() => onReplay(status.run_id, 0, "matched", `Scenario ${r.scenario.name} vs matched baseline`)} disabled={replaying !== null}>Replay vs matched</button>}
          <button onClick={() => onReplay(status.run_id, 0, "original", `Scenario ${r.scenario.name} vs original nominal`)} disabled={replaying !== null}>Replay vs original</button>
        </div>
      </div>
    );
  }
  const att = r.aggregates?.attacked;
  return (
    <div className="small result-box">
      {att && <div><strong>Pushed: {att.physical_falls} falls / {att.valid_trials} valid trials</strong> ({pct(att.fall_rate)}); matched baseline {r.aggregates?.nominal.physical_falls} falls. Trials and replays below (ghost: {ghost}).</div>}
    </div>
  );
}

function shortExample(text: string): string {
  const words = text.split(/\s+/);
  return words.length <= 4 ? text : `${words.slice(0, 4).join(" ")}…`;
}

function describeDoc(doc: ScenarioDoc, nominalMu: number): string {
  const parts = [doc.environment.ground_friction_mu === null ? `trained floor μ ${nominalMu}` : `floor μ ${doc.environment.ground_friction_mu}`];
  parts.push(doc.payload.mass_kg > 0 ? `${doc.payload.mass_kg} kg payload` : "no payload");
  if (doc.push) {
    const dir = DIRECTIONS.find((d) => d.deg === doc.push!.direction_deg)?.label ?? `${doc.push.direction_deg}°`;
    parts.push(`${doc.push.force_n} N ${dir} at ${doc.push.start_s} s for ${doc.push.duration_s} s`);
  } else parts.push("no shove");
  parts.push(`${doc.horizon_s} s at vx ${doc.command[0]}`);
  return parts.join(" · ");
}
