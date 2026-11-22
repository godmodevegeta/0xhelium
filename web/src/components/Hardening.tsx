// Policy Hardening (M6): search a bounded residual reflex around the frozen policy on the server's
// CPU, then compare baseline and candidate on identical episodes. This mode never promotes: the
// status pill shows the verdict of the pre-registered M7 gate (gate-v1) for the candidate shown.
import { useCallback, useEffect, useRef, useState } from "react";
import type { ShowReplay } from "../App";
import {
  COLORS,
  fmt,
  gateStatusFor,
  hardeningApi,
  type CandidateRecord,
  type GateInfo,
  type GateStatus,
  type ComparisonRow,
  type HardeningInfo,
  type HardeningStatus,
  type HistoryEntry,
  type SetSummary,
} from "../api";
import { PALETTE } from "../theme";
import { Advanced, Details, Evidence, EvidenceLink, ModeIntro, useJobReport } from "./Workspace";

interface Props {
  info: HardeningInfo;
  gate: GateInfo | null;
  show: ShowReplay;
  onRefresh: () => void;
  replaying: string | null;
}

type Ghost = "baseline" | "matched";
interface Current {
  key: string;
  record: CandidateRecord;
  origin: "precomputed" | "executed";
}

const CHANGE_TEXT: Record<string, string> = {
  recovered: "baseline fell → candidate survived",
  regressed: "baseline survived → candidate FELL",
  both_fell: "both fell",
  both_survived: "both survived",
  numerical: "numerical failure",
};

function GatePill({ status }: { status: GateStatus | null }) {
  if (!status) return <span className="status-pill">Candidate · not evaluated by the regression gate</span>;
  if (status.status === "promoted") return <span className="status-pill ok">Promoted by {status.gate} as {status.version}</span>;
  if (status.status === "rejected") return <span className="status-pill">Rejected by {status.gate} · frozen baseline stays active</span>;
  return <span className="status-pill">Evaluated by {status.gate} · not eligible for promotion</span>;
}

function surv(s: SetSummary | null | undefined) {
  return s ? `${s.attack_survived}/${s.attacks}` : "—";
}

export function Hardening({ info, gate, show, onRefresh, replaying }: Props) {
  const [current, setCurrent] = useState<Current | null>(
    info.preset ? { key: info.preset.key, record: info.preset.record, origin: "precomputed" } : null,
  );
  const [focus, setFocus] = useState<string>("");
  const [generations, setGenerations] = useState(info.defaults.max_generations);
  const [seedText, setSeedText] = useState("");
  const [status, setStatus] = useState<HardeningStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [episode, setEpisode] = useState<string>("m2-100N-500");
  const [ghost, setGhost] = useState<Ghost>("baseline");
  const timer = useRef<number | null>(null);
  const autoReplay = useRef<string | null>(null);
  const selectable = info.episodes.filter((e) => e.selectable);
  const replayable = info.episodes;

  const replay = useCallback(
    (key: string, record: CandidateRecord, ep: string, g: Ghost) =>
      show(`hardening/${key}/${ep}/${g}`, `Hardening · ${ep} · candidate ${record.candidate_id.slice(5, 13)} vs ${g === "baseline" ? "frozen baseline" : "no-shove baseline"}`,
        () => hardeningApi.replay(key, ep, g)),
    [show],
  );

  const poll = useCallback(
    (runId: string) => {
      const tick = async () => {
        try {
          const s = await hardeningApi.status(runId);
          setStatus(s);
          if (s.status === "queued" || s.status === "running") {
            timer.current = window.setTimeout(tick, 400);
            return;
          }
          onRefresh();
          if (s.status === "done" && s.record) {
            setCurrent({ key: runId, record: s.record, origin: "executed" });
            if (autoReplay.current === runId) {
              autoReplay.current = null;
              const ep = s.record.optimizer.config.focus ?? s.record.sets.training_ids[0];
              setEpisode(ep);
              void replay(runId, s.record, ep, "baseline");
            }
          }
        } catch (e) {
          setError(String(e));
        }
      };
      void tick();
    },
    [onRefresh, replay],
  );

  useEffect(() => {
    // Deep links: ?tab=hardening&hrun=<id> opens a run; &hreplay=<key>:<episode>:<baseline|matched>.
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("hrun");
    if (runId && /^hd-\d{8}T\d{6}Z-\d{1,20}$/.test(runId)) poll(runId);
    const hr = params.get("hreplay")?.match(/^((?:hd-\d{8}T\d{6}Z-\d{1,20})|preset-v1):([A-Za-z0-9._-]{1,64}):(baseline|matched)$/);
    if (hr && hr[1] === "preset-v1" && info.preset) {
      setEpisode(hr[2]);
      setGhost(hr[3] as Ghost);
      void replay(hr[1], info.preset.record, hr[2], hr[3] as Ghost);
    }
    return () => { if (timer.current) window.clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async () => {
    setError(null);
    const seed = seedText.trim() === "" ? undefined : Number(seedText);
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0)) return setError("Seed must be a non-negative integer or blank.");
    const [lo, hi] = info.limits.max_generations;
    if (!Number.isInteger(generations) || generations < lo || generations > hi) return setError(`Generations must be ${lo}–${hi}.`);
    try {
      const started = await hardeningApi.start({ generations, ...(seed !== undefined ? { seed } : {}), ...(focus ? { focus } : {}) });
      autoReplay.current = started.run_id;
      setStatus({ run_id: started.run_id, status: "queued", stage: "queued", request: started.request, history: [] });
      poll(started.run_id);
    } catch (e) {
      setError(String(e));
    }
  };
  const cancel = async () => {
    if (!status) return;
    try {
      await hardeningApi.cancel(status.run_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const running = status?.status === "queued" || status?.status === "running";
  const history = running ? status?.history ?? [] : current?.record.history ?? [];
  const last = history[history.length - 1];
  const runGenerations = status?.request.config.max_generations ?? generations;
  const rec = current?.record;
  const tr = rec?.measurements.train;
  const va = rec?.measurements.validation_preview;
  const rows = [...(tr?.comparison ?? []), ...(va?.comparison ?? [])];
  const row = rows.find((r) => r.id === episode);
  const epInfo = replayable.find((e) => e.id === episode);
  const replayKey = current ? `hardening/${current.key}/${episode}/${ghost}` : "";

  useJobReport(running, `CPU hardening${last ? ` · generation ${last.generation}/${runGenerations}` : ` · ${status?.stage ?? "queued"}`}`);
  const totalNominalFell = (m?: typeof tr) => (m?.baseline_summary.nominal_fell ?? 0);
  const gateStatus = gateStatusFor(gate, rec?.candidate_id) ?? (current?.origin === "precomputed" ? info.preset?.gate_status ?? null : null);
  return (
    <>
      <div className="mode">
        <ModeIntro eyebrow="Harden controller · CPU reflex search" title="Tune a bounded reflex around the frozen policy">
          <p>The pretrained policy stays unchanged. Seeded CMA-ES tunes a 10-parameter reflex on the server CPU against recorded shove failures and nominal walking. Not PPO, not fine-tuning.</p>
        </ModeIntro>
        <div><GatePill status={gateStatus} /></div>
        <div className="group">
          <label className="group-label" htmlFor="harden-focus">Failures to harden against</label>
          <select id="harden-focus" value={focus} onChange={(e) => setFocus(e.target.value)} disabled={running}>
            <option value="">All {selectable.length} recorded training failures (+ variants, nominal walks)</option>
            {selectable.map((e) => (
              <option key={e.id} value={e.id}>{e.id} · {e.push ? `${e.push.force_n} N at ${e.push.direction_deg.toFixed(0)}°, ${e.push.start_s.toFixed(3)} s` : ""}</option>
            ))}
          </select>
        </div>
        <div className="cta">
          <button className="primary big wide" onClick={start} disabled={running}>
            {running ? `Hardening on the server… ${status?.stage ?? "queued"}` : "Harden controller"}
          </button>
          <div className="inline-status">
            {running && (
              <>
                <div className="progress"><div style={{ width: `${(100 * (last?.generation ?? 0)) / runGenerations}%` }} /></div>
                <div className="row wrap">
                  <span>
                    CPU optimization: <strong>{status?.stage}</strong>
                    {last && <> · generation {last.generation}/{runGenerations} · {last.rollouts} rollouts · best J {fmt(last.best_objective, 3)} vs baseline {fmt(last.baseline_objective, 3)} · {last.best_attack_survived}/{last.attacks} attacks survived · {fmt(last.elapsed_s, 1)} s</>}
                  </span>
                  <button onClick={cancel} disabled={status?.cancel_requested}>{status?.cancel_requested ? "Cancelling…" : "Cancel run"}</button>
                </div>
              </>
            )}
            {error && <span className="error">{error}</span>}
            {status?.status === "failed" && <span className="error">Hardening failed: {status.error}</span>}
            {!running && !error && <span>Population {info.defaults.popsize} on {info.defaults.workers} CPU workers, {info.limits.wall_clock_s} s cap. A run here produces a candidate; it is never promoted from this page.</span>}
          </div>
        </div>

        {rec && current && (
          <div className="summary-box">
            <div className="row wrap">
              <span className={`badge ${current.origin === "precomputed" ? "pre" : "live"}`}>{current.origin === "precomputed" ? "Precomputed sample candidate" : `Executed now · ${current.key}`}</span>
              <span className="xsmall muted">{rec.candidate_id}</span>
            </div>
            <table className="compare">
              <thead><tr><th /><th>Frozen baseline</th><th>Candidate</th></tr></thead>
              <tbody>
                <tr><td>training attacks survived</td><td>{surv(tr?.baseline_summary)}</td><td><strong>{surv(tr?.candidate_summary)}</strong></td></tr>
                <tr><td>validation-preview attacks</td><td>{surv(va?.baseline_summary)}</td><td><strong>{surv(va?.candidate_summary)}</strong></td></tr>
                <tr><td>nominal falls</td><td>{totalNominalFell(tr) + totalNominalFell(va)}</td>
                  <td className={(tr?.candidate_summary?.nominal_fell ?? 0) + (va?.candidate_summary?.nominal_fell ?? 0) ? "bad" : ""}>{(tr?.candidate_summary?.nominal_fell ?? 0) + (va?.candidate_summary?.nominal_fell ?? 0)}</td></tr>
              </tbody>
            </table>
            {!rec.result.improved_training_objective && <div className="small">No parameter vector beat the baseline on training; the frozen baseline is preserved.</div>}
          </div>
        )}
        {!rec && <p className="muted small">No candidate yet. Run a hardening job to create one.</p>}

        {rec && current && (
          <div className="group">
            <label className="group-label" htmlFor="harden-episode">Compare on an episode</label>
            <select id="harden-episode" value={episode} onChange={(e) => setEpisode(e.target.value)}>
              {(["train", "validation_preview"] as const).map((split) => (
                <optgroup key={split} label={split === "train" ? "training" : "validation preview (not used for tuning)"}>
                  {replayable.filter((e) => e.split === split).map((e) => {
                    const r = rows.find((x) => x.id === e.id);
                    return <option key={e.id} value={e.id}>{e.id}{r ? ` · ${CHANGE_TEXT[r.change] ?? r.change}` : ""}</option>;
                  })}
                </optgroup>
              ))}
            </select>
            <div className="row wrap">
              <div className="segmented" role="group" aria-label="Ghost robot">
                <button aria-pressed={ghost === "baseline"} onClick={() => setGhost("baseline")}>Ghost: baseline, same shove</button>
                <button aria-pressed={ghost === "matched"} onClick={() => setGhost("matched")} disabled={!epInfo?.push}>Ghost: no shove</button>
              </div>
              <button onClick={() => void replay(current.key, rec, episode, ghost)} disabled={replaying !== null}>
                {replaying === replayKey ? "Re-simulating…" : "Replay baseline vs candidate"}
              </button>
            </div>
            {row && <EpisodeCompare row={row} />}
          </div>
        )}

        <Advanced summary="Advanced · generations and seed">
          <div className="row wrap">
            <label className="field">generations <input type="number" min={info.limits.max_generations[0]} max={info.limits.max_generations[1]} value={generations} onChange={(e) => setGenerations(Number(e.target.value))} disabled={running} /></label>
            <label className="field">seed <input placeholder="random" value={seedText} onChange={(e) => setSeedText(e.target.value)} disabled={running} /></label>
          </div>
          <span className="xsmall muted">Generations {info.limits.max_generations[0]}–{info.limits.max_generations[1]}; the server caps every run at {info.limits.wall_clock_s} s and {info.limits.max_rollouts} rollouts.</span>
        </Advanced>
        <EvidenceLink />
      </div>
      <Details>
        <HardeningDetails info={info} current={current} history={history} running={running} onOpen={poll} gateStatus={gateStatus} />
      </Details>
    </>
  );
}

function EpisodeCompare({ row }: { row: ComparisonRow }) {
  const line = (label: string, b: ComparisonRow["baseline"]) => (
    <div>
      <span className="dot" style={{ background: COLORS[b.outcome] }} /> {label}: {b.outcome === "fell" ? `fell (${b.reason}) ${fmt(b.time_after_push_s, 2)} s after the shove` : b.outcome}
      {b.max_tilt_after_push_deg !== null && `, max tilt ${fmt(b.max_tilt_after_push_deg, 1)}°`}, progress {fmt(b.progress_m, 2)} m
      {b.reflex && `, reflex active ${(100 * b.reflex.activation_fraction).toFixed(1)}% of ticks`}
    </div>
  );
  return (
    <div className="small result-box">
      {line("Frozen baseline", row.baseline)}
      {line("Candidate", row.candidate)}
      <div className="muted">Recorded in the candidate record; the replay re-simulates and checks it.</div>
    </div>
  );
}

function HistoryPlot({ history, baseline }: { history: HistoryEntry[]; baseline?: number }) {
  if (!history.length) return null;
  const W = 700;
  const H = 180;
  const pad = { l: 44, r: 10, t: 10, b: 26 };
  const base = baseline ?? history[0].baseline_objective;
  const values = history.flatMap((h) => [h.best_objective, h.generation_median]);
  const yMax = Math.max(...values, base ?? 0) * 1.1 || 1;
  const x = (g: number) => pad.l + ((g - 0.5) / history.length) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / yMax) * (H - pad.t - pad.b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="search-svg" style={{ height: 180 }}>
      {[0, yMax / 2, yMax].map((v) => (
        <g key={v}><line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke={PALETTE.border} /><text x={pad.l - 6} y={y(v) + 4} fontSize={11} textAnchor="end" fill={PALETTE.muted}>{v.toFixed(2)}</text></g>
      ))}
      {base !== undefined && <line x1={pad.l} x2={W - pad.r} y1={y(base)} y2={y(base)} stroke={PALETTE.fell} strokeDasharray="4 4"><title>frozen baseline J</title></line>}
      <polyline fill="none" stroke={PALETTE.candidate} strokeWidth={2} points={history.map((h) => `${x(h.generation)},${y(h.best_objective)}`).join(" ")} />
      {history.map((h) => <circle key={`m${h.generation}`} cx={x(h.generation)} cy={y(h.generation_median)} r={3} fill={PALETTE.faint}><title>{`generation ${h.generation} median ${h.generation_median.toFixed(3)}`}</title></circle>)}
      <text x={W / 2} y={H - 6} fontSize={11} textAnchor="middle" fill={PALETTE.muted}>generation (blue: best-so-far training J; grey: generation median; red dashed: frozen baseline)</text>
    </svg>
  );
}

function ComparisonTable({ title, rows, open }: { title: string; rows: ComparisonRow[]; open?: boolean }) {
  const recovered = rows.filter((r) => r.change === "recovered").length;
  const regressed = rows.filter((r) => r.change === "regressed").length;
  return (
    <Evidence title={title} summary={`${recovered} recovered · ${regressed} regressed · ${rows.length} episodes`} open={open}>
      <table className="cases">
        <thead><tr><th>Episode</th><th>Baseline</th><th>Candidate</th><th>Change</th><th>Tilt ° (b → c)</th><th>Progress m (b → c)</th><th>Reflex active</th><th>First activation</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td><span className="dot" style={{ background: COLORS[r.baseline.outcome] }} /> {r.baseline.outcome}{r.baseline.time_after_push_s !== null ? ` +${fmt(r.baseline.time_after_push_s, 2)} s` : ""}</td>
              <td><span className="dot" style={{ background: COLORS[r.candidate.outcome] }} /> {r.candidate.outcome}{r.candidate.time_after_push_s !== null ? ` +${fmt(r.candidate.time_after_push_s, 2)} s` : ""}</td>
              <td className={r.change === "regressed" ? "bad" : ""}>{CHANGE_TEXT[r.change] ?? r.change}</td>
              <td>{fmt(r.baseline.max_tilt_after_push_deg, 1)} → {fmt(r.candidate.max_tilt_after_push_deg, 1)}</td>
              <td>{fmt(r.baseline.progress_m, 2)} → {fmt(r.candidate.progress_m, 2)}</td>
              <td>{r.candidate.reflex ? `${(100 * r.candidate.reflex.activation_fraction).toFixed(1)}%` : "—"}</td>
              <td>{r.candidate.reflex?.first_activation_s !== null && r.candidate.reflex?.first_activation_s !== undefined ? `${fmt(r.candidate.reflex.first_activation_s, 2)} s` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Evidence>
  );
}

function HardeningDetails({ info, current, history, running, onOpen, gateStatus }: { info: HardeningInfo; current: Current | null; history: HistoryEntry[]; running: boolean; onOpen: (id: string) => void; gateStatus: GateStatus | null }) {
  const rec = current?.record;
  const grids = current?.origin === "precomputed" ? info.preset?.grids : null;
  return (
    <>
      {history.length > 0 && (
        <Evidence title={running ? "CPU optimization progress (live)" : "Optimization history"} summary={running ? "updates every generation" : `${rec?.candidate_id} · ${history.length} generations`} open={running}>
          <HistoryPlot history={history} baseline={running ? undefined : rec?.objective.baseline_training} />
        </Evidence>
      )}
      {rec && (
        <>
          <p className="finding small">{rec.result.statement} Training and validation-preview sets are reported separately; the locked test set
            ({rec.sets.locked_test.episodes} episodes, sha {rec.sets.locked_test.sha256.slice(0, 12)}) was not evaluated by this hardening run. The M7 gate
            evaluated it once for the five registered candidates; see Results · Regression gate.</p>
          {rec.measurements.train.comparison && <ComparisonTable title="Training set (used for tuning)" rows={rec.measurements.train.comparison} open />}
          {rec.measurements.validation_preview.comparison && <ComparisonTable title="Validation preview (not used for tuning; distinct gait phases and attack configurations)" rows={rec.measurements.validation_preview.comparison} />}
          {grids && (
            <Evidence title="Exploratory dense grids (precomputed)" summary="overlap training/validation timings, locked timings skipped; not a holdout">
              <table>
                <thead><tr><th>Grid</th><th>Tested</th><th>Baseline fell</th><th>Candidate fell</th><th>Regressions</th></tr></thead>
                <tbody>
                  {Object.entries(grids.grids).map(([name, g]) => (
                    <tr key={name}><td>{name} (toward the right, 0.1 s, every 4 ms in 2.000–3.020 s)</td><td>{g.tested}</td><td>{g.baseline_fell}</td><td>{g.candidate_fell}</td><td className={g.regressions.length ? "bad" : ""}>{g.regressions.length ? g.regressions.join(", ") : "none"}</td></tr>
                  ))}
                </tbody>
              </table>
            </Evidence>
          )}
          <Evidence title="Reflex parameters and provenance" summary={`reproducibility ${rec.reproducibility_sha256.slice(0, 12)}`}>
          <div className="two-col">
            <div className="card">
              <h3>Reflex parameters</h3>
              <table className="small">
                <thead><tr><th>Parameter</th><th>Value</th><th>Range</th><th>Unit</th></tr></thead>
                <tbody>
                  {rec.reflex.schema.parameters.map((p) => (
                    <tr key={p.name} title={p.doc}><td>{p.name}</td><td>{fmt(rec.reflex.parameters[p.name], 3)}</td><td>[{p.lo}, {p.hi}]</td><td>{p.unit}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card small">
              <h3>Provenance</h3>
              <dl className="facts">
                <dt>Record</dt><dd>{rec.status} · written at M6: “{rec.regression_gate}” (records are immutable)</dd>
                <dt>Gate</dt><dd>{gateStatus ? gateStatus.label : "not evaluated by the regression gate"}</dd>
                <dt>Base policy</dt><dd>{rec.base_policy.path} · sha {rec.base_policy.sha256.slice(0, 12)} · weights unmodified</dd>
                <dt>Sets</dt><dd>{rec.sets.name} · sha {rec.sets.sha256.slice(0, 12)}</dd>
                <dt>Optimizer</dt><dd>CMA-ES seed {rec.optimizer.config.seed}, popsize {rec.optimizer.config.popsize}, ≤ {rec.optimizer.config.max_generations} generations{rec.optimizer.config.focus ? `, focus ${rec.optimizer.config.focus}` : ""}</dd>
                <dt>Rollouts</dt><dd>{rec.accounting.optimization_rollouts} optimization + {rec.accounting.baseline_rollouts} baseline + {rec.accounting.reference_rollouts} reference + {rec.accounting.validation_preview_rollouts} validation preview</dd>
                <dt>Commit</dt><dd>{rec.provenance.git.commit.slice(0, 7)}{rec.provenance.git.dirty ? " (dirty)" : ""}</dd>
                <dt>Reproducibility</dt><dd>{rec.reproducibility_sha256.slice(0, 16)}</dd>
                {rec.request && <><dt>Reproduce</dt><dd><code>{rec.request.reproduce}</code></dd></>}
              </dl>
            </div>
          </div>
          </Evidence>
        </>
      )}
      <details className="card">
        <summary>About the data sets</summary>
        <ul>{info.sets.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      </details>
      {info.runs.length > 0 && (
        <details className="card">
          <summary>Previous hardening runs ({info.runs.length})</summary>
          <table>
            <thead><tr><th>Run</th><th>Seed</th><th>Focus</th><th>Status</th><th>Train survived</th><th>Validation survived</th><th /></tr></thead>
            <tbody>
              {info.runs.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.run_id}</td><td>{r.seed}</td><td>{r.focus ?? "all"}</td><td>{r.status}{r.stop_reason ? ` (${r.stop_reason})` : ""}</td>
                  <td>{surv(r.train)}</td><td>{surv(r.validation)}</td>
                  <td>{r.status === "done" && <button onClick={() => onOpen(r.run_id)} disabled={running}>Open</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </>
  );
}
