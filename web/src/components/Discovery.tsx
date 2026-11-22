import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  COLORS,
  fmt,
  type DiscoveryInfo,
  type DiscoveryRecord,
  type DiscoveryStatus,
  type Evaluation,
  type PairedTrialRecord,
  type SearchCandidate,
} from "../api";
import { Details, Panel } from "./Workspace";

interface Props {
  info: DiscoveryInfo;
  onReplay: (runId: string, which: "final" | "search") => void;
  onRefresh: () => void;
  replaying: string | null;
}

const ALGORITHMS = [
  { id: "random", label: "Random Search" },
  { id: "cmaes", label: "CMA-ES" },
];

function stanceText(stance: Record<string, boolean> | null | undefined): string {
  if (!stance) return "—";
  const down = ["FR", "FL", "RR", "RL"].filter((f) => stance[f]);
  return down.length ? down.join("+") : "none";
}

function candidateText(c: SearchCandidate): string {
  return `${c.force_n.toFixed(1)} N at ${c.direction_deg.toFixed(1)}°, ${c.start_s.toFixed(3)}–${(c.start_s + c.duration_s).toFixed(3)} s (${(c.duration_s * 1000).toFixed(0)} ms)`;
}

export function Discovery({ info, onReplay, onRefresh, replaying }: Props) {
  const [algorithm, setAlgorithm] = useState("cmaes");
  const [budget, setBudget] = useState(info.defaults.budget);
  const [shrinkBudget, setShrinkBudget] = useState(info.defaults.shrink_budget);
  const [seedText, setSeedText] = useState("");
  const [status, setStatus] = useState<DiscoveryStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const autoReplay = useRef<string | null>(null); // replay the result of a run started here
  const lim = info.limits;

  const poll = useCallback(
    (runId: string) => {
      const tick = async () => {
        try {
          const s = await api.discoveryStatus(runId);
          setStatus(s);
          if (s.status === "queued" || s.status === "running") timer.current = window.setTimeout(tick, 250);
          else {
            onRefresh();
            if (autoReplay.current === runId && s.record?.final) onReplay(runId, "final");
          }
        } catch (e) {
          setError(String(e));
        }
      };
      void tick();
    },
    [onRefresh],
  );
  useEffect(() => {
    // Deep links: ?tab=discovery&run=<id> opens a run; &dreplay=final|search also opens its replay.
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("run");
    if (runId && /^ds-\d{8}T\d{6}Z-(random|cmaes)-\d{1,20}$/.test(runId)) {
      poll(runId);
      const which = params.get("dreplay");
      if (which === "final" || which === "search") onReplay(runId, which);
    }
    return () => { if (timer.current) window.clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async () => {
    setError(null);
    const seed = seedText.trim() === "" ? null : Number(seedText);
    if (seed !== null && (!Number.isInteger(seed) || seed < 0)) return setError("Seed must be a non-negative integer or blank.");
    if (!Number.isInteger(budget) || budget < lim.budget.min || budget > lim.budget.max || budget % lim.budget.multiple_of)
      return setError(`Budget must be a multiple of ${lim.budget.multiple_of} between ${lim.budget.min} and ${lim.budget.max}.`);
    if (!Number.isInteger(shrinkBudget) || shrinkBudget < lim.shrink_budget.min || shrinkBudget > lim.shrink_budget.max)
      return setError(`Shrink budget must be between ${lim.shrink_budget.min} and ${lim.shrink_budget.max}.`);
    try {
      const started = await api.startDiscovery({ algorithm, budget, seed, shrink_budget: shrinkBudget });
      autoReplay.current = started.run_id;
      poll(started.run_id);
    } catch (e) {
      setError(String(e));
    }
  };
  const cancel = async () => {
    if (!status) return;
    try {
      await api.cancelDiscovery(status.run_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const running = status?.status === "queued" || status?.status === "running";
  const record = status?.record;
  const evaluations: Evaluation[] = record?.evaluations ?? status?.evaluations ?? [];
  const runBudget = record?.config.budget ?? status?.request.budget ?? budget;
  const candidateFails = evaluations.filter((e) => e.outcome === "fell");
  const bestCandidate = candidateFails.reduce<Evaluation | null>((b, e) => (b === null || e.candidate.impulse_ns < b.candidate.impulse_ns ? e : b), null);

  const final = record?.final ?? null;
  return (
    <>
      <Panel eyebrow="Failure Discovery · M4" title="Search for a low-impulse shove that makes the robot fall">
        <p className="small">
          Force 10–150 N, direction 0–360° in the starting frame (0° forward, 90° left), start 2.000–3.020 s, duration 40–200 ms.
          Every candidate is a real MuJoCo rollout on the server. The target is the lowest impulse among <em>confirmed</em> falls
          found within the budget, not a global minimum.
        </p>
        <div className="row wrap">
          {ALGORITHMS.map((a) => (
            <label key={a.id} className="radio">
              <input type="radio" name="alg" checked={algorithm === a.id} onChange={() => setAlgorithm(a.id)} disabled={running} /> {a.label}
            </label>
          ))}
        </div>
        <div className="row wrap">
          <label className="field">budget <input type="number" min={lim.budget.min} max={lim.budget.max} step={lim.budget.multiple_of} value={budget} onChange={(e) => setBudget(Number(e.target.value))} disabled={running} /></label>
          <label className="field">shrink <input type="number" min={lim.shrink_budget.min} max={lim.shrink_budget.max} value={shrinkBudget} onChange={(e) => setShrinkBudget(Number(e.target.value))} disabled={running} /></label>
          <label className="field">seed <input placeholder="random" value={seedText} onChange={(e) => setSeedText(e.target.value)} disabled={running} /></label>
        </div>
        <div className="row">
          <button className="primary big" onClick={start} disabled={running}>
            {running ? `${status?.stage === "search" ? "Searching" : status?.stage ?? "Running"}… ${status?.completed ?? 0}/${runBudget}` : "▶ Discover Failure"}
          </button>
          {running && <button onClick={cancel} disabled={status?.cancel_requested}>{status?.cancel_requested ? "Cancelling…" : "Cancel"}</button>}
        </div>
        {running && <div className="progress"><div style={{ width: `${(100 * (status?.completed ?? 0)) / runBudget}%` }} /></div>}
        {error && <p className="error">{error}</p>}
        {status?.status === "failed" && <p className="error">Search failed: {status.error}</p>}
        <p className="muted small">Server limit {lim.timeout_s} s per search; one expensive job at a time.</p>
        {status && (
          <div className="card inset small">
            <strong>{record ? "Executed now" : "Executing"} · {status.request.algorithm === "cmaes" ? "CMA-ES" : "Random Search"} · seed {status.request.seed} · run {status.run_id}</strong>
            <div>{evaluations.length} evaluations · {candidateFails.length} candidate falls
              {!record && bestCandidate && <> · best so far {fmt(bestCandidate.candidate.impulse_ns, 2)} N·s <span className="muted">(not yet confirmed)</span></>}
            </div>
            {record && !record.search.found_failure && <div className="bad">No physical failure found in {record.accounting.search_rollouts} evaluations. Nothing is shown in its place.</div>}
            {final && (
              <>
                <div>Final ({final.source}): <strong>{candidateText(final.candidate)}</strong> = {fmt(final.candidate.impulse_ns, 2)} N·s</div>
                <button className="primary" onClick={() => onReplay(status.run_id, "final")} disabled={replaying !== null}>
                  {replaying === `discovery/${status.run_id}/final` ? "Simulating…" : "▶ Replay Discovered Failure"}
                </button>
              </>
            )}
          </div>
        )}
      </Panel>
      <Details>
        {status && (
          <div className="card">
            <div className="sweep-head">
              <strong>
                {record ? "Executed now" : "Executing"} · {status.request.algorithm === "cmaes" ? "CMA-ES" : "Random Search"} · seed {status.request.seed} ({status.request.seed_source}) · run {status.run_id}
              </strong>
              {record && <span className="badge live">new simulations · commit {record.provenance.git.commit.slice(0, 7)} · {fmt(record.wall_s, 1)} s</span>}
            </div>
            <SearchPlot evaluations={evaluations} budget={runBudget} record={record ?? null} />
            {record && <Result record={record} onReplay={(which) => onReplay(status.run_id, which)} replaying={replaying} runId={status.run_id} />}
          </div>
        )}
        <ComparisonCard info={info} />
        {info.runs.length > 0 && (
          <details className="card">
            <summary>Previous discovery runs ({info.runs.length})</summary>
            <table>
              <thead><tr><th>Run</th><th>Algorithm</th><th>Budget</th><th>Seed</th><th>Result</th><th /></tr></thead>
              <tbody>
                {info.runs.map((r) => (
                  <tr key={r.run_id}>
                    <td>{r.run_id}</td><td>{r.algorithm}</td><td>{r.budget}</td><td>{r.seed}</td>
                    <td>{r.status !== "done" ? r.status : r.found_failure ? `${fmt(r.final_impulse_ns, 2)} N·s (${r.final_source})` : `no failure (${r.stop_reason})`}</td>
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

function SearchPlot({ evaluations, budget, record }: { evaluations: Evaluation[]; budget: number; record: DiscoveryRecord | null }) {
  const shrinkTests = record?.shrink?.tested ?? [];
  const W = 900;
  const H = 220;
  const pad = { l: 44, r: 10, t: 10, b: 26 };
  const gap = shrinkTests.length ? 4 : 0;
  const slots = budget + gap + shrinkTests.length;
  const x = (i: number) => pad.l + ((i - 0.5) / Math.max(slots, 1)) * (W - pad.l - pad.r);
  const yMax = 30;
  const y = (impulse: number) => pad.t + (1 - impulse / yMax) * (H - pad.t - pad.b);
  const confirmedIndex = record?.confirmation.confirmed?.evaluation ?? null;
  const finalImpulse = record?.final?.candidate.impulse_ns ?? null;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="search-svg">
      {[0, 10, 20, 30].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="#e3e7ec" />
          <text x={pad.l - 6} y={y(v) + 4} fontSize={11} textAnchor="end" fill="#66707c">{v}</text>
        </g>
      ))}
      <text x={12} y={H / 2} fontSize={11} fill="#66707c" transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle">impulse N·s</text>
      <text x={x(budget / 2)} y={H - 6} fontSize={11} fill="#66707c" textAnchor="middle">search evaluations 1–{budget}</text>
      {shrinkTests.length > 0 && <text x={x(budget + gap + shrinkTests.length / 2)} y={H - 6} fontSize={11} fill="#66707c" textAnchor="middle">shrink</text>}
      {finalImpulse !== null && <line x1={pad.l} x2={W - pad.r} y1={y(finalImpulse)} y2={y(finalImpulse)} stroke="#d64545" strokeDasharray="4 4" />}
      {evaluations.map((e) => (
        <circle key={e.index} cx={x(e.index)} cy={y(e.candidate.impulse_ns)} r={e.outcome === "fell" ? 4.5 : 3} fill={COLORS[e.outcome]} opacity={e.outcome === "survived" ? 0.55 : 0.95}>
          <title>{`#${e.index} ${e.outcome}${e.reason ? ` (${e.reason})` : ""}: ${candidateText(e.candidate)} = ${e.candidate.impulse_ns.toFixed(2)} N·s`}</title>
        </circle>
      ))}
      {confirmedIndex !== null && (() => {
        const e = evaluations.find((v) => v.index === confirmedIndex);
        return e ? <circle cx={x(e.index)} cy={y(e.candidate.impulse_ns)} r={8} fill="none" stroke="#111" strokeWidth={2}><title>confirmed search failure</title></circle> : null;
      })()}
      {shrinkTests.map((t, i) => {
        const cx = x(budget + gap + i + 1);
        const cy = y(t.candidate.impulse_ns);
        const color = t.status === "unconfirmed" ? "#e0a030" : COLORS[t.outcome];
        return t.status === "confirmed_improvement" ? (
          <path key={i} d={`M ${cx} ${cy - 7} L ${cx + 7} ${cy} L ${cx} ${cy + 7} L ${cx - 7} ${cy} Z`} fill={COLORS.fell} stroke="#111" strokeWidth={1.5}>
            <title>{`shrunk (confirmed): ${candidateText(t.candidate)} = ${t.candidate.impulse_ns.toFixed(2)} N·s`}</title>
          </path>
        ) : (
          <rect key={i} x={cx - 3} y={cy - 3} width={6} height={6} fill={color} opacity={0.8}>
            <title>{`shrink ${t.phase}: ${t.status}: ${candidateText(t.candidate)} = ${t.candidate.impulse_ns.toFixed(2)} N·s`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

function FailureRow({ label, kind, c, trial, evaluation }: { label: string; kind: string; c: SearchCandidate; trial?: PairedTrialRecord | null; evaluation?: Evaluation }) {
  const stance = trial ? stanceText(trial.stance_at_start) : evaluation?.stance_at_start ?? "—";
  const fall = trial
    ? `${trial.attacked.reason} @ ${fmt(trial.attacked.fall_time_s)} s`
    : evaluation
      ? `${evaluation.reason} @ ${fmt(evaluation.fall_time_s)} s`
      : "—";
  return (
    <tr>
      <td><strong>{label}</strong><div className="muted small">{kind}</div></td>
      <td>{c.force_n.toFixed(1)} N</td>
      <td>{c.direction_deg.toFixed(1)}°</td>
      <td>{c.start_s.toFixed(3)} s</td>
      <td>{(c.duration_s * 1000).toFixed(0)} ms</td>
      <td><strong>{c.impulse_ns.toFixed(2)}</strong></td>
      <td>{stance}</td>
      <td>{fall}</td>
    </tr>
  );
}

function Result({ record, onReplay, replaying, runId }: { record: DiscoveryRecord; onReplay: (which: "final" | "search") => void; replaying: string | null; runId: string }) {
  const s = record.search;
  const a = record.accounting;
  const best = s.best_candidate_index ? record.evaluations[s.best_candidate_index - 1] : null;
  const confirmed = record.confirmation.confirmed;
  const shrink = record.shrink;
  return (
    <div>
      <p className="result-line">
        Stopped: <strong>{s.stop_reason}</strong> · {a.search_rollouts} search rollouts{a.cache_hits ? ` (+${a.cache_hits} cached repeats, not simulated)` : ""} ·{" "}
        {s.candidate_failures} candidate falls · {s.numerical} numerical · first fall at evaluation {s.evaluations_to_first_failure ?? "—"}
      </p>
      {!s.found_failure ? (
        <p className="banner">No physical failure was found in {a.search_rollouts} evaluations. Nothing is shown in its place.</p>
      ) : (
        <table className="failures">
          <thead><tr><th /><th>Force</th><th>Direction</th><th>Start</th><th>Duration</th><th>Impulse N·s</th><th>Stance</th><th>Fall</th></tr></thead>
          <tbody>
            {best && <FailureRow label="Best candidate" kind={`evaluation #${best.index}, single rollout`} c={best.candidate} evaluation={best} />}
            {confirmed ? <FailureRow label="Confirmed" kind={`paired re-simulation of #${confirmed.evaluation}`} c={confirmed.candidate} trial={confirmed.trial} /> : <tr><td colSpan={8} className="bad">Candidate could not be confirmed.</td></tr>}
            {shrink && shrink.improved && record.final && <FailureRow label="Shrunk" kind="confirmed lower-impulse failure" c={record.final.candidate} trial={record.final.trial} />}
          </tbody>
        </table>
      )}
      {shrink && (
        <p className="muted small">
          Shrinking: {shrink.improved ? `${fmt(shrink.start.impulse_ns, 2)} → ${fmt(shrink.final.impulse_ns, 2)} N·s` : "no confirmed improvement"} in {shrink.rollouts} rollouts
          (+{shrink.confirmation_rollouts} confirmation){shrink.budget_exhausted ? ", budget exhausted" : ""}; {shrink.surviving_below_final.length} tested points below the final impulse survived. {shrink.statement}
        </p>
      )}
      {record.final && (
        <div className="row wrap">
          <button className="primary" onClick={() => onReplay("final")} disabled={replaying !== null}>
            {replaying === `discovery/${runId}/final` ? "Simulating…" : `▶ Replay Discovered Failure (${record.final.source})`}
          </button>
          {record.final.source === "shrunk" && (
            <button onClick={() => onReplay("search")} disabled={replaying !== null}>Replay confirmed search failure</button>
          )}
        </div>
      )}
      <p className="muted small">
        Rollouts: {a.physics_rollouts_total} total = search {a.search_rollouts} + confirmation {a.confirmation_rollouts} + shrink {a.shrink_rollouts} + shrink confirmation {a.shrink_confirmation_rollouts} + 1 nominal reference.
        Reproduce: <code>{record.request?.reproduce}</code> · results {record.results_sha256.slice(0, 12)}
      </p>
      <p className="muted small">{record.statement}</p>
    </div>
  );
}

function ComparisonCard({ info }: { info: DiscoveryInfo }) {
  const c = info.comparison;
  const r = c.aggregate.random;
  const m = c.aggregate.cmaes;
  const p = c.paired;
  return (
    <div className="card reference comparison">
      <span className="badge pre">Precomputed · matched comparison · commit {c.git.commit.slice(0, 7)}</span>
      <p className="small">{c.protocol.seeds.length} seeds × budget {c.protocol.budget} + shrink {c.protocol.shrink_budget}, same seeds and settings, no warm start.</p>
      <table className="small">
        <thead><tr><th /><th>Random</th><th>CMA-ES</th></tr></thead>
        <tbody>
          <tr><td>found a fall</td><td>{r.found_failure}/{r.runs}</td><td>{m.found_failure}/{m.runs}</td></tr>
          <tr><td>median evals to 1st</td><td>{r.median_evaluations_to_first_failure}</td><td>{m.median_evaluations_to_first_failure}</td></tr>
          <tr><td>median confirmed N·s</td><td>{fmt(r.median_search_confirmed_impulse_ns, 1)}</td><td>{fmt(m.median_search_confirmed_impulse_ns, 1)}</td></tr>
          <tr><td>median after shrink</td><td>{fmt(r.median_final_impulse_ns, 1)}</td><td>{fmt(m.median_final_impulse_ns, 1)}</td></tr>
          <tr><td>lowest found</td><td>{fmt(r.min_final_impulse_ns, 2)}</td><td>{fmt(m.min_final_impulse_ns, 2)}</td></tr>
        </tbody>
      </table>
      <p className="muted small">
        Found-a-fall difference: Fisher p = {fmt(c.found_failure_fisher_p_two_sided, 3)}. Where both found one (n = {p.final_impulse_both_found.pairs}), CMA-ES was lower in{" "}
        {p.final_impulse_both_found.cmaes_better} and Random in {p.final_impulse_both_found.random_better} (sign test p = {fmt(p.final_impulse_both_found.sign_test_p_two_sided, 2)}).
      </p>
    </div>
  );
}
