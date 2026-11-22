// Results: the recorded evidence of M2–M7 in one place (precomputed presets plus recent server
// runs), each with a replay into the shared stage. Replaces the M3–M6 "Hidden Phase" tab; the
// recorded sweep view lives here.
import { useEffect, useRef, useState } from "react";
import {
  COLORS,
  fmt,
  gateApi,
  hardeningApi,
  pct,
  type CompactTrial,
  type DiscoveryInfo,
  type GateInfo,
  type HardeningInfo,
  type HiddenPhase as HiddenPhaseData,
  type LiveStress,
  type Replay,
  type ScenarioConfig,
} from "../api";
import { PALETTE } from "../theme";
import { gateHeadline, GateEvidence } from "./GateEvidence";
import { Details, Evidence, EvidenceLink, ModeIntro } from "./Workspace";

interface Props {
  hidden: HiddenPhaseData;
  live: LiveStress | null;
  discovery: DiscoveryInfo | null;
  designer: ScenarioConfig | null;
  hardening: HardeningInfo | null;
  gate: GateInfo | null;
  onHidden: (substep: number) => void;
  onLive: (runId: string, trial: number) => void;
  onDiscovery: (runId: string, which: "final" | "search") => void;
  onScenario: (runId: string, trial: number, ghost: "primary" | "matched" | "original", title: string) => void;
  show: (key: string, title: string, load: () => Promise<Replay>) => Promise<void>;
  replaying: string | null;
  hiddenLoading: number | null;
}

export function Results({ hidden, live, discovery, designer, hardening, gate, onHidden, onLive, onDiscovery, onScenario, show, replaying, hiddenLoading }: Props) {
  const [selected, setSelected] = useState<number>(hidden.verified_failure_substep);
  const att = hidden.aggregates.attacked;
  const failing = hidden.trials.filter((t) => t.outcome === "fell");
  const busy = replaying !== null;
  const liveRef = live?.reference.aggregates.attacked;
  const comparison = discovery?.comparison;
  const preset = hardening?.preset;
  const latestLive = live?.runs.find((r) => r.status === "done");
  const latestDiscovery = discovery?.runs.find((r) => r.status === "done" && r.found_failure);
  const latestScenario = designer?.runs.find((r) => r.status === "done");
  const replayCandidate = (key: string, id: string, episode: string) =>
    show(`hardening/${key}/${episode}/baseline`, `Hardening · ${episode} · candidate ${id.slice(5, 13)} vs frozen baseline`, () => hardeningApi.replay(key, episode, "baseline"));
  const replayGate = (candidate: string, episode: string, ghost: "baseline" | "matched" = "baseline") =>
    show(`gate/${candidate}/${episode}/${ghost}`, `Gate · ${episode} · candidate ${candidate.slice(5, 13)} vs frozen baseline`, () => gateApi.replay(candidate, episode, ghost));
  const primary = gate?.candidates?.find((c) => c.primary);
  const gateText = gate ? gateHeadline(gate) : null;
  const firstRecovered = primary?.G2_holdout.recovered[0];
  const presetGate = preset?.gate_status;
  const linked = useRef(false);
  useEffect(() => {
    // Deep link: ?tab=results&greplay=<candidate>:<episode>:<baseline|matched>
    if (linked.current || !gate?.available) return;
    linked.current = true;
    const m = new URLSearchParams(window.location.search).get("greplay")?.match(/^(cand-[0-9a-f]{12}):([A-Za-z0-9._-]{1,64}):(baseline|matched)$/);
    if (m) void replayGate(m[1], m[2], m[3] as "baseline" | "matched");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gate]);

  return (
    <>
      <div className="mode">
        <ModeIntro eyebrow="Results · recorded evidence" title="What the simulator has measured so far">
          <p>Precomputed presets from earlier milestones and the latest runs on this server. Every replay is re-simulated or loaded from the recorded trajectory.</p>
        </ModeIntro>
        <div className="summary-box">
          <div className="group-label">Recorded sweep · M2</div>
          <div><strong>{att.physical_falls}/{att.valid_trials}</strong> shove timings fell ({pct(att.fall_rate)}) at {hidden.config.attack.force_n} N toward the right, every 4 ms start in [{hidden.sampling.interval_s[0].toFixed(3)}, {hidden.sampling.interval_s[1].toFixed(3)}) s</div>
          <div className="chip-row" aria-label="Failing timings">
            {failing.map((t) => (
              <button key={t.start_substep} className="chip-btn" aria-pressed={selected === t.start_substep} disabled={busy}
                onClick={() => { setSelected(t.start_substep); onHidden(t.start_substep); }} title={`${t.stance} · fell ${fmt(t.fall_time_s)} s`}>
                {hiddenLoading === t.start_substep ? "…" : `${t.start_s.toFixed(3)} s`}
              </button>
            ))}
          </div>
        </div>
        <div className="cta">
          <button className="primary big wide" onClick={() => { setSelected(hidden.verified_failure_substep); onHidden(hidden.verified_failure_substep); }} disabled={busy}>
            {hiddenLoading === hidden.verified_failure_substep ? "Re-simulating…" : "Replay a recorded failure"}
          </button>
        </div>
        <div className="stack">
          {liveRef && (
            <ResultRow label="Stress test · M3" text={`${liveRef.physical_falls}/${liveRef.valid_trials} timings fell at 150 N (exhaustive)`}
              action={latestLive ? { label: `Replay latest run`, onClick: () => onLive(latestLive.run_id, 0) } : undefined} disabled={busy} />
          )}
          {comparison && (
            <ResultRow label="Failure discovery · M4" text={`CMA-ES found a fall in ${comparison.aggregate.cmaes.found_failure}/${comparison.aggregate.cmaes.runs} seeds, Random Search in ${comparison.aggregate.random.found_failure}/${comparison.aggregate.random.runs}`}
              action={latestDiscovery ? { label: "Replay latest discovery", onClick: () => onDiscovery(latestDiscovery.run_id, "final") } : undefined} disabled={busy} />
          )}
          {designer && (
            <ResultRow label="Scenarios · M5" text={latestScenario ? `latest: ${latestScenario.name} (${latestScenario.mode}) · ${latestScenario.mode === "single" ? latestScenario.outcome : `${latestScenario.physical_falls}/${latestScenario.valid_trials} fell`}` : "no scenario runs on this server yet"}
              action={latestScenario ? { label: "Replay latest scenario", onClick: () => onScenario(latestScenario.run_id, 0, "primary", `Scenario ${latestScenario.name}`) } : undefined} disabled={busy} />
          )}
          {preset && (
            <ResultRow label="Hardening candidate · M6" text={`sample candidate: training ${preset.record.measurements.train.candidate_summary?.attack_survived}/${preset.record.measurements.train.candidate_summary?.attacks}, validation ${preset.record.measurements.validation_preview.candidate_summary?.attack_survived}/${preset.record.measurements.validation_preview.candidate_summary?.attacks} attacks survived · ${presetGate ? presetGate.label : "not evaluated by the regression gate"}`}
              action={{ label: "Replay baseline vs candidate", onClick: () => replayCandidate(preset.key, preset.record.candidate_id, "m2-100N-500") }} disabled={busy} />
          )}
          {gate && (
            <ResultRow label="Regression gate · M7" text={gateText ?? "no gate result on this server"}
              action={primary && firstRecovered ? { label: "Replay a locked holdout attack", onClick: () => replayGate(primary.candidate_id, firstRecovered) } : undefined} disabled={busy} />
          )}
        </div>
        <EvidenceLink />
      </div>
      <Details>
        {gate && <GateEvidence gate={gate} onReplay={replayGate} busy={busy} />}
        <Evidence title="Recorded sweep (M2)" summary={`${att.physical_falls}/${att.valid_trials} timings fell · results ${hidden.results_sha256.slice(0, 8)}`} open>
          <SweepChart trials={hidden.trials} selected={selected} onSelect={(k) => { setSelected(k); }} />
          <SelectedTiming trial={hidden.trials.find((t) => t.start_substep === selected)!} onReplay={() => onHidden(selected)} busy={busy} />
          {hidden.stance_finding && (
            <p className="finding small">
              All {att.physical_falls} falls happened while the front-right and rear-left feet were in stance (FR+RL), but only{" "}
              <strong>{hidden.stance_finding.fell} of the {hidden.stance_finding.tested}</strong> FR+RL timings fell. Stance alone does not predict a fall.
            </p>
          )}
          <div className="two-col">
            <div>
              <h3>Outcome by stance at the shove</h3>
              <table>
                <thead><tr><th>Feet down</th><th>Tested</th><th>Fell</th></tr></thead>
                <tbody>{hidden.stance_table.map((r) => <tr key={r.stance}><td>{r.stance}</td><td>{r.tested}</td><td className={r.fell ? "bad" : ""}>{r.fell}</td></tr>)}</tbody>
              </table>
            </div>
            <div className="small muted">
              <h3>Provenance</h3>
              Precomputed M2 sweep at commit {hidden.provenance.git.commit.slice(0, 7)}; a replay re-runs that paired trial on the server and checks it against the recorded
              trajectory digests. Stance = foot contact at the last control tick before the shove, nominal run.
            </div>
          </div>
        </Evidence>
        <Evidence title="What the recorded sweep does and does not show">
          <ul>{hidden.interpretation.map((line) => <li key={line}>{line}</li>)}</ul>
        </Evidence>
        <Evidence title="Recent runs on this server" summary="replay any finished run">
          <RecentRuns live={live} discovery={discovery} designer={designer} hardening={hardening} onLive={onLive} onDiscovery={onDiscovery} onScenario={onScenario} onCandidate={replayCandidate} busy={busy} />
        </Evidence>
      </Details>
    </>
  );
}

function ResultRow({ label, text, action, disabled }: { label: string; text: string; action?: { label: string; onClick: () => void }; disabled: boolean }) {
  return (
    <div className="summary-box">
      <div className="group-label">{label}</div>
      <div>{text}</div>
      {action && <div><button onClick={action.onClick} disabled={disabled}>{action.label}</button></div>}
    </div>
  );
}

function SelectedTiming({ trial, onReplay, busy }: { trial: CompactTrial; onReplay: () => void; busy: boolean }) {
  const events = Object.entries(trial.events).sort((a, b) => a[1] - b[1]);
  return (
    <div className="card inset">
      <div className="row wrap" style={{ justifyContent: "space-between" }}>
        <strong>Selected timing: {fmt(trial.start_s)} s</strong>
        <button onClick={onReplay} disabled={busy}>Replay this timing</button>
      </div>
      <dl className="facts">
        <dt>Outcome</dt><dd><span className="dot" style={{ background: COLORS[trial.outcome] }} /> {trial.outcome === "fell" ? `fell (${trial.reason})` : trial.outcome}</dd>
        <dt>Stance at shove</dt><dd>{trial.stance}</dd>
        <dt>First fall</dt><dd>{trial.fall_time_s === null ? "—" : `${fmt(trial.fall_time_s)} s (+${fmt(trial.time_after_start_s)} s)`}</dd>
        <dt>Events</dt><dd>{events.length ? events.map(([n, t]) => `${n} ${fmt(t)} s`).join(", ") : "none"}</dd>
        <dt>Progress</dt><dd>{fmt(trial.forward_progress_m, 2)} m shoved vs {fmt(trial.nominal_progress_m, 2)} m nominal</dd>
      </dl>
    </div>
  );
}

function SweepChart({ trials, selected, onSelect }: { trials: CompactTrial[]; selected: number; onSelect: (k: number) => void }) {
  const w = 4;
  const width = trials.length * w;
  const stanceColor = (s: string) => (s === "FR+RL" ? PALETTE.warn : s === "FL+RR" ? PALETTE.candidate : PALETTE.surface3);
  return (
    <div>
      <div className="sweep-head">
        <span className="small">All {trials.length} tested shove start times (select one)</span>
        <span className="legend">
          <span className="dot" style={{ background: COLORS.survived }} /> survived <span className="dot" style={{ background: COLORS.fell }} /> fell
          <span className="dot" style={{ background: COLORS.numerical }} /> numerical <span className="swatch" style={{ background: PALETTE.warn }} /> FR+RL stance
          <span className="swatch" style={{ background: PALETTE.candidate }} /> FL+RR stance
        </span>
      </div>
      <svg viewBox={`0 0 ${width} 46`} preserveAspectRatio="none" className="sweep-svg" role="img" aria-label="Outcome of every tested shove start time">
        {trials.map((t, i) => (
          <g key={t.start_substep} onClick={() => onSelect(t.start_substep)} style={{ cursor: "pointer" }}>
            <title>{`${t.start_s.toFixed(3)} s · ${t.stance} · ${t.outcome}${t.fall_time_s !== null ? ` @ ${t.fall_time_s.toFixed(3)} s` : ""}`}</title>
            <rect x={i * w} y={t.outcome === "survived" ? 18 : 0} width={w - 0.6} height={t.outcome === "survived" ? 14 : 32} fill={COLORS[t.outcome]} />
            <rect x={i * w} y={36} width={w} height={8} fill={stanceColor(t.stance)} />
            {t.start_substep === selected && <rect x={i * w - 0.5} y={0} width={w + 0.4} height={46} fill="none" stroke={PALETTE.text} strokeWidth={1} />}
          </g>
        ))}
      </svg>
      <div className="axis"><span>{trials[0].start_s.toFixed(3)} s</span><span>shove start time (4 ms steps)</span><span>{trials[trials.length - 1].start_s.toFixed(3)} s</span></div>
    </div>
  );
}

function RecentRuns({ live, discovery, designer, hardening, onLive, onDiscovery, onScenario, onCandidate, busy }: {
  live: LiveStress | null; discovery: DiscoveryInfo | null; designer: ScenarioConfig | null; hardening: HardeningInfo | null;
  onLive: Props["onLive"]; onDiscovery: Props["onDiscovery"]; onScenario: Props["onScenario"];
  onCandidate: (key: string, id: string, episode: string) => void; busy: boolean;
}) {
  const rows: { kind: string; id: string; when: string; text: string; replay?: () => void }[] = [];
  live?.runs.forEach((r) => rows.push({ kind: "Stress test", id: r.run_id, when: r.requested_utc, text: r.status === "done" ? `${r.physical_falls}/${r.valid_trials} fell` : r.status, replay: r.status === "done" ? () => onLive(r.run_id, 0) : undefined }));
  discovery?.runs.forEach((r) => rows.push({ kind: "Discovery", id: r.run_id, when: r.run_id.slice(3, 19), text: r.status !== "done" ? r.status : r.found_failure ? `${fmt(r.final_impulse_ns, 2)} N·s (${r.final_source})` : `no failure (${r.stop_reason})`, replay: r.status === "done" && r.found_failure ? () => onDiscovery(r.run_id, "final") : undefined }));
  designer?.runs.forEach((r) => rows.push({ kind: "Scenario", id: r.run_id, when: r.requested_utc, text: r.status !== "done" ? r.status : r.mode === "single" ? `${r.name}: ${r.outcome}` : `${r.name}: ${r.physical_falls}/${r.valid_trials} fell`, replay: r.status === "done" ? () => onScenario(r.run_id, 0, "primary", `Scenario ${r.name}`) : undefined }));
  hardening?.runs.forEach((r) => rows.push({ kind: "Hardening", id: r.run_id, when: r.requested_utc, text: r.status !== "done" ? r.status : `candidate ${r.candidate_id} · train ${r.train?.attack_survived}/${r.train?.attacks}, validation ${r.validation?.attack_survived}/${r.validation?.attacks}`, replay: r.status === "done" && r.candidate_id ? () => onCandidate(r.run_id, r.candidate_id!, r.focus ?? "m2-100N-500") : undefined }));
  if (!rows.length) return <p className="muted small">No runs on this server yet.</p>;
  return (
    <table>
      <thead><tr><th>Kind</th><th>Run</th><th>Result</th><th /></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}><td>{r.kind}</td><td>{r.id}</td><td>{r.text}</td><td>{r.replay && <button onClick={r.replay} disabled={busy}>Replay</button>}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
