import { useState } from "react";
import { COLORS, fmt, pct, type CompactTrial, type HiddenPhase as HiddenPhaseData } from "../api";
import { Details, Panel } from "./Workspace";

interface Props {
  data: HiddenPhaseData;
  onReplay: (substep: number) => void;
  replaying: number | null;
}

export function HiddenPhase({ data, onReplay, replaying }: Props) {
  const [selected, setSelected] = useState<number>(data.verified_failure_substep);
  const att = data.aggregates.attacked;
  const trial = data.trials.find((t) => t.start_substep === selected)!;
  const finding = data.stance_finding;
  const [low, high] = data.sampling.interval_s;
  const cfg = data.config;

  const failing = data.trials.filter((t) => t.outcome === "fell");
  return (
    <>
      <Panel eyebrow={`Preset 1 · ${data.preset.title}`} title={`${att.physical_falls} / ${att.valid_trials} shove timings made the robot fall (${pct(att.fall_rate)})`}>
        <p className="small">
          {cfg.attack.force_n} N toward the robot's right for {cfg.attack.duration_s} s while walking at {cfg.command[0]} m/s, started at each of
          the {data.trials.length} 4 ms steps in [{low.toFixed(3)}, {high.toFixed(3)}) s. Exact frequency over that timing distribution in this
          deterministic simulation; numerical failures {att.numerical_failures}.
        </p>
        <span className="badge pre">Precomputed · M2 sweep · commit {data.provenance.git.commit.slice(0, 7)}</span>
        <button className="primary big wide" onClick={() => { setSelected(data.verified_failure_substep); onReplay(data.verified_failure_substep); }} disabled={replaying !== null}>
          {replaying === data.verified_failure_substep ? "Simulating…" : "▶ Replay a failure"}
          <span className="small">shove at {fmt(data.verified_failure_substep * 0.004)} s</span>
        </button>
        <div className="small"><strong>All {failing.length} failing timings</strong> (click to replay):</div>
        <div className="chips compact">
          {failing.map((t) => (
            <button key={t.start_substep} className="chip" style={{ background: COLORS.fell }} disabled={replaying !== null}
              onClick={() => { setSelected(t.start_substep); onReplay(t.start_substep); }} title={`${t.stance} · fell ${fmt(t.fall_time_s)} s`}>
              <span>{t.start_s.toFixed(3)} s</span>
            </button>
          ))}
        </div>
        <div className="card inset">
          <h3>Selected timing: {fmt(trial.start_s)} s</h3>
          <TrialFacts trial={trial} />
          <button onClick={() => onReplay(trial.start_substep)} disabled={replaying !== null}>
            {replaying === trial.start_substep ? "Simulating…" : "Replay this timing (re-simulate now)"}
          </button>
        </div>
      </Panel>
      <Details>
        <SweepChart trials={data.trials} selected={selected} onSelect={setSelected} />
        {finding && (
          <p className="finding">
            All {att.physical_falls} falls happened while the front-right and rear-left feet were in stance (FR+RL diagonal), but only{" "}
            <strong>{finding.fell} of the {finding.tested}</strong> FR+RL timings fell. Stance alone does not predict a fall; the
            outcome changes between neighbouring 4 ms start times.
          </p>
        )}
        <div className="two-col">
          <div className="card">
            <h3>Outcome by stance at the shove</h3>
            <table>
              <thead><tr><th>Feet down</th><th>Tested</th><th>Fell</th></tr></thead>
              <tbody>
                {data.stance_table.map((r) => (
                  <tr key={r.stance}><td>{r.stance}</td><td>{r.tested}</td><td className={r.fell ? "bad" : ""}>{r.fell}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">Stance = foot-floor contact at the last 20 ms control tick before the shove, nominal run.</p>
          </div>
          <div className="card">
            <h3>About the replay</h3>
            <p className="muted small">Recorded outcomes are from the precomputed sweep (results {data.results_sha256.slice(0, 8)}). A replay re-runs that paired trial on the server and checks it against the recorded trajectory digests.</p>
          </div>
        </div>
        <details className="card">
          <summary>What this does and does not show</summary>
          <ul>{data.interpretation.map((line) => <li key={line}>{line}</li>)}</ul>
        </details>
      </Details>
    </>
  );
}

function TrialFacts({ trial }: { trial: CompactTrial }) {
  const events = Object.entries(trial.events).sort((a, b) => a[1] - b[1]);
  return (
    <dl className="facts">
      <dt>Outcome</dt>
      <dd><span className="dot" style={{ background: COLORS[trial.outcome] }} /> {trial.outcome === "fell" ? `fell (${trial.reason})` : trial.outcome}</dd>
      <dt>Stance at shove</dt><dd>{trial.stance}</dd>
      <dt>First fall</dt><dd>{trial.fall_time_s === null ? "—" : `${fmt(trial.fall_time_s)} s (+${fmt(trial.time_after_start_s)} s)`}</dd>
      <dt>Events</dt><dd>{events.length ? events.map(([n, t]) => `${n} ${fmt(t)} s`).join(", ") : "none"}</dd>
      <dt>First non-foot body</dt><dd>{trial.first_nonfoot_body ?? "—"}</dd>
      <dt>Progress</dt><dd>{fmt(trial.forward_progress_m, 2)} m shoved vs {fmt(trial.nominal_progress_m, 2)} m nominal</dd>
    </dl>
  );
}

function SweepChart({ trials, selected, onSelect }: { trials: CompactTrial[]; selected: number; onSelect: (k: number) => void }) {
  const w = 4;
  const width = trials.length * w;
  const stanceColor = (s: string) => (s === "FR+RL" ? "#e0a030" : s === "FL+RR" ? "#9db7d9" : "#dde1e6");
  return (
    <div className="card sweep">
      <div className="sweep-head">
        <strong>All {trials.length} tested shove start times</strong>
        <span className="legend">
          <span className="dot" style={{ background: COLORS.survived }} /> survived <span className="dot" style={{ background: COLORS.fell }} /> fell
          <span className="dot" style={{ background: COLORS.numerical }} /> numerical <span className="swatch" style={{ background: "#e0a030" }} /> FR+RL stance
          <span className="swatch" style={{ background: "#9db7d9" }} /> FL+RR stance
        </span>
      </div>
      <svg viewBox={`0 0 ${width} 46`} preserveAspectRatio="none" className="sweep-svg">
        {trials.map((t, i) => (
          <g key={t.start_substep} onClick={() => onSelect(t.start_substep)} style={{ cursor: "pointer" }}>
            <title>{`${t.start_s.toFixed(3)} s · ${t.stance} · ${t.outcome}${t.fall_time_s !== null ? ` @ ${t.fall_time_s.toFixed(3)} s` : ""}`}</title>
            <rect x={i * w} y={t.outcome === "survived" ? 18 : 0} width={w - 0.6} height={t.outcome === "survived" ? 14 : 32} fill={COLORS[t.outcome]} />
            <rect x={i * w} y={36} width={w} height={8} fill={stanceColor(t.stance)} />
            {t.start_substep === selected && <rect x={i * w - 0.5} y={0} width={w + 0.4} height={46} fill="none" stroke="#111" strokeWidth={1} />}
          </g>
        ))}
      </svg>
      <div className="axis"><span>{trials[0].start_s.toFixed(3)} s</span><span>shove start time (4 ms steps)</span><span>{trials[trials.length - 1].start_s.toFixed(3)} s</span></div>
    </div>
  );
}
