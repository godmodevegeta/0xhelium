import { useCallback, useEffect, useRef, useState } from "react";
import { api, COLORS, fmt, pct, type LiveStress as LiveStressData, type Outcome, type RunStatus } from "../api";
import { Advanced, Details, Evidence, EvidenceLink, ModeIntro, useJobReport } from "./Workspace";

interface Props {
  data: LiveStressData;
  onReplay: (runId: string, trial: number) => void;
  onRefresh: () => void;
  replaying: string | null;
}

export function LiveStress({ data, onReplay, onRefresh, replaying }: Props) {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [seedText, setSeedText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const autoReplay = useRef<string | null>(null); // replay a fall from a run started here
  const cfg = data.config;
  const ref = data.reference.aggregates.attacked;

  const poll = useCallback(
    (runId: string) => {
      const tick = async () => {
        try {
          const s = await api.run(runId);
          setStatus(s);
          if (s.status === "queued" || s.status === "running") {
            timer.current = window.setTimeout(tick, 300);
          } else {
            onRefresh();
            if (autoReplay.current === runId && s.summary) {
              autoReplay.current = null;
              const t = s.summary.trials.find((x) => x.outcome === "fell") ?? s.summary.trials[0];
              if (t) onReplay(runId, t.trial);
            }
          }
        } catch (e) {
          setError(String(e));
        }
      };
      void tick();
    },
    [onRefresh, onReplay],
  );

  useEffect(() => {
    const runId = new URLSearchParams(window.location.search).get("run"); // deep link to a run
    if (runId && /^ls-\d{8}T\d{6}Z-\d{1,20}$/.test(runId)) poll(runId);
    return () => { if (timer.current) window.clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async () => {
    setError(null);
    const seed = seedText.trim() === "" ? null : Number(seedText);
    if (seed !== null && (!Number.isInteger(seed) || seed < 0)) {
      setError("Seed must be a non-negative integer, or blank for a fresh random seed.");
      return;
    }
    try {
      const started = await api.startRun(seed);
      autoReplay.current = started.run_id;
      setStatus({ run_id: started.run_id, status: "queued", seed: started.seed, seed_source: "", requested_utc: "", reproduce: "", completed: 0, total: started.total });
      poll(started.run_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const running = status?.status === "queued" || status?.status === "running";
  const summary = status?.summary;
  const total = status?.total ?? summary?.trials.length ?? data.trials;
  const chips: { trial: number; start_s: number; outcome: Outcome | null; reason: string | null; fall: number | null }[] =
    summary
      ? summary.trials.map((t) => ({ trial: t.trial, start_s: t.start_s, outcome: t.outcome, reason: t.reason, fall: t.fall_time_s }))
      : Array.from({ length: total }, (_, i) => {
          const t = status?.trials?.[i];
          return { trial: i, start_s: t?.start_s ?? NaN, outcome: t?.attacked.outcome ?? null, reason: t?.attacked.reason ?? null, fall: t?.attacked.fall_time_s ?? null };
        });
  const att = summary?.aggregates.attacked;

  useJobReport(running, `Stress test ${status?.completed ?? 0}/${total}`);
  return (
    <>
      <div className="mode">
        <ModeIntro eyebrow="Stress test · verified preset" title={`${cfg.trials} randomly timed ${cfg.attack.force_n} N shoves`}>
          <p>Each trial pairs a nominal walk with the same walk shoved toward the robot's right for {cfg.attack.duration_s} s; only the shove start varies.</p>
        </ModeIntro>
        <div className="summary-box">
          <div><span className="muted">Preset</span> <strong>{data.preset.title}</strong> v{data.preset.version} · {cfg.command[0]} m/s · start times stratified over [{cfg.attack.start.low_s.toFixed(3)}, {cfg.attack.start.high_s.toFixed(3)}) s</div>
          <div><span className="badge pre">Precomputed reference</span> {ref.physical_falls}/{ref.valid_trials} = {pct(ref.fall_rate)} of all 256 timings fell</div>
        </div>
        <div className="cta">
          <button className="primary big wide" onClick={start} disabled={running}>
            {running ? `Running on the server… ${status?.completed ?? 0}/${total}` : `Run ${cfg.trials} trials`}
          </button>
          <div className="inline-status">
            {running && <div className="progress"><div style={{ width: `${(100 * (status?.completed ?? 0)) / total}%` }} /></div>}
            {error && <span className="error">{error}</span>}
            {status?.status === "failed" && <span className="error">Run failed: {status.error}</span>}
            {!status && <span>A new seed is drawn for each run unless you set one under Advanced.</span>}
          </div>
        </div>
        {status && (
          <div className="group">
            <div className="group-label">{summary ? "Executed now" : "Executing"} · run {status.run_id} · seed {status.seed}</div>
            <div className="chips" role="group" aria-label="Trials">
              {chips.map((c) => (
                <button key={c.trial} className="chip" style={{ background: c.outcome ? COLORS[c.outcome] : COLORS.none, color: c.outcome ? undefined : "var(--muted)" }}
                  disabled={!summary || replaying !== null}
                  aria-label={`Replay trial ${c.trial}${c.outcome ? `, ${c.outcome}` : ", pending"}`}
                  title={c.outcome ? `trial ${c.trial}: ${c.outcome}${c.fall !== null ? ` @ ${fmt(c.fall)} s` : ""}` : "pending"}
                  onClick={() => onReplay(status.run_id, c.trial)}>
                  <span>{Number.isNaN(c.start_s) ? "…" : `${c.start_s.toFixed(3)} s`}</span>
                  <small>{replaying === `${status.run_id}/${c.trial}` ? "loading" : c.outcome === "fell" ? `fell ${fmt(c.fall, 2)}` : c.outcome ?? "pending"}</small>
                </button>
              ))}
            </div>
            {att && summary && (
              <div className="small">
                <strong>{att.physical_falls} falls / {att.valid_trials} valid trials</strong> ({pct(att.fall_rate)})
                {att.wilson_95 && <> · Wilson 95% [{fmt(att.wilson_95[0])}, {fmt(att.wilson_95[1])}]</>}; numerical {att.numerical_failures}. Select a trial to replay it.
              </div>
            )}
          </div>
        )}
        <Advanced summary="Advanced · fixed seed">
          <label className="field wide">Seed (blank draws a new one)
            <input className="seed" placeholder="seed" value={seedText} onChange={(e) => setSeedText(e.target.value)} disabled={running} />
          </label>
        </Advanced>
        <EvidenceLink />
      </div>
      <Details>
        {status && summary && att && (
          <div className="card">
            <div className="sweep-head">
              <strong>Stress Test run {status.run_id}</strong>
              <span className="badge live">new simulation · commit {summary.provenance.git.commit.slice(0, 7)}{summary.provenance.git.dirty ? " (dirty tree)" : ""} · {fmt(status.timing?.wall_s, 2)} s</span>
            </div>
            <p className="result-line">
              <strong>{att.physical_falls} falls / {att.valid_trials} valid trials</strong> ({pct(att.fall_rate)}); survived {att.survived}; numerical failures {att.numerical_failures}
              {" "}· nominal: {summary.aggregates.nominal.physical_falls} falls, {summary.aggregates.nominal.distinct_trajectories} distinct trajectory
            </p>
            <p className="muted small">
              Estimate over the stated timing distribution only. Reproduce: <code>{status.reproduce}</code> · results {summary.results_sha256.slice(0, 12)}
            </p>
          </div>
        )}
        <Evidence title="Precomputed reference" summary={`${ref.physical_falls}/${ref.valid_trials} timings fell at ${cfg.attack.force_n} N (exhaustive sweep)`}>
          <p className="small">
            Every one of the 256 start times in [{cfg.attack.start.low_s.toFixed(3)}, {cfg.attack.start.high_s.toFixed(3)}) s was simulated once at {cfg.attack.force_n} N:
            {" "}{ref.physical_falls}/{ref.valid_trials} = {pct(ref.fall_rate)} fell; numerical failures {ref.numerical_failures}. Commit {data.reference.provenance.git.commit.slice(0, 7)}.
            A 16-trial run estimates this; individual runs vary.
          </p>
        </Evidence>
        {data.runs.length > 0 && (
          <details className="card">
            <summary>Previous Stress Test runs ({data.runs.length})</summary>
            <table>
              <thead><tr><th>Run</th><th>Seed</th><th>Status</th><th>Falls / valid</th><th /></tr></thead>
              <tbody>
                {data.runs.map((r) => (
                  <tr key={r.run_id}>
                    <td>{r.run_id}</td><td>{r.seed}</td><td>{r.status}</td>
                    <td>{r.status === "done" ? `${r.physical_falls}/${r.valid_trials}${r.numerical_failures ? ` (+${r.numerical_failures} numerical)` : ""}` : "—"}</td>
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
