import { useCallback, useEffect, useRef, useState } from "react";
import { api, COLORS, fmt, pct, type LiveStress as LiveStressData, type Outcome, type RunStatus } from "../api";
import { Details, Panel } from "./Workspace";

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

  return (
    <>
      <Panel eyebrow={`Preset 2 · ${data.preset.title}`} title={`${cfg.attack.force_n} N shove, ${cfg.trials} randomly timed trials, run live`}>
        <p className="small">
          {cfg.attack.force_n} N toward the robot's right for {cfg.attack.duration_s} s at {cfg.command[0]} m/s. Each run draws {cfg.trials} start
          times by stratified sampling over [{cfg.attack.start.low_s.toFixed(3)}, {cfg.attack.start.high_s.toFixed(3)}) s from a seed and
          simulates a nominal and a shoved episode for each on the server.
        </p>
        <div className="row wrap">
          <input className="seed" placeholder="seed (blank = new random)" value={seedText} onChange={(e) => setSeedText(e.target.value)} disabled={running} />
          <button className="primary big" onClick={start} disabled={running}>
            {running ? `Running… ${status?.completed ?? 0}/${total}` : `▶ Run ${cfg.trials} Trials`}
          </button>
        </div>
        {running && <div className="progress"><div style={{ width: `${(100 * (status?.completed ?? 0)) / total}%` }} /></div>}
        {error && <p className="error">{error}</p>}
        {status?.status === "failed" && <p className="error">Run failed: {status.error}</p>}
        <p className="small">
          <span className="badge pre">Precomputed reference</span> {ref.physical_falls}/{ref.valid_trials} = {pct(ref.fall_rate)} of all 256 start times fell at {cfg.attack.force_n} N
          (exhaustive sweep, commit {data.reference.provenance.git.commit.slice(0, 7)}). A 16-trial run estimates this; individual runs vary.
        </p>
        {status && (
          <div className="card inset">
            <strong className="small">{summary ? "Executed now" : "Executing"} · run {status.run_id} · seed {status.seed}{status.seed_source ? ` (${status.seed_source})` : ""}</strong>
            <div className="chips compact">
              {chips.map((c) => (
                <button key={c.trial} className="chip" style={{ background: c.outcome ? COLORS[c.outcome] : COLORS.none }}
                  disabled={!summary || replaying !== null}
                  title={c.outcome ? `trial ${c.trial}: ${c.outcome}${c.fall !== null ? ` @ ${fmt(c.fall)} s` : ""}` : "pending"}
                  onClick={() => onReplay(status.run_id, c.trial)}>
                  <span>{Number.isNaN(c.start_s) ? "…" : `${c.start_s.toFixed(3)} s`}</span>
                  <small>{replaying === `${status.run_id}/${c.trial}` ? "loading" : c.outcome === "fell" ? `fell ${fmt(c.fall, 2)}` : c.outcome ?? "pending"}</small>
                </button>
              ))}
            </div>
            {att && summary && (
              <p className="small">
                <strong>{att.physical_falls} falls / {att.valid_trials} valid trials</strong> ({pct(att.fall_rate)})
                {att.wilson_95 && <> · Wilson 95% [{fmt(att.wilson_95[0])}, {fmt(att.wilson_95[1])}]</>}; numerical {att.numerical_failures}. Click a trial to replay it.
              </p>
            )}
          </div>
        )}
      </Panel>
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
