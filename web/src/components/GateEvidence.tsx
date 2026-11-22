// Regression gate (M7) evidence for the Results mode: the pre-registered gate-v1 decision, the
// criteria for each registered candidate, per-episode holdout and nominal tables, the gauntlet with
// every individual regression, and the policy registry. All numbers come from the committed gate
// result (precomputed); replays re-simulate a gate episode and check it against that result.
import { useState } from "react";
import { COLORS, fmt, pct, type GateCandidate, type GateInfo, type GateRate, type GateRow, type HardeningEpisode, type Outcome } from "../api";
import { Evidence } from "./Workspace";

export type GateReplay = (candidate: string, episode: string, ghost?: "baseline" | "matched") => void;

function change(b: Outcome, c: Outcome): { text: string; bad: boolean } {
  if (b === "fell" && c === "survived") return { text: "recovered", bad: false };
  if (b === "survived" && c !== "survived") return { text: "REGRESSED", bad: true };
  if (b === c) return { text: b === "fell" ? "both fell" : "both survived", bad: false };
  return { text: `${b} → ${c}`, bad: c !== "survived" };
}

function rate(r: GateRate): string {
  const ci = r.wilson_95 ? ` [${pct(r.wilson_95[0])}, ${pct(r.wilson_95[1])}]` : "";
  return `${r.fell}/${r.valid} (${pct(r.fall_rate)}${ci})`;
}

function pValue(p: number): string {
  return p < 0.001 ? p.toExponential(1) : p.toFixed(4);
}

export function episodeText(e: HardeningEpisode): string {
  const parts: string[] = [];
  if (e.push) parts.push(`${e.push.force_n.toFixed(0)} N at ${e.push.direction_deg.toFixed(0)}°, ${e.push.start_s.toFixed(3)} s for ${e.push.duration_s.toFixed(2)} s`);
  else parts.push(`command (${e.command.map((v) => v.toFixed(1)).join(", ")})`);
  if (e.ground_friction_mu !== null) parts.push(`μ ${e.ground_friction_mu}`);
  if (e.payload_kg > 0) parts.push(`${e.payload_kg} kg payload`);
  return parts.join(" · ");
}

function OutcomeCell({ row }: { row: GateRow }) {
  return (
    <span>
      <span className="dot" style={{ background: COLORS[row.outcome] }} /> {row.outcome}
      {row.outcome === "fell" && row.time_after_push_s !== null ? ` +${fmt(row.time_after_push_s, 2)} s` : row.outcome === "fell" ? ` ${fmt(row.fall_time_s, 2)} s` : ""}
    </span>
  );
}

function Pass({ ok }: { ok: boolean }) {
  return <span className={ok ? "" : "bad"}>{ok ? "pass" : "FAIL"}</span>;
}

export function gateHeadline(gate: GateInfo): string | null {
  const primary = gate.candidates?.find((c) => c.primary);
  if (!gate.available || !primary || !gate.decision) return null;
  const h = primary.G2_holdout;
  const g = primary.G3_regression;
  return `${gate.gate!.name}: ${gate.decision.statement} Locked holdout ${h.baseline.fell}/${h.baseline.valid} → ${h.candidate.fell}/${h.candidate.valid} falls (one-sided McNemar p ${pValue(h.mcnemar_one_sided_p)}); regression gauntlet ${g.baseline.fell} → ${g.candidate.fell}/${g.candidate.valid} falls with ${g.regressed} individual regressions.`;
}

export function GateEvidence({ gate, onReplay, busy }: { gate: GateInfo; onReplay: GateReplay; busy: boolean }) {
  const candidates = gate.candidates ?? [];
  const primary = candidates.find((c) => c.primary);
  const [selected, setSelected] = useState<string | undefined>(primary?.candidate_id);
  const reg = gate.registry;
  if (!gate.available || !primary || !gate.decision) {
    return (
      <Evidence title="Regression gate (M7)" summary="no gate result on this server" open>
        <p className="small">No gate result is available. The registry points at {reg.active ?? "nothing"}; {gate.semantics}</p>
      </Evidence>
    );
  }
  const cand = candidates.find((c) => c.candidate_id === selected) ?? primary;
  const def = gate.criteria_definition!;
  return (
    <>
      <Evidence title="Regression gate (M7) · decision" summary={`${gate.decision.verdict === "promote" ? "promoted" : "rejected"} ${gate.decision.candidate_id} · active ${reg.active}`} open>
        <p className="finding small">
          <strong>{gate.decision.statement}</strong> {gate.decision.note}
        </p>
        <div className="two-col">
          <div className="small">
            <h3>Policy registry</h3>
            <dl className="facts">
              <dt>Active pointer</dt><dd>{reg.active} ({reg.path}/active.json)</dd>
              <dt>History</dt><dd>{reg.history.map((h) => `${h.version} (${h.reason})`).join(" → ")}</dd>
              {reg.versions.map((v) => (
                <span key={v.version} style={{ display: "contents" }}>
                  <dt>{v.version}</dt>
                  <dd>{v.reflex ? `frozen ${v.policy.path} + reflex ${v.reflex.record}` : `frozen ${v.policy.path}, no reflex`} · policy sha {v.policy.sha256.slice(0, 12)}</dd>
                </span>
              ))}
              <dt>Rejections</dt><dd>{reg.rejected.length ? reg.rejected.map((r) => r.candidate_id).join(", ") : "none recorded"}</dd>
            </dl>
          </div>
          <div className="small muted">
            <h3>What this means</h3>
            <p>{gate.semantics}</p>
            <p>
              The gate was written down and committed before it ran ({gate.config_path}, sha {gate.gate!.sha256.slice(0, 12)}), evaluated once, and its primary
              candidate was fixed in advance from M6 evidence only. 10 locked attacks is a small sample: the result supports this candidate on these
              conditions, not a general robustness claim.
            </p>
          </div>
        </div>
        <div className="row wrap">
          <span className="group-label">Show criteria for</span>
          <select value={cand.candidate_id} onChange={(e) => setSelected(e.target.value)} aria-label="Gate candidate">
            {candidates.map((c) => (
              <option key={c.candidate_id} value={c.candidate_id}>
                {c.candidate_id} · seed {c.seed} · {c.primary ? "primary (eligible)" : "not eligible"} · {c.passes_all ? "passed G1–G4" : `failed ${c.failed_criteria.join(", ")}`}
              </option>
            ))}
          </select>
        </div>
        <CriteriaTable cand={cand} def={def} />
      </Evidence>
      <HoldoutTable gate={gate} cand={cand} onReplay={onReplay} busy={busy} />
      <GauntletTable gate={gate} cand={cand} onReplay={onReplay} busy={busy} />
      <NominalTable gate={gate} cand={cand} onReplay={onReplay} busy={busy} />
      <Evidence title="All registered candidates" summary={`${candidates.length} M6 candidates · only the primary is eligible`}>
        <table>
          <thead><tr><th>Candidate</th><th>Seed</th><th>G1</th><th>G2</th><th>G3</th><th>G4</th><th>Holdout falls</th><th>Gauntlet falls</th><th>Regressions</th><th>Known failures still failing</th><th>Gate status</th></tr></thead>
          <tbody>
            {candidates.map((c) => (
              <tr key={c.candidate_id}>
                <td>{c.candidate_id}{c.primary ? " (primary)" : ""}</td><td>{c.seed}</td>
                <td><Pass ok={c.G1_nominal.pass} /></td><td><Pass ok={c.G2_holdout.pass} /></td><td><Pass ok={c.G3_regression.pass} /></td><td><Pass ok={c.G4_numerical.pass} /></td>
                <td>{c.G2_holdout.candidate.fell}/{c.G2_holdout.candidate.valid}</td>
                <td>{c.G3_regression.candidate.fell}/{c.G3_regression.candidate.valid}</td>
                <td className={c.G3_regression.regressed ? "bad" : ""}>{c.G3_regression.regressed}</td>
                <td>{c.known_failures.still_failing.length}/{c.known_failures.candidate.valid}</td>
                <td>{c.gate_status.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="xsmall muted">Baseline: holdout {primary.G2_holdout.baseline.fell}/{primary.G2_holdout.baseline.valid}, gauntlet {primary.G3_regression.baseline.fell}/{primary.G3_regression.baseline.valid}, known failures {primary.known_failures.baseline.fell}/{primary.known_failures.baseline.valid} fell. Non-primary candidates are reported so the holdout is not used to pick a winner; they cannot be promoted by gate-v1.</p>
      </Evidence>
      <Evidence title="Gate provenance" summary={`results ${gate.results_sha256!.slice(0, 12)}`}>
        <dl className="facts small">
          <dt>Origin</dt><dd>precomputed: {gate.preset!.title} ({gate.preset!.id} v{gate.preset!.version})</dd>
          <dt>Gate config</dt><dd>{gate.config_path} · sha {gate.gate!.sha256.slice(0, 16)} · locked set sha {gate.gate!.locked_sha256.slice(0, 12)}</dd>
          <dt>Suites</dt><dd>{Object.entries(gate.suites!).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>
          <dt>Statistics</dt><dd>{gate.statistics?.paired_test}; {gate.statistics?.interval}</dd>
          <dt>Rollouts</dt><dd>{gate.accounting!.rollouts} for {gate.accounting!.controllers} controllers in {fmt(gate.timing!.wall_s, 1)} s on {gate.timing!.workers} CPU workers</dd>
          <dt>Commit</dt><dd>{gate.provenance!.git.commit.slice(0, 7)}{gate.provenance!.git.dirty ? " (dirty)" : " (clean)"} · {gate.created_utc}</dd>
          <dt>Results</dt><dd>{gate.results_sha256}</dd>
          <dt>Reproduce</dt><dd><code>{gate.reproduce}</code> (recomputes and compares; a second evaluation is refused)</dd>
        </dl>
      </Evidence>
    </>
  );
}

function CriteriaTable({ cand, def }: { cand: GateCandidate; def: NonNullable<GateInfo["criteria_definition"]> }) {
  const g1 = cand.G1_nominal;
  const g2 = cand.G2_holdout;
  const g3 = cand.G3_regression;
  const worst = [...g3.conditions].sort((a, b) => b.increase - a.increase)[0];
  return (
    <table className="cases">
      <thead><tr><th>Criterion (pre-registered)</th><th>Measured</th><th>Result</th></tr></thead>
      <tbody>
        <tr>
          <td>G1 nominal walking: {def.G1_nominal.max_falls} falls, {def.G1_nominal.max_numerical} numerical failures, end displacement within max({def.G1_nominal.displacement_tolerance_m} m, {100 * def.G1_nominal.displacement_tolerance_rel}%) of the baseline on every episode</td>
          <td>{g1.falls} falls, {g1.numerical} numerical in {g1.episodes} episodes{g1.failures.length ? `; failing: ${g1.failures.join(", ")}` : "; all within tolerance"}</td>
          <td><Pass ok={g1.pass} /></td>
        </tr>
        <tr>
          <td>G2 locked holdout: fewer falls than the baseline; fall rate ≤ max({pct(def.G2_holdout.max_fall_rate_floor)}, baseline − {100 * def.G2_holdout.required_rate_drop} points); exact one-sided McNemar p ≤ {def.G2_holdout.mcnemar_one_sided_alpha}</td>
          <td>baseline {rate(g2.baseline)} → candidate {rate(g2.candidate)}; cap {pct(g2.rate_cap)}; {g2.recovered.length} recovered, {g2.regressed.length} regressed, p = {pValue(g2.mcnemar_one_sided_p)}</td>
          <td><Pass ok={g2.pass} /></td>
        </tr>
        <tr>
          <td>G3 regression gauntlet: fall rate ≤ baseline + {100 * def.G3_regression.max_fall_rate_increase_abs} points overall{def.G3_regression.per_condition ? " and in every condition" : ""}</td>
          <td>overall {rate(g3.baseline)} → {rate(g3.candidate)}; largest condition increase {worst.id} {(100 * worst.increase).toFixed(2)} points; {g3.regressed} individual regressions, {g3.recovered} recoveries</td>
          <td><Pass ok={g3.pass} /></td>
        </tr>
        <tr>
          <td>G4 numerical: ≤ {def.G4_numerical.max_numerical} numerical failures in any suite</td>
          <td>{cand.G4_numerical.candidate_numerical}</td>
          <td><Pass ok={cand.G4_numerical.pass} /></td>
        </tr>
        <tr>
          <td className="muted">Known M4 failures (reported, not a criterion)</td>
          <td>baseline {rate(cand.known_failures.baseline)} → candidate {rate(cand.known_failures.candidate)}; still failing: {cand.known_failures.still_failing.join(", ") || "none"}</td>
          <td className="muted">—</td>
        </tr>
      </tbody>
    </table>
  );
}

function short(id: string) {
  return id.slice(5, 13);
}

function HoldoutTable({ gate, cand, onReplay, busy }: { gate: GateInfo; cand: GateCandidate; onReplay: GateReplay; busy: boolean }) {
  const rows = gate.holdout ?? [];
  const h = cand.G2_holdout;
  return (
    <Evidence title={`Locked holdout · ${rows.length} attacks never used for tuning`} summary={`${h.baseline.fell} → ${h.candidate.fell} falls · ${cand.candidate_id}`} open>
      <table className="cases">
        <thead><tr><th>Episode</th><th>Shove and conditions</th><th>Frozen baseline v0</th><th>Candidate {short(cand.candidate_id)}</th><th>Change</th><th>Reflex active</th><th /></tr></thead>
        <tbody>
          {rows.map((r) => {
            const c = r.candidates[cand.candidate_id];
            const ch = change(r.baseline.outcome, c.outcome);
            return (
              <tr key={r.episode.id}>
                <td>{r.episode.id}</td><td>{episodeText(r.episode)}</td>
                <td><OutcomeCell row={r.baseline} /></td><td><OutcomeCell row={c} /></td>
                <td className={ch.bad ? "bad" : ""}>{ch.text}</td>
                <td>{c.reflex_activation === null ? "—" : `${(100 * c.reflex_activation).toFixed(1)}%`}</td>
                <td><button onClick={() => onReplay(cand.candidate_id, r.episode.id)} disabled={busy}>Replay</button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="xsmall muted">Ten attacks with correlated timings and conditions; Wilson 95% intervals are wide. Replays re-simulate both controllers and compare outcome, fall time and trajectory digest with the committed gate result.</p>
    </Evidence>
  );
}

function GauntletTable({ gate, cand, onReplay, busy }: { gate: GateInfo; cand: GateCandidate; onReplay: GateReplay; busy: boolean }) {
  const g = cand.G3_regression;
  const span = gate.gauntlet!;
  return (
    <Evidence title={`Regression gauntlet · ${g.conditions.length} conditions × ${span.starts_substeps[1] - span.starts_substeps[0]} shove timings`} summary={`${g.baseline.fell} → ${g.candidate.fell} falls · ${g.regressed} individual regressions`}>
      <p className="small">{gate.suite_notes?.gauntlet}. Every shove points toward the robot's right for {(span.duration_substeps * 0.004).toFixed(2)} s, starting every 4 ms. A condition passes if its fall rate rises by at most {(100 * g.limit).toFixed(0)} points; the individual regressions below are listed whether or not the condition passed.</p>
      <table className="cases">
        <thead><tr><th>Condition</th><th>Baseline fell</th><th>Candidate fell</th><th>Change</th><th>Recovered</th><th>Regressed</th><th>McNemar p (improvement)</th><th>G3</th></tr></thead>
        <tbody>
          {g.conditions.map((c) => (
            <tr key={c.id}>
              <td>{c.id}</td><td>{rate(c.baseline)}</td><td>{rate(c.candidate)}</td>
              <td className={c.increase > 0 ? "bad" : ""}>{c.increase > 0 ? "+" : ""}{(100 * c.increase).toFixed(2)} pts</td>
              <td>{c.recovered}</td><td className={c.regressions.length ? "bad" : ""}>{c.regressions.length}</td>
              <td>{pValue(c.mcnemar_one_sided_p)}</td><td><Pass ok={c.pass} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {g.conditions.filter((c) => c.regressions.length).map((c) => (
        <div key={c.id} className="group">
          <div className="group-label">{c.id}: baseline survived, candidate fell ({c.regressions.length})</div>
          <div className="chip-row">
            {c.regressions.slice(0, 40).map((r) => (
              <button key={r.id} className="chip-btn" disabled={busy} onClick={() => onReplay(cand.candidate_id, r.id)}
                title={`${r.id} · candidate ${r.candidate.outcome}${r.candidate.time_after_push_s !== null ? ` +${fmt(r.candidate.time_after_push_s, 2)} s` : ""}`}>
                {r.start_s.toFixed(3)} s
              </button>
            ))}
            {c.regressions.length > 40 && <span className="xsmall muted">… and {c.regressions.length - 40} more</span>}
          </div>
        </div>
      ))}
    </Evidence>
  );
}

function NominalTable({ gate, cand, onReplay, busy }: { gate: GateInfo; cand: GateCandidate; onReplay: GateReplay; busy: boolean }) {
  const episodes = new Map((gate.nominal_episodes ?? []).map((e) => [e.id, e]));
  const g1 = cand.G1_nominal;
  return (
    <Evidence title={`Nominal walking · ${g1.episodes} episodes without a shove`} summary={`${g1.falls} falls · ${g1.failures.length} outside tolerance`}>
      <table className="cases">
        <thead><tr><th>Episode</th><th>Conditions</th><th>Baseline</th><th>Candidate</th><th>End displacement m (b → c)</th><th>Difference / tolerance</th><th>Reflex active</th><th /></tr></thead>
        <tbody>
          {g1.rows.map((r) => {
            const e = episodes.get(r.id);
            return (
              <tr key={r.id}>
                <td>{r.id}</td><td>{e ? `${episodeText(e)} · ${e.horizon_s ?? 5} s` : ""}</td>
                <td><span className="dot" style={{ background: COLORS[r.baseline] }} /> {r.baseline}</td>
                <td><span className="dot" style={{ background: COLORS[r.candidate] }} /> {r.candidate}</td>
                <td>({fmt(r.baseline_displacement_m[0], 2)}, {fmt(r.baseline_displacement_m[1], 2)}) → ({fmt(r.candidate_displacement_m[0], 2)}, {fmt(r.candidate_displacement_m[1], 2)})</td>
                <td className={r.within_tolerance ? "" : "bad"}>{fmt(r.displacement_difference_m, 3)} / {fmt(r.tolerance_m, 3)}</td>
                <td className={r.reflex_activation > 0.25 ? "warn-text" : ""}>{(100 * r.reflex_activation).toFixed(1)}%</td>
                <td><button onClick={() => onReplay(cand.candidate_id, r.id)} disabled={busy}>Replay</button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="xsmall muted">Displacement in the initial-heading frame (x forward, y left). Reflex activity above 25% of ticks on an unshoved walk is highlighted: the reflex then changes nominal gait even though the walk stays within tolerance.</p>
    </Evidence>
  );
}
