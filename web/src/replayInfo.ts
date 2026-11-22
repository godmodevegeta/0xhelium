// Short, honest names and provenance text for a replay, shared by the stage and the summary strip.
import type { Replay } from "./api";

export interface RobotNames {
  main: string;
  ghost: string;
}

export function robotNames(r: Replay): RobotNames {
  if (r.mode === "hardening" || r.mode === "gate") {
    // "Promoted" only when the gate's registry version exists for this exact candidate id.
    const promoted = r.gate_status?.status === "promoted" && r.gate_status.version;
    return {
      main: promoted ? `Promoted ${r.gate_status!.version!.split("-")[0]} · policy + reflex` : "Candidate · frozen policy + reflex",
      ghost: r.ghost?.kind === "matched" ? "Frozen baseline v0 · no shove" : "Frozen baseline v0",
    };
  }
  if (r.mode === "single" || r.mode === "suite") {
    return {
      main: r.push ? "Scenario (shoved)" : "Scenario",
      ghost: r.ghost?.kind === "original" ? "Original nominal" : "Matched baseline",
    };
  }
  return { main: r.push ? "Shoved robot" : "Robot", ghost: "Nominal · no shove" };
}

export type Tone = "ok" | "warn" | "neutral";

export function sourceInfo(r: Replay, sample: boolean): { text: string; tone: Tone } {
  if (r.source === "recorded") {
    const where = r.mode ? `scenario run ${r.run_id}${r.mode === "suite" ? ` · trial ${r.trial}` : ""}` : `stress run ${r.run_id} · trial ${r.trial}`;
    return { text: `Recorded server result · ${where}`, tone: "neutral" };
  }
  const v = r.verification;
  const prefix = sample ? "Sample · precomputed, re-simulated" : "Re-simulated on the server";
  if (!v) return { text: prefix, tone: "neutral" };
  const what = v.what ?? (r.which ? "discovery record" : "M2 sweep");
  return v.all_match
    ? { text: `${prefix} · matches ${what}`, tone: "ok" }
    : { text: `${prefix} · DOES NOT match the recorded ${what}`, tone: "warn" };
}

export function outcomeText(c: Replay["cases"]["attacked"]["outcome"]): string {
  if (c.outcome === "fell") return `fell at ${c.fall_time_s?.toFixed(2)} s`;
  if (c.outcome === "numerical") return "numerical failure";
  return "survived";
}

export function directionWord(world: [number, number, number]): string {
  const [x, y] = world;
  const deg = ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
  const names = ["forward", "forward-left", "left", "back-left", "backward", "back-right", "right", "forward-right"];
  return `toward ${names[Math.round(deg / 45) % 8]} (${deg.toFixed(0)}°)`;
}

const n = (v: number) => v.toLocaleString("en-US");

// What the server computed to produce this replay. Only measured counts; a replay loaded from a
// saved run says so instead.
export function computeText(r: Replay): string {
  const c = r.compute;
  if (!c || c.episodes === 0) return "Recorded by the server's MuJoCo run · replayed here";
  return `Server MuJoCo: ${c.episodes} episodes · ${n(c.physics_steps)} physics steps · ${n(c.control_ticks)} policy steps · ${c.wall_s.toFixed(2)} s${c.cached ? " (cached)" : ""} · replayed here`;
}

export function evidenceText(r: Replay): string | null {
  const e = r.evidence_compute;
  return e ? `from the ${e.label}: ${n(e.rollouts)} rollouts, ${e.wall_s.toFixed(0)} s on ${e.workers} CPU workers` : null;
}
