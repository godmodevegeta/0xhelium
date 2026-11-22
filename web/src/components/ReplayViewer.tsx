// Three.js playback of recorded MuJoCo poses. No physics runs here: every frame is a pose the
// server computed with mj_kinematics from a recorded qpos row.
//
// One renderer, scene and camera live as long as the robot model (they are not rebuilt when the
// replay changes); each new replay only swaps the two robot mesh groups, so the camera offset and
// the playback speed carry over between replays and modes.
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { COLORS, decodeFloat32, decodeInt32, fmt, type ModelPayload, type Replay } from "../api";

interface Props {
  model: ModelPayload;
  replay: Replay;
  title: string;
  sample?: boolean;
  busy?: string | null; // e.g. "Simulating on the server…" while the next replay loads
}

interface Robot {
  group: THREE.Group;
  meshes: THREE.Mesh[];
  pos: Float32Array;
  quat: Float32Array;
  frames: number;
  payload: { mesh: THREE.Mesh; pos: Float32Array; quat: Float32Array } | null;
}

interface Stage {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  arrow: THREE.ArrowHelper;
  materials: { ghost: THREE.Material; main: THREE.Material; ghostBox: THREE.Material; box: THREE.Material };
}

const EVENT_LABEL: Record<string, string> = {
  tilt: "tilt > 45°",
  nonfoot_contact: "non-foot contact",
  low_height: "trunk < 0.18 m",
  nonfinite_state: "non-finite state",
  mujoco_warning: "MuJoCo warning",
};

function buildGeometries(model: ModelPayload): Record<string, THREE.BufferGeometry> {
  const out: Record<string, THREE.BufferGeometry> = {};
  for (const [name, mesh] of Object.entries(model.meshes)) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(decodeFloat32(mesh.vertices), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(decodeInt32(mesh.faces)), 1));
    geometry.computeVertexNormals();
    out[name] = geometry;
  }
  return out;
}

function outcomeAt(c: Replay["cases"]["attacked"]["outcome"], t: number): { text: string; color: string } {
  if (c.outcome === "fell" && c.fall_time_s !== null && t >= c.fall_time_s - 1e-9) {
    return { text: `FELL (${EVENT_LABEL[c.reason ?? ""] ?? c.reason}) at ${fmt(c.fall_time_s, 2)} s`, color: COLORS.fell };
  }
  if (c.outcome === "numerical" && c.numerical_time_s !== null && t >= c.numerical_time_s - 1e-9) {
    return { text: "numerical failure (not a fall)", color: COLORS.numerical };
  }
  if (c.outcome === "survived") return { text: "upright · survives the horizon", color: COLORS.survived };
  return { text: "upright", color: COLORS.survived };
}

let frameParamUsed = false;

export function ReplayViewer({ model, replay, title, sample, busy }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const robotsRef = useRef<Robot[]>([]);
  const replayRef = useRef<Replay>(replay);
  const floorRef = useRef<THREE.Mesh | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0.5);
  const frameRef = useRef(0);
  // Continuous playhead in seconds; keeping the fractional time lets slow speeds advance.
  const timeRef = useRef(0);
  const playingRef = useRef(true);
  const speedRef = useRef(0.5);
  const geometries = useMemo(() => buildGeometries(model), [model]);
  const duration = (replay.frames - 1) / replay.fps;
  const durationRef = useRef(duration);

  playingRef.current = playing;
  speedRef.current = speed;
  replayRef.current = replay;
  durationRef.current = duration;

  // Renderer, scene, camera and loop: once per model.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f4f6f9");
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / Math.max(mount.clientHeight, 1), 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(-1.35, 0.9, 0.62); // behind and to the left: a sideways shove reads left-to-right
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.25);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, -3, 5);
    scene.add(sun);
    const grid = new THREE.GridHelper(40, 80, 0x9aa3ad, 0xd5dae0);
    grid.rotation.x = Math.PI / 2;
    grid.position.x = 15;
    scene.add(grid);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), 0.5, 0xe07a10, 0.18, 0.12);
    arrow.visible = false;
    scene.add(arrow);
    const materials = {
      ghost: new THREE.MeshStandardMaterial({ color: "#5b8bd6", transparent: true, opacity: 0.35, depthWrite: false }),
      main: new THREE.MeshStandardMaterial({ color: "#3a3f47", metalness: 0.1, roughness: 0.6 }),
      ghostBox: new THREE.MeshStandardMaterial({ color: "#e08a1e", transparent: true, opacity: 0.35, depthWrite: false }),
      box: new THREE.MeshStandardMaterial({ color: "#e08a1e", roughness: 0.5 }),
    };
    stageRef.current = { renderer, scene, camera, controls, arrow, materials };

    const trunk = (robot: Robot, f: number) => {
      const G = replayRef.current.geom_count;
      const i = Math.min(f, robot.frames - 1) * G * 3; // geom 0 is the trunk
      return new THREE.Vector3(robot.pos[i], robot.pos[i + 1], robot.pos[i + 2]);
    };
    const pose = (robot: Robot, f: number) => {
      const G = replayRef.current.geom_count;
      const i = Math.min(f, robot.frames - 1);
      robot.meshes.forEach((mesh, j) => {
        const p = (i * G + j) * 3;
        const q = (i * G + j) * 4;
        mesh.position.set(robot.pos[p], robot.pos[p + 1], robot.pos[p + 2]);
        mesh.quaternion.set(robot.quat[q + 1], robot.quat[q + 2], robot.quat[q + 3], robot.quat[q]);
      });
      if (robot.payload) {
        const { mesh, pos, quat } = robot.payload;
        mesh.position.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        mesh.quaternion.set(quat[i * 4 + 1], quat[i * 4 + 2], quat[i * 4 + 3], quat[i * 4]);
      }
    };

    let last = performance.now();
    let raf = 0;
    let previousTarget: THREE.Vector3 | null = null;
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1); // a hidden tab must not jump the playhead
      last = now;
      const rp = replayRef.current;
      const robots = robotsRef.current;
      if (playingRef.current) {
        timeRef.current += dt * speedRef.current;
        if (timeRef.current > durationRef.current) timeRef.current = 0; // loop
        const f = Math.min(rp.frames - 1, Math.floor(timeRef.current * rp.fps + 1e-9));
        if (f !== frameRef.current) {
          frameRef.current = f;
          setFrame(f);
        }
      }
      const f = frameRef.current;
      if (robots.length === 2) {
        robots.forEach((r) => pose(r, f));
        const t = f / rp.fps;
        if (rp.push && t >= rp.push.start_s - 1e-9 && t < rp.push.end_s + 1e-9) {
          const at = trunk(robots[1], f);
          const [fx, fy] = rp.push.world_force_n;
          const away = new THREE.Vector3(-fx, -fy, 0).normalize().multiplyScalar(0.55);
          arrow.position.copy(at).add(away);
          arrow.visible = true;
        } else {
          arrow.visible = false;
        }
        const target = trunk(robots[1], f).add(trunk(robots[0], f)).multiplyScalar(0.5);
        target.z = 0.25;
        if (previousTarget === null) {
          // A new replay: keep the user's camera offset, move it to the new robots.
          const offset = camera.position.clone().sub(controls.target);
          camera.position.copy(target).add(offset);
        } else {
          camera.position.add(target.clone().sub(previousTarget));
        }
        controls.target.copy(target);
        previousTarget = target;
      }
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);
    (stageRef.current as Stage & { resetCamera?: () => void }).resetCamera = () => {
      previousTarget = null;
    };
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      Object.values(materials).forEach((m) => m.dispose());
      mount.removeChild(renderer.domElement);
      stageRef.current = null;
    };
  }, [model]);

  // Robots, floor cue and shove arrow: per replay.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const { scene, materials, arrow } = stage;
    const boxGeometries: THREE.BufferGeometry[] = [];
    const makeRobot = (key: "nominal" | "attacked", material: THREE.Material, boxMaterial: THREE.Material): Robot => {
      const group = new THREE.Group();
      const meshes = model.geoms.map((g) => {
        const mesh = new THREE.Mesh(geometries[g.mesh], material);
        group.add(mesh);
        return mesh;
      });
      const c = replay.cases[key];
      let payload: Robot["payload"] = null;
      if (c.payload_box) {
        // Pose from server kinematics of the welded payload geom; the browser only draws it.
        const [hx, hy, hz] = c.payload_box.half_size_m;
        const geometry = new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
        boxGeometries.push(geometry);
        const mesh = new THREE.Mesh(geometry, boxMaterial);
        group.add(mesh);
        payload = { mesh, pos: decodeFloat32(c.payload_box.pos), quat: decodeFloat32(c.payload_box.quat) };
      }
      scene.add(group);
      return { group, meshes, pos: decodeFloat32(c.pos), quat: decodeFloat32(c.quat), frames: c.frames, payload };
    };
    robotsRef.current = [makeRobot("nominal", materials.ghost, materials.ghostBox), makeRobot("attacked", materials.main, materials.box)];
    if (replay.push) {
      const [fx, fy, fz] = replay.push.world_force_n;
      arrow.setDirection(new THREE.Vector3(fx, fy, fz).normalize());
    }
    arrow.visible = false;
    const env = replay.environment;
    if (env && Math.abs(env.mu - env.nominal_mu) > 1e-9) {
      const color = env.mu < env.nominal_mu ? "#a9d8f0" : "#e2c9a0";
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color, roughness: env.mu < env.nominal_mu ? 0.15 : 0.95 }));
      floor.position.set(15, 0, -0.002);
      scene.add(floor);
      floorRef.current = floor;
    }
    (stage as Stage & { resetCamera?: () => void }).resetCamera?.();
    // Initial frame: ?frame=<n> deep link applies to the first replay only, else 0.
    const requested = frameParamUsed ? 0 : Number(new URLSearchParams(window.location.search).get("frame"));
    frameParamUsed = true;
    const start = Number.isInteger(requested) && requested > 0 ? Math.min(requested, replay.frames - 1) : 0;
    frameRef.current = start;
    timeRef.current = start / replay.fps;
    setFrame(start);
    setPlaying(start === 0);
    return () => {
      robotsRef.current.forEach((r) => scene.remove(r.group));
      robotsRef.current = [];
      boxGeometries.forEach((g) => g.dispose());
      if (floorRef.current) {
        scene.remove(floorRef.current);
        floorRef.current.geometry.dispose();
        (floorRef.current.material as THREE.Material).dispose();
        floorRef.current = null;
      }
    };
  }, [model, replay, geometries]);

  const seek = (f: number) => {
    frameRef.current = f;
    timeRef.current = f / replay.fps;
    setFrame(f);
  };
  const togglePlay = () => {
    // Play at the last frame restarts from the beginning instead of sitting on the end.
    if (!playing && frameRef.current >= replay.frames - 1) seek(0);
    setPlaying((p) => !p);
  };
  const fullscreen = () => {
    const el = rootRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen?.();
  };
  const att = replay.cases.attacked;
  const nom = replay.cases.nominal;
  const f = Math.min(frame, att.frames - 1);
  const t = frame / replay.fps;
  const tilt = (Math.acos(Math.max(-1, Math.min(1, att.up_z[f] ?? 1))) * 180) / Math.PI;
  const events = Object.entries(att.outcome.events).sort((a, b) => a[1] - b[1]);
  const pctOf = (time: number) => `${(100 * time) / duration}%`;
  const attackedLabel = replay.labels?.attacked ?? (replay.push ? "Shoved robot" : "Robot");
  const nominalLabel = replay.labels?.nominal ?? "Nominal robot (no shove)";
  const env = replay.environment;
  const payloadKg = att.payload_box?.mass_kg ?? 0;
  const attNow = outcomeAt(att.outcome, t);
  const nomNow = outcomeAt(nom.outcome, t);
  const activation = replay.reflex?.activation;

  return (
    <div className="replay" ref={rootRef}>
      <div className="replay-head">
        <div className="replay-title">
          <strong>{title}</strong>
          {sample && <span className="badge pre">Sample replay · precomputed preset, re-simulated on load</span>}
        </div>
        <div className="row">
          <SourceBadge replay={replay} />
          <button className="icon" onClick={fullscreen} title="Expand / fullscreen">⤢</button>
        </div>
      </div>
      <div className="canvas" ref={mountRef}>
        <div className="legend-overlay">
          <div><span className="swatch-robot main" /> {attackedLabel} <span className="now" style={{ color: attNow.color }}>· {attNow.text}</span></div>
          <div><span className="swatch-robot ghost" /> {nominalLabel} <span className="now" style={{ color: nomNow.color }}>· {nomNow.text}</span></div>
          {replay.push && <div><span className="swatch-arrow" /> shove {replay.push.force_n.toFixed(0)} N, {fmt(replay.push.start_s, 3)}–{fmt(replay.push.end_s, 3)} s</div>}
          {payloadKg > 0 && <div><span className="swatch-box" /> {payloadKg} kg payload (welded, collision-free)</div>}
          {env && Math.abs(env.mu - env.nominal_mu) > 1e-9 && (
            <div>floor μ {env.mu.toFixed(2)} (trained {env.nominal_mu.toFixed(1)}) · {env.mu < env.nominal_mu ? "blue: more slippery" : "tan: grippier"}</div>
          )}
        </div>
        <div className="physics-note">Recorded MuJoCo poses · no physics in the browser</div>
        {busy && <div className="busy-overlay">{busy}</div>}
      </div>
      <div className="timeline" title="orange: shove · red ticks: fall events · purple band: reflex activation">
        {replay.push && (
          <div className="tl-push" style={{ left: pctOf(replay.push.start_s), width: pctOf(replay.push.end_s - replay.push.start_s) }} />
        )}
        {activation &&
          activation.map((a, i) => (a > 0 ? <div key={`a${i}`} className="tl-reflex" style={{ left: pctOf(i / replay.fps), opacity: 0.25 + 0.75 * a }} /> : null))}
        {att.nonfoot.map((on, i) => (on ? <div key={i} className="tl-nonfoot" style={{ left: pctOf(i / replay.fps) }} /> : null))}
        {events.map(([name, time]) => (
          <div key={name} className={`tl-event ${name === att.outcome.reason ? "primary" : ""}`} style={{ left: pctOf(time) }} title={`${EVENT_LABEL[name] ?? name} @ ${fmt(time)} s`} />
        ))}
        <div className="tl-cursor" style={{ left: pctOf(t) }} />
      </div>
      <div className="controls">
        <button onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
        <input type="range" aria-label="replay frame" min={0} max={replay.frames - 1} value={frame} onChange={(e) => { setPlaying(false); seek(Number(e.target.value)); }} />
        <select value={speed} aria-label="playback speed" onChange={(e) => setSpeed(Number(e.target.value))}>
          {[0.1, 0.25, 0.5, 1].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        {replay.push && <button onClick={() => { setPlaying(false); seek(Math.max(0, Math.round((replay.push!.start_s - 0.3) * replay.fps))); }}>Jump to shove</button>}
      </div>
      <div className="readout">
        <span>t = {fmt(t, 2)} s</span>
        <span>opaque tilt {fmt(tilt, 1)}°</span>
        <span>trunk z {fmt(att.trunk_z[f], 3)} m</span>
        <span>feet {replay.foot_order.map((foot, i) => (att.feet[f]?.[i] ? foot : "··")).join(" ")}</span>
        <span className={att.nonfoot[f] ? "bad" : ""}>non-foot contact {att.nonfoot[f] ? "YES" : "no"}</span>
        {activation && <span>reflex activation {fmt(activation[Math.min(frame, activation.length - 1)], 2)}</span>}
        {f >= att.frames - 1 && frame > f && <span className="muted">opaque episode ended at {fmt((att.frames - 1) / replay.fps, 2)} s</span>}
      </div>
      <div className="events">
        <EpisodeLine label={attackedLabel} c={att.outcome} horizon={duration} />
        <EpisodeLine label={nominalLabel} c={nom.outcome} horizon={duration} />
        {replay.push && events.length > 0 && (
          <div className="muted small">
            Opaque robot events: {events.map(([n, time]) => `${EVENT_LABEL[n] ?? n} ${fmt(time)} s`).join(", ")}.
            {att.outcome.first_nonfoot_body && ` First non-foot contact: ${att.outcome.first_nonfoot_body}.`}
          </div>
        )}
      </div>
    </div>
  );
}

function EpisodeLine({ label, c, horizon }: { label: string; c: Replay["cases"]["attacked"]["outcome"]; horizon: number }) {
  const text =
    c.outcome === "survived"
      ? `survived the ${fmt(horizon, 2)} s horizon`
      : c.outcome === "fell"
        ? `FELL: ${c.reason} at ${fmt(c.fall_time_s)} s${c.time_after_start_s !== null ? ` (${fmt(c.time_after_start_s)} s after shove start)` : ""}`
        : `NUMERICAL FAILURE (${c.reason}) at ${fmt(c.numerical_time_s)} s — not a fall`;
  return (
    <div>
      <span className="dot" style={{ background: COLORS[c.outcome] }} /> <strong>{label}:</strong> {text}; progress {fmt(c.forward_progress_m, 2)} m
    </div>
  );
}

function SourceBadge({ replay }: { replay: Replay }) {
  if (replay.source === "recorded") {
    if (replay.mode) {
      return (
        <span className="badge live">
          Recorded trajectory · scenario run {replay.run_id}
          {replay.mode === "suite" ? ` trial ${replay.trial}` : ""}
        </span>
      );
    }
    return <span className="badge live">Recorded trajectory · live run {replay.run_id} trial {replay.trial}</span>;
  }
  const v = replay.verification;
  const what = v?.what ?? (replay.which ? `discovery record (${replay.which === "final" ? "final failure" : "search failure"})` : "M2 result");
  if (!v) return <span className="badge pre">Re-simulated now on the server</span>;
  return (
    <span className={`badge ${v.all_match ? "ok" : "warn"}`} title={JSON.stringify(v.cases)}>
      Re-simulated now · {v.all_match ? `matches recorded ${what} (commit ${v.recorded_commit.slice(0, 7)})` : "DOES NOT match the recorded result"}
    </span>
  );
}
