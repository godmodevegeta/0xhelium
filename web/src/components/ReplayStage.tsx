// The persistent 3D stage: Three.js playback of recorded MuJoCo poses. No physics runs here; every
// frame is a pose the server computed with mj_kinematics from a recorded qpos row.
//
// One renderer, scene and camera live as long as the robot model; a new replay only swaps the two
// robot mesh groups, so mode switches never remount the stage or refetch meshes, and the camera
// offset and playback speed carry over between replays.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { COLORS, decodeFloat32, decodeInt32, fmt, type ModelPayload, type Replay } from "../api";
import { directionWord, robotNames, sourceInfo } from "../replayInfo";
import { PALETTE } from "../theme";

interface Current {
  data: Replay;
  title: string;
  sample: boolean;
}

interface Props {
  model: ModelPayload | null;
  current: Current | null;
  busy: string | null; // a replay is loading or a job is running: the shown replay is the previous one
  error: string | null;
}

interface Robot {
  group: THREE.Group;
  meshes: THREE.Mesh[];
  pos: Float32Array;
  quat: Float32Array;
  frames: number;
  payload: { mesh: THREE.Mesh; pos: Float32Array; quat: Float32Array } | null;
}

interface Engine {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  arrow: THREE.ArrowHelper;
  ground: THREE.Mesh;
  materials: { ghost: THREE.Material; main: THREE.Material; ghostBox: THREE.Material; box: THREE.Material };
  geometries: Record<string, THREE.BufferGeometry>;
  resetCamera: () => void;
  retarget: () => void;
}

const DEFAULT_OFFSET = new THREE.Vector3(-1.35, 0.9, 0.62); // three-quarter view from behind-left
const LABEL_LIFT_M = 0.14;
const EVENT_LABEL: Record<string, string> = {
  tilt: "tilt > 45°",
  nonfoot_contact: "non-foot contact",
  low_height: "trunk < 0.18 m",
  nonfinite_state: "non-finite state",
  mujoco_warning: "MuJoCo warning",
};
let frameParamUsed = false;

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

type CaseOutcome = Replay["cases"]["attacked"]["outcome"];

export function stateAt(c: CaseOutcome, t: number): { text: string; color: string } {
  if (c.outcome === "fell" && c.fall_time_s !== null && t >= c.fall_time_s - 1e-9) {
    return { text: `FELL · ${EVENT_LABEL[c.reason ?? ""] ?? c.reason} at ${fmt(c.fall_time_s, 2)} s`, color: COLORS.fell };
  }
  if (c.outcome === "numerical" && c.numerical_time_s !== null && t >= c.numerical_time_s - 1e-9) {
    return { text: "numerical failure (not a fall)", color: COLORS.numerical };
  }
  return { text: "upright", color: COLORS.survived };
}

/** Consecutive frames with activation > 0, as [start, end) frame segments. */
function segments(values: number[] | undefined): [number, number][] {
  const out: [number, number][] = [];
  if (!values) return out;
  let start = -1;
  values.forEach((v, i) => {
    if (v > 0 && start < 0) start = i;
    if (v <= 0 && start >= 0) {
      out.push([start, i]);
      start = -1;
    }
  });
  if (start >= 0) out.push([start, values.length]);
  return out;
}

export function ReplayStage({ model, current, busy, error }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const mainLabelRef = useRef<HTMLDivElement>(null);
  const ghostLabelRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const placeRef = useRef<(() => void) | null>(null); // re-place labels from outside the loop
  const robotsRef = useRef<Robot[]>([]);
  const replayRef = useRef<Replay | null>(null);
  const floorRef = useRef<THREE.Mesh | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0.5);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);
  const frameRef = useRef(0);
  const timeRef = useRef(0); // continuous playhead in seconds (slow speeds advance fractionally)
  const playingRef = useRef(true);
  const speedRef = useRef(0.5);
  const replay = current?.data ?? null;
  const duration = replay ? (replay.frames - 1) / replay.fps : 0;
  const durationRef = useRef(duration);
  const geometries = useMemo(() => (model ? buildGeometries(model) : null), [model]);

  playingRef.current = playing;
  speedRef.current = speed;
  replayRef.current = replay;
  durationRef.current = duration;

  // Renderer, scene, camera and loop: once per model.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !geometries) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.prepend(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.stageBg);
    scene.fog = new THREE.Fog(PALETTE.stageBg, 6, 16);
    const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / Math.max(mount.clientHeight, 1), 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.copy(DEFAULT_OFFSET).add(new THREE.Vector3(0, 0, 0.25));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.25);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.minDistance = 0.6;
    controls.maxDistance = 6;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x1a1a1a, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(2, -3, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xb9bdff, 0.7);
    rim.position.set(-3, 3, 2);
    scene.add(rim);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: PALETTE.ground, roughness: 1 }));
    ground.position.set(15, 0, -0.003);
    scene.add(ground);
    const grid = new THREE.GridHelper(60, 120, PALETTE.gridMajor, PALETTE.gridMinor);
    grid.rotation.x = Math.PI / 2;
    grid.position.x = 15;
    scene.add(grid);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), 0.5, PALETTE.accent, 0.18, 0.12);
    arrow.visible = false;
    scene.add(arrow);
    const materials = {
      ghost: new THREE.MeshStandardMaterial({ color: PALETTE.ghost, transparent: true, opacity: 0.34, depthWrite: false, roughness: 0.5 }),
      // "transparent" with opacity 1 puts the robot in the same render list as the ghost, so
      // renderOrder draws the ghost first and the robot over it where the two coincide.
      main: new THREE.MeshStandardMaterial({ color: PALETTE.robot, metalness: 0.15, roughness: 0.55, transparent: true, opacity: 1 }),
      ghostBox: new THREE.MeshStandardMaterial({ color: PALETTE.payload, transparent: true, opacity: 0.34, depthWrite: false }),
      box: new THREE.MeshStandardMaterial({ color: PALETTE.payload, roughness: 0.5 }),
    };

    let previousTarget: THREE.Vector3 | null = null;
    let forceDefaultView = false;
    const engine: Engine = {
      renderer, scene, camera, controls, arrow, ground, materials, geometries,
      resetCamera: () => { forceDefaultView = true; previousTarget = null; },
      retarget: () => { previousTarget = null; },
    };
    engineRef.current = engine;

    const trunk = (robot: Robot, f: number) => {
      const G = replayRef.current!.geom_count;
      const i = Math.min(f, robot.frames - 1) * G * 3; // geom 0 is the trunk
      return new THREE.Vector3(robot.pos[i], robot.pos[i + 1], robot.pos[i + 2]);
    };
    const pose = (robot: Robot, f: number) => {
      const G = replayRef.current!.geom_count;
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
    const project = (v: THREE.Vector3, w: number, h: number) => {
      const p = v.clone().project(camera);
      return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h, visible: p.z < 1 && p.z > -1 };
    };
    // Robot labels: anchored above each trunk, kept inside the canvas and below the HUD, and never
    // overlapping each other. When the two would collide, the ghost's label stacks above the
    // opaque robot's (or the pair moves down together if there is no room above).
    const GAP = 6;
    const placeLabels = (robots: Robot[], f: number) => {
      const main = mainLabelRef.current;
      const ghost = ghostLabelRef.current;
      if (!main || !ghost) return;
      const W = mount.clientWidth;
      const H = mount.clientHeight;
      const hud = mount.querySelector<HTMLElement>(".hud");
      const top = (hud && hud.offsetHeight ? hud.offsetTop + hud.offsetHeight : 0) + GAP;
      const lift = new THREE.Vector3(0, 0, LABEL_LIFT_M);
      const pa = project(trunk(robots[1], f).add(lift), W, H);
      const pb = project(trunk(robots[0], f).add(lift), W, H);
      const wa = main.offsetWidth, ha = main.offsetHeight, wb = ghost.offsetWidth, hb = ghost.offsetHeight;
      const clampX = (x: number, w: number) => Math.min(Math.max(x, w / 2 + GAP), Math.max(w / 2 + GAP, W - w / 2 - GAP));
      const clampY = (y: number, h: number) => Math.min(Math.max(y, top + h), H - GAP);
      let ax = clampX(pa.x, wa), ay = clampY(pa.y, ha);
      let bx = clampX(pb.x, wb), by = clampY(pb.y, hb);
      const xOverlap = Math.min(ax + wa / 2, bx + wb / 2) - Math.max(ax - wa / 2, bx - wb / 2) > -GAP;
      const yOverlap = Math.min(ay, by) - Math.max(ay - ha, by - hb) > -GAP;
      if (xOverlap && yOverlap) {
        by = ay - ha - GAP; // ghost label directly above the opaque robot's label
        if (by - hb < top) {
          by = top + hb;
          ay = by + GAP + ha;
        }
        bx = clampX(ax, wb);
        ax = clampX(ax, wa);
      }
      main.style.transform = `translate(${ax}px, ${ay}px) translate(-50%, -100%)`;
      ghost.style.transform = `translate(${bx}px, ${by}px) translate(-50%, -100%)`;
      main.style.visibility = pa.visible ? "visible" : "hidden";
      ghost.style.visibility = pb.visible ? "visible" : "hidden";
    };

    let last = performance.now();
    let raf = 0;
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1); // a hidden tab must not jump the playhead
      last = now;
      const rp = replayRef.current;
      const robots = robotsRef.current;
      if (rp && playingRef.current) {
        timeRef.current += dt * speedRef.current;
        if (timeRef.current > durationRef.current) timeRef.current = 0; // loop
        const f = Math.min(rp.frames - 1, Math.floor(timeRef.current * rp.fps + 1e-9));
        if (f !== frameRef.current) {
          frameRef.current = f;
          setFrame(f);
        }
      }
      const f = frameRef.current;
      if (rp && robots.length === 2) {
        robots.forEach((r) => pose(r, f));
        const t = f / rp.fps;
        if (rp.push && t >= rp.push.start_s - 1e-9 && t < rp.push.end_s + 1e-9) {
          const at = trunk(robots[1], f);
          const [fx, fy] = rp.push.world_force_n;
          arrow.position.copy(at).add(new THREE.Vector3(-fx, -fy, 0).normalize().multiplyScalar(0.55));
          arrow.visible = true;
        } else {
          arrow.visible = false;
        }
        const target = trunk(robots[1], f).add(trunk(robots[0], f)).multiplyScalar(0.5);
        target.z = 0.25;
        if (previousTarget === null) {
          // New replay or reset: keep the user's offset (or restore the default one) at the new target.
          const offset = forceDefaultView ? DEFAULT_OFFSET.clone() : camera.position.clone().sub(controls.target);
          forceDefaultView = false;
          camera.position.copy(target).add(offset);
        } else {
          camera.position.add(target.clone().sub(previousTarget));
        }
        controls.target.copy(target);
        previousTarget = target;
        ground.position.x = target.x + 15;
        if (floorRef.current) floorRef.current.position.x = target.x + 15;
        grid.position.x = Math.round(target.x) + 15;
      }
      controls.update();
      renderer.render(scene, camera);
      if (rp && robots.length === 2) placeLabels(robots, f);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    placeRef.current = () => {
      const rp = replayRef.current;
      const robots = robotsRef.current;
      if (rp && robots.length === 2) placeLabels(robots, frameRef.current);
    };
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
    return () => {
      cancelAnimationFrame(raf);
      placeRef.current = null;
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      Object.values(materials).forEach((m) => m.dispose());
      renderer.domElement.remove();
      engineRef.current = null;
    };
  }, [geometries]);

  // A label's text (and width) changes in a React commit; place it again before the browser paints
  // instead of waiting for the next animation frame, so a wider label is never drawn unclamped.
  useLayoutEffect(() => {
    placeRef.current?.();
  });

  // Robots, floor cue and shove arrow: per replay.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !replay || !model) return;
    const { scene, materials, arrow } = engine;
    const boxGeometries: THREE.BufferGeometry[] = [];
    const makeRobot = (key: "nominal" | "attacked", material: THREE.Material, boxMaterial: THREE.Material): Robot => {
      const group = new THREE.Group();
      const meshes = model.geoms.map((g) => {
        const mesh = new THREE.Mesh(engine.geometries[g.mesh], material);
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
      group.traverse((o) => { o.renderOrder = key === "nominal" ? 0 : 1; });
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
      const color = env.mu < env.nominal_mu ? PALETTE.lowFriction : PALETTE.highFriction;
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color, roughness: 1, metalness: 0 }));
      floor.position.set(15, 0, -0.002);
      scene.add(floor);
      floorRef.current = floor;
    }
    engine.retarget();
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
    if (replay) timeRef.current = f / replay.fps;
    setFrame(f);
  };
  const togglePlay = () => {
    if (!replay) return;
    if (!playing && frameRef.current >= replay.frames - 1) seek(0);
    setPlaying((p) => !p);
  };
  // Fullscreen, or an in-page expanded view where the browser refuses fullscreen (some mobile
  // browsers, embedded and headless contexts).
  const fullscreen = () => {
    const el = rootRef.current;
    if (!el) return;
    if (document.fullscreenElement) return void document.exitFullscreen();
    if (expanded) return setExpanded(false);
    const request = el.requestFullscreen?.();
    if (!request) setExpanded(true);
    else request.catch(() => setExpanded(true));
  };

  const names = replay ? robotNames(replay) : null;
  const src = replay && current ? sourceInfo(replay, current.sample) : null;
  const t = replay ? frame / replay.fps : 0;
  const att = replay?.cases.attacked;
  const nom = replay?.cases.nominal;
  const mainState = att ? stateAt(att.outcome, t) : null;
  const ghostState = nom ? stateAt(nom.outcome, t) : null;
  const env = replay?.environment;
  const payloadKg = att?.payload_box?.mass_kg ?? 0;

  return (
    <div className={`stage-card${expanded ? " expanded" : ""}`} ref={rootRef}>
      <div className="stage-head">
        <div className="stage-title">
          <strong title={current?.title}>{current ? current.title : "Replay"}</strong>
          <span className="xsmall muted">
            {names ? <>Opaque: {names.main} · Translucent: {names.ghost}</> : "Recorded MuJoCo trajectories from the server"}
          </span>
        </div>
        {src && <span className={`badge ${src.tone === "ok" ? "ok" : src.tone === "warn" ? "warn" : current?.sample ? "pre" : "live"}`} title={src.text}>{src.text}</span>}
      </div>
      <div className="canvas" ref={mountRef}>
        <div className="hud" aria-label="Scene facts">
          {replay?.push && (
            <span className="hud-chip attack">
              shove {replay.push.force_n.toFixed(0)} N {directionWord(replay.push.world_force_n)} · {fmt(replay.push.start_s, 3)}–{fmt(replay.push.end_s, 3)} s
            </span>
          )}
          {replay && !replay.push && <span className="hud-chip">no shove</span>}
          {env && Math.abs(env.mu - env.nominal_mu) > 1e-9 && <span className="hud-chip">floor μ {env.mu.toFixed(2)} (trained {env.nominal_mu.toFixed(1)})</span>}
          {payloadKg > 0 && <span className="hud-chip">{payloadKg} kg payload, welded</span>}
          {busy && replay && <span className="hud-chip prev">Previous replay · kept until the new result arrives</span>}
        </div>
        {replay && names && mainState && ghostState && (
          <>
            <div className="robot-label main" ref={mainLabelRef} aria-hidden="true">
              <span className="swatch-label" style={{ background: PALETTE.robot }} /> {names.main} · <span className="state" style={{ color: mainState.color }}>{mainState.text}</span>
            </div>
            <div className="robot-label ghost" ref={ghostLabelRef} aria-hidden="true">
              <span className="swatch-label ghost" /> {names.ghost} · <span className="state" style={{ color: ghostState.color }}>{ghostState.text}</span>
            </div>
          </>
        )}
        <div className="stage-note">Recorded MuJoCo poses · no physics in the browser</div>
        {(!model || !current) && (
          <div className="stage-overlay" role="status">
            {error ? <span className="error">{error}</span> : <><span className="spinner" /> {!model ? "Loading robot meshes…" : "Loading the sample replay (re-simulated on the server)…"}</>}
          </div>
        )}
        {model && current && (busy || error) && (
          <div className="stage-overlay" role="status">
            {busy ? <><span className="spinner" /> {busy}</> : <span className="error">{error}</span>}
          </div>
        )}
      </div>
      {replay && att && nom ? (
        <Transport
          replay={replay}
          frame={frame}
          playing={playing}
          speed={speed}
          duration={duration}
          onToggle={togglePlay}
          onSeek={(f) => { setPlaying(false); seek(f); }}
          onSpeed={setSpeed}
          onReset={() => engineRef.current?.resetCamera()}
          onFullscreen={fullscreen}
          expanded={expanded}
        />
      ) : (
        <div className="transport" aria-hidden="true"><span className="muted small">Playback appears when a replay has loaded.</span></div>
      )}
    </div>
  );
}

function Transport({ replay, frame, playing, speed, duration, onToggle, onSeek, onSpeed, onReset, onFullscreen, expanded }: {
  replay: Replay; frame: number; playing: boolean; speed: number; duration: number; expanded: boolean;
  onToggle: () => void; onSeek: (f: number) => void; onSpeed: (s: number) => void; onReset: () => void; onFullscreen: () => void;
}) {
  const att = replay.cases.attacked;
  const nom = replay.cases.nominal;
  const pct = (time: number) => `${Math.max(0, Math.min(100, (100 * time) / Math.max(duration, 1e-9)))}%`;
  const t = frame / replay.fps;
  const f = Math.min(frame, att.frames - 1);
  const tilt = (Math.acos(Math.max(-1, Math.min(1, att.up_z[f] ?? 1))) * 180) / Math.PI;
  const reflex = segments(replay.reflex?.activation);
  const names = robotNames(replay);
  const fallTitle = (c: CaseOutcome, who: string) =>
    `${who} fell: ${EVENT_LABEL[c.reason ?? ""] ?? c.reason} at ${fmt(c.fall_time_s)} s${c.time_after_start_s !== null ? ` (${fmt(c.time_after_start_s)} s after the shove)` : ""}`;
  return (
    <div className="transport">
      <button className="play" onClick={onToggle} aria-label={playing ? "Pause" : "Play"} title={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "▶"}
      </button>
      <div className="track-wrap">
        <div className="track">
          <div className="track-rail" />
          <div className="track-fill" style={{ width: pct(t) }} />
          {replay.push && <div className="track-push" style={{ left: pct(replay.push.start_s), width: `calc(${pct(replay.push.end_s - replay.push.start_s)} + 2px)` }} title={`shove ${fmt(replay.push.start_s)}–${fmt(replay.push.end_s)} s`} />}
          {reflex.map(([a, b]) => <div key={a} className="track-reflex" style={{ left: pct(a / replay.fps), width: pct((b - a) / replay.fps) }} />)}
          {nom.outcome.outcome === "fell" && nom.outcome.fall_time_s !== null && <div className="track-fall ghost" style={{ left: pct(nom.outcome.fall_time_s) }} title={fallTitle(nom.outcome, names.ghost)} />}
          {att.outcome.outcome === "fell" && att.outcome.fall_time_s !== null && <div className="track-fall main" style={{ left: pct(att.outcome.fall_time_s) }} title={fallTitle(att.outcome, names.main)} />}
          <div className="track-head" style={{ left: pct(t) }} />
          <input type="range" aria-label="Scrub replay" min={0} max={replay.frames - 1} value={frame} onChange={(e) => onSeek(Number(e.target.value))} />
        </div>
        <div className="track-meta">
          <span>{fmt(t, 2)} / {fmt(duration, 2)} s · tilt {fmt(tilt, 1)}° · trunk {fmt(att.trunk_z[f], 2)} m · feet {replay.foot_order.map((foot, i) => (att.feet[f]?.[i] ? foot : "··")).join(" ")}</span>
        </div>
      </div>
      <div className="transport-tools">
        <span className="key xsmall muted">
          {replay.push && <span><span className="key-swatch" style={{ background: PALETTE.accent }} />shove</span>}
          <span><span className="key-swatch" style={{ background: PALETTE.fell }} />opaque fall</span>
          <span><span className="key-swatch" style={{ background: PALETTE.candidate }} />ghost fall</span>
          {reflex.length > 0 && <span><span className="key-swatch" style={{ background: PALETTE.candidate, height: 3 }} />reflex active</span>}
        </span>
        <span className="spacer" />
        <div className="segmented" role="group" aria-label="Playback speed">
          {[0.1, 0.25, 0.5, 1].map((s) => (
            <button key={s} aria-pressed={speed === s} onClick={() => onSpeed(s)}>{s}×</button>
          ))}
        </div>
        {replay.push && <button className="icon" onClick={() => onSeek(Math.max(0, Math.round((replay.push!.start_s - 0.3) * replay.fps)))} aria-label="Jump to the shove" title="Jump to the shove">⇥</button>}
        <button className="icon" onClick={onReset} aria-label="Reset camera view" title="Reset camera view">⟲</button>
        <button className="icon" onClick={onFullscreen} aria-label={expanded ? "Exit expanded view" : "Fullscreen"} title={expanded ? "Exit expanded view (Esc)" : "Fullscreen"}>{expanded ? "✕" : "⤢"}</button>
      </div>
    </div>
  );
}
