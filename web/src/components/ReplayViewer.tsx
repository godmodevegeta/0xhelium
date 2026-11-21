// Three.js playback of recorded MuJoCo poses. No physics runs here: every frame is a pose the
// server computed with mj_kinematics from a recorded qpos row.
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { COLORS, decodeFloat32, decodeInt32, fmt, type ModelPayload, type Replay } from "../api";

interface Props {
  model: ModelPayload;
  replay: Replay;
  title: string;
}

interface Robot {
  group: THREE.Group;
  meshes: THREE.Mesh[];
  pos: Float32Array;
  quat: Float32Array;
  frames: number;
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

export function ReplayViewer({ model, replay, title }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0.5);
  const frameRef = useRef(0);
  // Continuous playhead in seconds. The displayed frame is derived from it; keeping the
  // fractional time is what lets slow speeds advance (0.5x at 60 Hz adds 0.42 frames per render).
  const timeRef = useRef(0);
  const playingRef = useRef(true);
  const speedRef = useRef(0.5);
  const geometries = useMemo(() => buildGeometries(model), [model]);
  const duration = (replay.frames - 1) / replay.fps;

  playingRef.current = playing;
  speedRef.current = speed;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f4f6f9");
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(-2.1, 1.3, 1.1); // behind and to the left: a sideways shove reads left-to-right
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

    const makeRobot = (key: "nominal" | "attacked", material: THREE.Material): Robot => {
      const group = new THREE.Group();
      const meshes = model.geoms.map((g) => {
        const mesh = new THREE.Mesh(geometries[g.mesh], material);
        group.add(mesh);
        return mesh;
      });
      scene.add(group);
      const c = replay.cases[key];
      return { group, meshes, pos: decodeFloat32(c.pos), quat: decodeFloat32(c.quat), frames: c.frames };
    };
    const nominalMaterial = new THREE.MeshStandardMaterial({ color: "#5b8bd6", transparent: true, opacity: 0.35, depthWrite: false });
    const attackedMaterial = new THREE.MeshStandardMaterial({ color: "#3a3f47", metalness: 0.1, roughness: 0.6 });
    const robots = [makeRobot("nominal", nominalMaterial), makeRobot("attacked", attackedMaterial)];

    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), 0.5, 0xe07a10, 0.18, 0.12);
    arrow.visible = false;
    scene.add(arrow);
    if (replay.push) {
      const [fx, fy, fz] = replay.push.world_force_n;
      arrow.setDirection(new THREE.Vector3(fx, fy, fz).normalize());
    }

    const G = replay.geom_count;
    const pose = (robot: Robot, f: number) => {
      const i = Math.min(f, robot.frames - 1);
      robot.meshes.forEach((mesh, j) => {
        const p = (i * G + j) * 3;
        const q = (i * G + j) * 4;
        mesh.position.set(robot.pos[p], robot.pos[p + 1], robot.pos[p + 2]);
        mesh.quaternion.set(robot.quat[q + 1], robot.quat[q + 2], robot.quat[q + 3], robot.quat[q]);
      });
    };
    const trunk = (robot: Robot, f: number) => {
      const i = Math.min(f, robot.frames - 1) * G * 3; // geom 0 is the trunk
      return new THREE.Vector3(robot.pos[i], robot.pos[i + 1], robot.pos[i + 2]);
    };

    let last = performance.now();
    let raf = 0;
    let previousTarget = trunk(robots[1], 0);
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1); // a hidden tab must not jump the playhead
      last = now;
      if (playingRef.current) {
        timeRef.current += dt * speedRef.current;
        if (timeRef.current > duration) timeRef.current = 0; // loop
        const f = Math.min(replay.frames - 1, Math.floor(timeRef.current * replay.fps + 1e-9));
        if (f !== frameRef.current) {
          frameRef.current = f;
          setFrame(f);
        }
      }
      const f = frameRef.current;
      robots.forEach((r) => pose(r, f));
      const t = f / replay.fps;
      if (replay.push && t >= replay.push.start_s - 1e-9 && t < replay.push.end_s + 1e-9) {
        const at = trunk(robots[1], f);
        const [fx, fy] = replay.push.world_force_n;
        const away = new THREE.Vector3(-fx, -fy, 0).normalize().multiplyScalar(0.55);
        arrow.position.copy(at).add(away);
        arrow.visible = true;
      } else {
        arrow.visible = false;
      }
      const target = trunk(robots[1], f).add(trunk(robots[0], f)).multiplyScalar(0.5);
      target.z = 0.25;
      camera.position.add(target.clone().sub(previousTarget));
      controls.target.copy(target);
      previousTarget = target;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      nominalMaterial.dispose();
      attackedMaterial.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [model, replay, geometries, duration]);

  useEffect(() => {
    const requested = Number(new URLSearchParams(window.location.search).get("frame"));
    const start = Number.isInteger(requested) && requested > 0 ? Math.min(requested, replay.frames - 1) : 0;
    frameRef.current = start;
    timeRef.current = start / replay.fps;
    setFrame(start);
    setPlaying(start === 0);
  }, [replay]);

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
  const att = replay.cases.attacked;
  const nom = replay.cases.nominal;
  const f = Math.min(frame, att.frames - 1);
  const t = frame / replay.fps;
  const tilt = (Math.acos(Math.max(-1, Math.min(1, att.up_z[f] ?? 1))) * 180) / Math.PI;
  const events = Object.entries(att.outcome.events).sort((a, b) => a[1] - b[1]);
  const pctOf = (time: number) => `${(100 * time) / duration}%`;

  return (
    <div className="replay">
      <div className="replay-head">
        <div>
          <strong>{title}</strong>
          <div className="muted small">
            Shoved robot opaque, nominal robot translucent blue; one shared clock. Poses are recorded MuJoCo
            states, not browser physics.
          </div>
        </div>
        <SourceBadge replay={replay} />
      </div>
      <div className="canvas" ref={mountRef} />
      <div className="timeline">
        {replay.push && (
          <div
            className="tl-push"
            style={{ left: pctOf(replay.push.start_s), width: pctOf(replay.push.end_s - replay.push.start_s) }}
            title={`shove ${fmt(replay.push.start_s)}–${fmt(replay.push.end_s)} s`}
          />
        )}
        {att.nonfoot.map((on, i) =>
          on ? <div key={i} className="tl-nonfoot" style={{ left: pctOf(i / replay.fps) }} /> : null,
        )}
        {events.map(([name, time]) => (
          <div key={name} className={`tl-event ${name === att.outcome.reason ? "primary" : ""}`} style={{ left: pctOf(time) }} title={`${EVENT_LABEL[name] ?? name} @ ${fmt(time)} s`} />
        ))}
        <div className="tl-cursor" style={{ left: pctOf(t) }} />
      </div>
      <div className="controls">
        <button onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
        <input type="range" min={0} max={replay.frames - 1} value={frame} onChange={(e) => { setPlaying(false); seek(Number(e.target.value)); }} />
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          {[0.1, 0.25, 0.5, 1].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        {replay.push && <button onClick={() => { setPlaying(false); seek(Math.max(0, Math.round((replay.push!.start_s - 0.3) * replay.fps))); }}>Jump to shove</button>}
      </div>
      <div className="readout">
        <span>t = {fmt(t, 2)} s</span>
        <span>shoved tilt {fmt(tilt, 1)}°</span>
        <span>trunk z {fmt(att.trunk_z[f], 3)} m</span>
        <span>feet {replay.foot_order.map((foot, i) => (att.feet[f]?.[i] ? foot : "··")).join(" ")}</span>
        <span className={att.nonfoot[f] ? "bad" : ""}>non-foot contact {att.nonfoot[f] ? "YES" : "no"}</span>
        {f >= att.frames - 1 && frame > f && <span className="muted">shoved episode ended at {fmt((att.frames - 1) / replay.fps, 2)} s</span>}
      </div>
      <div className="events">
        <EpisodeLine label="Shoved" c={att.outcome} />
        <EpisodeLine label="Nominal" c={nom.outcome} />
        {replay.push && (
          <div className="muted small">
            Shove: {replay.push.force_n} N, world force [{replay.push.world_force_n.map((v) => v.toFixed(0)).join(", ")}] N, {fmt(replay.push.start_s)}–{fmt(replay.push.end_s)} s.
            {events.length > 0 && " Events: " + events.map(([n, time]) => `${EVENT_LABEL[n] ?? n} ${fmt(time)} s`).join(", ") + "."}
            {att.outcome.first_nonfoot_body && ` First non-foot contact: ${att.outcome.first_nonfoot_body}.`}
          </div>
        )}
      </div>
    </div>
  );
}

function EpisodeLine({ label, c }: { label: string; c: Replay["cases"]["attacked"]["outcome"] }) {
  const text =
    c.outcome === "survived"
      ? "survived the 5 s horizon"
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
    return <span className="badge live">Recorded trajectory · live run {replay.run_id} trial {replay.trial}</span>;
  }
  const v = replay.verification;
  return (
    <span className={`badge ${v?.all_match ? "ok" : "warn"}`} title={v ? JSON.stringify(v.cases) : ""}>
      Re-simulated now · {v?.all_match ? `matches recorded M2 result (commit ${v.recorded_commit.slice(0, 7)})` : "DOES NOT match the recorded result"}
    </span>
  );
}
