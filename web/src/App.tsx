import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  hardeningApi,
  type DiscoveryInfo,
  type HardeningInfo,
  type HiddenPhase as HiddenPhaseData,
  type LiveStress as LiveStressData,
  type ModelPayload,
  type Replay,
  type ScenarioConfig,
} from "./api";
import { Discovery } from "./components/Discovery";
import { Hardening } from "./components/Hardening";
import { HiddenPhase } from "./components/HiddenPhase";
import { LiveStress } from "./components/LiveStress";
import { ReplayViewer } from "./components/ReplayViewer";
import { ScenarioDesigner } from "./components/ScenarioDesigner";
import { ModeContext } from "./components/Workspace";

type Tab = "hidden" | "live" | "discovery" | "designer" | "hardening";
const TABS: { key: Tab; label: string }[] = [
  { key: "designer", label: "Scenario Designer" },
  { key: "hidden", label: "Hidden Phase" },
  { key: "live", label: "Stress Test" },
  { key: "discovery", label: "Failure Discovery" },
  { key: "hardening", label: "Policy Hardening" },
];

export type ShowReplay = (key: string, title: string, load: () => Promise<Replay>) => Promise<void>;

export function App() {
  const params = new URLSearchParams(window.location.search);
  const initialTab = (TABS.find((t) => t.key === params.get("tab"))?.key ?? "designer") as Tab;
  const [tab, setTab] = useState<Tab>(initialTab);
  const [model, setModel] = useState<ModelPayload | null>(null);
  const [hidden, setHidden] = useState<HiddenPhaseData | null>(null);
  const [live, setLive] = useState<LiveStressData | null>(null);
  const [discovery, setDiscovery] = useState<DiscoveryInfo | null>(null);
  const [designer, setDesigner] = useState<ScenarioConfig | null>(null);
  const [hardening, setHardening] = useState<HardeningInfo | null>(null);
  const [replay, setReplay] = useState<{ data: Replay; title: string; sample: boolean } | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [loadingTitle, setLoadingTitle] = useState<string | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<HTMLElement | null>(null);
  const stageRef = useRef<HTMLElement>(null);
  const requested = useRef(false); // a deep link or a user action asked for a replay

  const refreshLive = useCallback(() => {
    api.liveStress().then(setLive).catch((e) => setError(String(e)));
  }, []);
  const refreshDiscovery = useCallback(() => {
    api.discoveryInfo().then(setDiscovery).catch((e) => setError(String(e)));
  }, []);
  const refreshDesigner = useCallback(() => {
    api.scenarioConfig().then(setDesigner).catch((e) => setError(String(e)));
  }, []);
  const refreshHardening = useCallback(() => {
    hardeningApi.info().then(setHardening).catch((e) => setError(String(e)));
  }, []);

  const show: ShowReplay = useCallback(async (key, title, load) => {
    requested.current = true;
    setLoadingKey(key);
    setLoadingTitle(title);
    setReplayError(null);
    try {
      const data = await load();
      setReplay({ data, title, sample: false });
      const stage = stageRef.current;
      if (stage && stage.getBoundingClientRect().top < 0) stage.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      setReplayError(`${title}: ${String(e)}`);
    } finally {
      setLoadingKey(null);
      setLoadingTitle(null);
    }
  }, []);

  const replayHidden = useCallback(
    (substep: number) => show(`hidden/${substep}`, `Hidden Phase · shove at ${(substep * 0.004).toFixed(3)} s`, () => api.hiddenReplay(substep)),
    [show],
  );
  const replayScenario = useCallback(
    (runId: string, trial: number, ghost: "primary" | "matched" | "original", title: string) =>
      show(`scenario/${runId}/${trial}/${ghost}`, title, () => api.scenarioReplay(runId, trial, ghost)),
    [show],
  );

  useEffect(() => {
    api.hiddenPhase()
      .then((h) => {
        setHidden(h);
        // Sample replay on first load (unless a deep link asked for something else).
        if (!requested.current) {
          const k = h.verified_failure_substep;
          setLoadingKey(`hidden/${k}`);
          setLoadingTitle("Loading sample replay…");
          api.hiddenReplay(k)
            .then((data) => {
              if (!requested.current) setReplay({ data, title: `Hidden Phase · 100 N shove at ${(k * 0.004).toFixed(3)} s`, sample: true });
            })
            .catch((e) => setReplayError(`sample replay: ${String(e)}`))
            .finally(() => { setLoadingKey((cur) => (cur === `hidden/${k}` ? null : cur)); setLoadingTitle(null); });
        }
      })
      .catch((e) => setError(String(e)));
    refreshLive();
    refreshDiscovery();
    refreshDesigner();
    refreshHardening();
    api.model().then(setModel).catch((e) => setError(String(e)));
    // Deep links: ?replay=<start substep> (Hidden Phase), ?scenario_replay=<run>:<trial>:<ghost>,
    // ?frame=<n> (initial frame); mode components handle ?run= and their own replay links.
    const substep = params.get("replay");
    if (substep !== null && /^\d+$/.test(substep)) void replayHidden(Number(substep));
    const sr = params.get("scenario_replay")?.match(/^(sc-\d{8}T\d{6}Z-[0-9a-f]{8}):(\d{1,2}):(primary|matched|original)$/);
    if (sr) void replayScenario(sr[1], Number(sr[2]), sr[3] as "primary" | "matched" | "original", `Scenario run ${sr[1]} · trial ${sr[2]} · ghost ${sr[3]}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const replayLive = (runId: string, trial: number) => show(`${runId}/${trial}`, `Stress Test · run ${runId} · trial ${trial}`, () => api.runReplay(runId, trial));
  const replayDiscovery = (runId: string, which: "final" | "search") =>
    show(`discovery/${runId}/${which}`, `Failure Discovery · ${runId} · ${which === "final" ? "discovered failure" : "confirmed search failure"}`, () => api.discoveryReplay(runId, which));
  const hiddenLoading = loadingKey?.startsWith("hidden/") ? Number(loadingKey.split("/")[1]) : null;

  const slot = (key: Tab) => ({ active: tab === key, details });
  const stage = (
    <>
      {replay && model && (
        <ReplayViewer model={model} replay={replay.data} title={replay.title} sample={replay.sample} busy={loadingTitle ? `${loadingTitle} — simulating on the server…` : null} />
      )}
      {(!replay || !model) && (
        <div className="stage-empty">
          {replayError ? <p className="error">{replayError}</p> : <p className="muted">{!model ? "Loading robot meshes…" : "Loading sample replay (re-simulated on the server)…"}</p>}
        </div>
      )}
      {replay && replayError && <p className="error small">{replayError}</p>}
    </>
  );

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Robustness Studio</h1>
          <div className="muted small">Unitree Go1 walking policy · native MuJoCo 3.15 on the server · recorded replays in the browser</div>
        </div>
        <nav aria-label="modes">
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? "tab active" : "tab"} onClick={() => setTab(t.key)}>{t.label}</button>
          ))}
        </nav>
      </header>
      {error && <div className="error banner">{error}</div>}
      <div className="workspace">
        <aside className="side" aria-label="experiment controls">
          <ModeContext.Provider value={slot("designer")}>
            <div hidden={tab !== "designer"}>
              {designer ? <ScenarioDesigner config={designer} onReplay={replayScenario} onRefresh={refreshDesigner} replaying={loadingKey} /> : <p className="muted">Loading…</p>}
            </div>
          </ModeContext.Provider>
          <ModeContext.Provider value={slot("hidden")}>
            <div hidden={tab !== "hidden"}>
              {hidden ? <HiddenPhase data={hidden} onReplay={replayHidden} replaying={hiddenLoading} /> : <p className="muted">Loading preset…</p>}
            </div>
          </ModeContext.Provider>
          <ModeContext.Provider value={slot("live")}>
            <div hidden={tab !== "live"}>
              {live ? <LiveStress data={live} onReplay={replayLive} onRefresh={refreshLive} replaying={loadingKey} /> : <p className="muted">Loading preset…</p>}
            </div>
          </ModeContext.Provider>
          <ModeContext.Provider value={slot("discovery")}>
            <div hidden={tab !== "discovery"}>
              {discovery ? <Discovery info={discovery} onReplay={replayDiscovery} onRefresh={refreshDiscovery} replaying={loadingKey} /> : <p className="muted">Loading…</p>}
            </div>
          </ModeContext.Provider>
          <ModeContext.Provider value={slot("hardening")}>
            <div hidden={tab !== "hardening"}>
              {hardening ? <Hardening info={hardening} show={show} onRefresh={refreshHardening} replaying={loadingKey} /> : <p className="muted">Loading…</p>}
            </div>
          </ModeContext.Provider>
        </aside>
        <section className="stage" ref={stageRef} aria-label="3D replay">{stage}</section>
      </div>
      <div className="details" ref={setDetails} />
      <footer className="muted small">
        Simulation results only: one simulator, an idealized trunk force, flat ground with a uniform friction coefficient, a
        rigid welded payload box. No claim about the real robot or about conditions other than the ones listed. Hardening
        results are candidates pending the M7 regression gate.
      </footer>
    </div>
  );
}
