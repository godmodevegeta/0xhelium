import { useCallback, useEffect, useRef, useState } from "react";
import { api, type DiscoveryInfo, type HiddenPhase as HiddenPhaseData, type LiveStress as LiveStressData, type ModelPayload, type Replay } from "./api";
import { Discovery } from "./components/Discovery";
import { HiddenPhase } from "./components/HiddenPhase";
import { LiveStress } from "./components/LiveStress";
import { ReplayViewer } from "./components/ReplayViewer";

type Tab = "hidden" | "live" | "discovery";

export function App() {
  const [tab, setTab] = useState<Tab>("hidden");
  const [model, setModel] = useState<ModelPayload | null>(null);
  const [hidden, setHidden] = useState<HiddenPhaseData | null>(null);
  const [live, setLive] = useState<LiveStressData | null>(null);
  const [discovery, setDiscovery] = useState<DiscoveryInfo | null>(null);
  const [replay, setReplay] = useState<{ data: Replay; title: string } | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const replayRef = useRef<HTMLDivElement>(null);

  const refreshLive = useCallback(() => {
    api.liveStress().then(setLive).catch((e) => setError(String(e)));
  }, []);
  const refreshDiscovery = useCallback(() => {
    api.discoveryInfo().then(setDiscovery).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    api.hiddenPhase().then(setHidden).catch((e) => setError(String(e)));
    refreshLive();
    refreshDiscovery();
    api.model().then(setModel).catch((e) => setError(String(e)));
    // Deep links: ?tab=live, ?replay=<start substep> (Preset 1), ?frame=<n> (initial frame).
    const params = new URLSearchParams(window.location.search);
    if (params.get("tab") === "live") setTab("live");
    if (params.get("tab") === "discovery") setTab("discovery");
    const substep = params.get("replay");
    if (substep !== null && /^\d+$/.test(substep)) void replayHidden(Number(substep));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshLive]);

  const show = async (key: string, title: string, load: () => Promise<Replay>) => {
    setLoadingKey(key);
    setError(null);
    try {
      const data = await load();
      setReplay({ data, title });
      requestAnimationFrame(() => replayRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingKey(null);
    }
  };

  const replayHidden = (substep: number) =>
    show(`hidden/${substep}`, `Preset 1 · shove at ${(substep * 0.004).toFixed(3)} s`, () => api.hiddenReplay(substep));
  const replayLive = (runId: string, trial: number) =>
    show(`${runId}/${trial}`, `Preset 2 · run ${runId} · trial ${trial}`, () => api.runReplay(runId, trial));
  const replayDiscovery = (runId: string, which: "final" | "search") =>
    show(`discovery/${runId}/${which}`, `Failure Discovery · ${runId} · ${which === "final" ? "discovered failure" : "confirmed search failure"}`, () => api.discoveryReplay(runId, which));
  const hiddenLoading = loadingKey?.startsWith("hidden/") ? Number(loadingKey.split("/")[1]) : null;

  return (
    <div className="app">
      <header>
        <div>
          <h1>Robustness Studio</h1>
          <div className="muted">Unitree Go1 walking policy · native MuJoCo 3.15 on the server · recorded replays in the browser</div>
        </div>
        <nav>
          <button className={tab === "hidden" ? "tab active" : "tab"} onClick={() => setTab("hidden")}>Hidden Phase Vulnerability</button>
          <button className={tab === "live" ? "tab active" : "tab"} onClick={() => setTab("live")}>Live Stress Test</button>
          <button className={tab === "discovery" ? "tab active" : "tab"} onClick={() => setTab("discovery")}>Failure Discovery</button>
        </nav>
      </header>
      {error && <div className="error banner">{error}</div>}
      {tab === "hidden" && (hidden ? <HiddenPhase data={hidden} onReplay={replayHidden} replaying={hiddenLoading} /> : <p className="muted">Loading preset…</p>)}
      {tab === "live" && (live ? <LiveStress data={live} onReplay={replayLive} onRefresh={refreshLive} replaying={loadingKey} /> : <p className="muted">Loading preset…</p>)}
      {tab === "discovery" && (discovery ? <Discovery info={discovery} onReplay={replayDiscovery} onRefresh={refreshDiscovery} replaying={loadingKey} /> : <p className="muted">Loading…</p>)}
      <div ref={replayRef}>
        {replay && model && <ReplayViewer model={model} replay={replay.data} title={replay.title} />}
        {replay && !model && <p className="muted">Loading robot meshes…</p>}
      </div>
      <footer className="muted small">
        Simulation results only: one simulator, an idealized trunk force, flat ground. No claim about the real robot or about
        disturbances other than the ones listed.
      </footer>
    </div>
  );
}
