import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  COLORS,
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
import { LiveStress } from "./components/LiveStress";
import { ReplayStage } from "./components/ReplayStage";
import { Results } from "./components/Results";
import { ScenarioDesigner } from "./components/ScenarioDesigner";
import { JobContext, ModeContext } from "./components/Workspace";
import { directionWord, outcomeText, robotNames, sourceInfo } from "./replayInfo";

type Tab = "designer" | "live" | "discovery" | "hardening" | "results";
const TABS: { key: Tab; label: string; aliases: string[] }[] = [
  { key: "designer", label: "Design scenario", aliases: ["design", "designer"] },
  { key: "live", label: "Stress test", aliases: ["stress", "live"] },
  { key: "discovery", label: "Discover failure", aliases: ["discover", "discovery"] },
  { key: "hardening", label: "Harden controller", aliases: ["harden", "hardening"] },
  { key: "results", label: "Results", aliases: ["results", "hidden"] },
];

export type ShowReplay = (key: string, title: string, load: () => Promise<Replay>) => Promise<void>;
interface Current {
  data: Replay;
  title: string;
  sample: boolean;
}

export function App() {
  const params = new URLSearchParams(window.location.search);
  const requestedTab = params.get("tab");
  const initialTab = TABS.find((t) => requestedTab !== null && t.aliases.includes(requestedTab))?.key ?? "designer";
  const [tab, setTab] = useState<Tab>(initialTab);
  const [model, setModel] = useState<ModelPayload | null>(null);
  const [hidden, setHidden] = useState<HiddenPhaseData | null>(null);
  const [live, setLive] = useState<LiveStressData | null>(null);
  const [discovery, setDiscovery] = useState<DiscoveryInfo | null>(null);
  const [designer, setDesigner] = useState<ScenarioConfig | null>(null);
  const [hardening, setHardening] = useState<HardeningInfo | null>(null);
  const [replay, setReplay] = useState<Current | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [loadingTitle, setLoadingTitle] = useState<string | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<HTMLElement | null>(null);
  const [jobs, setJobs] = useState<Record<string, string>>({});
  const stageRef = useRef<HTMLElement>(null);
  const requested = useRef(false); // a deep link or a user action asked for a replay

  const reportJob = useCallback((mode: string, label: string | null) => {
    setJobs((prev) => {
      if (label === null) {
        if (!(mode in prev)) return prev;
        const next = { ...prev };
        delete next[mode];
        return next;
      }
      return prev[mode] === label ? prev : { ...prev, [mode]: label };
    });
  }, []);

  const refreshLive = useCallback(() => { api.liveStress().then(setLive).catch((e) => setError(String(e))); }, []);
  const refreshDiscovery = useCallback(() => { api.discoveryInfo().then(setDiscovery).catch((e) => setError(String(e))); }, []);
  const refreshDesigner = useCallback(() => { api.scenarioConfig().then(setDesigner).catch((e) => setError(String(e))); }, []);
  const refreshHardening = useCallback(() => { hardeningApi.info().then(setHardening).catch((e) => setError(String(e))); }, []);

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
    (substep: number) => show(`hidden/${substep}`, `Recorded sweep · 100 N shove at ${(substep * 0.004).toFixed(3)} s`, () => api.hiddenReplay(substep)),
    [show],
  );
  const replayScenario = useCallback(
    (runId: string, trial: number, ghost: "primary" | "matched" | "original", title: string) =>
      show(`scenario/${runId}/${trial}/${ghost}`, title, () => api.scenarioReplay(runId, trial, ghost)),
    [show],
  );
  const replayLive = useCallback(
    (runId: string, trial: number) => show(`${runId}/${trial}`, `Stress test · run ${runId} · trial ${trial}`, () => api.runReplay(runId, trial)),
    [show],
  );
  const replayDiscovery = useCallback(
    (runId: string, which: "final" | "search") =>
      show(`discovery/${runId}/${which}`, `Discovered failure · ${runId}${which === "search" ? " · before shrinking" : ""}`, () => api.discoveryReplay(runId, which)),
    [show],
  );

  useEffect(() => {
    api.hiddenPhase()
      .then((h) => {
        setHidden(h);
        // Sample replay on first load (unless a deep link or the user asked for another one).
        if (!requested.current) {
          const k = h.verified_failure_substep;
          setLoadingKey(`hidden/${k}`);
          setLoadingTitle("Loading the sample replay");
          api.hiddenReplay(k)
            .then((data) => {
              if (!requested.current) setReplay({ data, title: `Sample · 100 N shove at ${(k * 0.004).toFixed(3)} s`, sample: true });
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
    // Deep links: ?replay=<start substep> (recorded sweep), ?scenario_replay=<run>:<trial>:<ghost>,
    // ?frame=<n> (initial frame). Mode components handle ?run=, ?hrun=, ?hreplay=, &dreplay=.
    const substep = params.get("replay");
    if (substep !== null && /^\d+$/.test(substep)) void replayHidden(Number(substep));
    const sr = params.get("scenario_replay")?.match(/^(sc-\d{8}T\d{6}Z-[0-9a-f]{8}):(\d{1,2}):(primary|matched|original)$/);
    if (sr) void replayScenario(sr[1], Number(sr[2]), sr[3] as "primary" | "matched" | "original", `Scenario run ${sr[1]} · trial ${sr[2]} · ghost ${sr[3]}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const navRef = useRef<HTMLElement>(null);
  useEffect(() => {
    // Keep the active tab visible when the navigation scrolls horizontally (narrow screens).
    const nav = navRef.current;
    const active = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (nav && active) nav.scrollTo({ left: active.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2, behavior: "smooth" });
  }, [tab]);
  const hiddenLoading = loadingKey?.startsWith("hidden/") ? Number(loadingKey.split("/")[1]) : null;
  const slot = (key: Tab) => ({ key, active: tab === key, details });
  const jobLabels = Object.values(jobs);
  const busy = loadingTitle ? `${loadingTitle} — simulating on the server` : jobLabels.length ? `${jobLabels.join(" · ")} running` : null;
  const selectTab = (key: Tab) => {
    setTab(key);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", TABS.find((t) => t.key === key)!.aliases[0]);
    window.history.replaceState(null, "", url);
  };

  return (
    <JobContext.Provider value={reportJob}>
      <div className="shell">
        <header className="topbar">
          <div className="topbar-inner">
            <div className="brand">
              <h1>Robustness Studio</h1>
              <span className="tag">Go1 walking policy · native MuJoCo on the server</span>
            </div>
            <div className="nav-scroller">
              <nav className="nav" aria-label="Modes" ref={navRef}>
                {TABS.map((t) => (
                  <button key={t.key} aria-current={tab === t.key ? "page" : undefined} onClick={() => selectTab(t.key)}>
                    {t.label}
                    {jobs[t.key] && <span className="dot" style={{ background: "var(--accent)", marginLeft: 6 }} aria-label="running" />}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </header>
        <main className="page">
          {error && <div className="banner error">{error}</div>}
          <div className="hero">
            <section className="controls-col" aria-label="Experiment controls">
              <ModeContext.Provider value={slot("designer")}>
                <div hidden={tab !== "designer"}>
                  {designer ? <ScenarioDesigner config={designer} onReplay={replayScenario} onRefresh={refreshDesigner} replaying={loadingKey} /> : <p className="muted">Loading…</p>}
                </div>
              </ModeContext.Provider>
              <ModeContext.Provider value={slot("live")}>
                <div hidden={tab !== "live"}>
                  {live ? <LiveStress data={live} onReplay={replayLive} onRefresh={refreshLive} replaying={loadingKey} /> : <p className="muted">Loading…</p>}
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
              <ModeContext.Provider value={slot("results")}>
                <div hidden={tab !== "results"}>
                  {hidden ? (
                    <Results hidden={hidden} live={live} discovery={discovery} designer={designer} hardening={hardening}
                      onHidden={replayHidden} onLive={replayLive} onDiscovery={replayDiscovery} onScenario={replayScenario} show={show}
                      replaying={loadingKey} hiddenLoading={hiddenLoading} />
                  ) : <p className="muted">Loading…</p>}
                </div>
              </ModeContext.Provider>
            </section>
            <section className="stage-col" ref={stageRef} aria-label="3D replay">
              <ReplayStage model={model} current={replay} busy={busy} error={replayError} />
            </section>
          </div>
          {replay && <SummaryStrip current={replay} />}
          <section className="evidence-section" id="evidence" aria-label="Detailed evidence">
            <div className="evidence-head">
              <h2>Evidence · {TABS.find((t) => t.key === tab)!.label}</h2>
              <span className="muted small">Recorded measurements, provenance and reproduce commands</span>
            </div>
            <div className="evidence-body" ref={setDetails} />
          </section>
        </main>
        <footer>
          Simulation results only: one simulator, an idealized trunk force, flat ground with a uniform friction coefficient, a rigid welded
          payload box. No claim about the real robot or about conditions other than the ones listed. Hardening results are candidates pending
          the M7 regression gate.
        </footer>
      </div>
    </JobContext.Provider>
  );
}

function SummaryStrip({ current }: { current: Current }) {
  const r = current.data;
  const names = robotNames(r);
  const src = sourceInfo(r, current.sample);
  const main = r.cases.attacked.outcome;
  const ghost = r.cases.nominal.outcome;
  return (
    <div className="summary-strip" aria-label="What happened">
      <span className="item"><strong>{current.title}</strong></span>
      {r.push && <span className="item">shove <strong>{r.push.force_n.toFixed(0)} N</strong> {directionWord(r.push.world_force_n)} · {r.push.start_s.toFixed(3)}–{r.push.end_s.toFixed(3)} s</span>}
      <span className="item"><span className="dot" style={{ background: COLORS[main.outcome] }} />{names.main}: <strong>{outcomeText(main)}</strong></span>
      <span className="item"><span className="dot" style={{ background: COLORS[ghost.outcome] }} />{names.ghost}: <strong>{outcomeText(ghost)}</strong></span>
      <span className="item">{src.text}</span>
      <span className="spacer" />
      <a href="#evidence" className="small">Evidence ↓</a>
    </div>
  );
}
