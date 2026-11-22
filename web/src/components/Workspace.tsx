// Shared workspace plumbing. Every mode renders its compact controls in the hero's left column and
// its detailed reports into the evidence section through <Details>, so a mode keeps one component
// (and one state) while the 3D stage stays shared and mounted.
import { createContext, useContext, useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface ModeSlot {
  key: string;
  active: boolean;
  details: HTMLElement | null;
}

export const ModeContext = createContext<ModeSlot>({ key: "", active: true, details: null });

export function Details({ children }: { children: ReactNode }) {
  const { active, details } = useContext(ModeContext);
  return active && details ? createPortal(children, details) : null;
}

export function useModeActive(): boolean {
  return useContext(ModeContext).active;
}

/** Modes report a running job (e.g. "CPU hardening, generation 4/12") so the stage can label the
 *  replay it keeps showing as the previous one. */
export const JobContext = createContext<(mode: string, label: string | null) => void>(() => {});

export function useJobReport(running: boolean, label: string) {
  const report = useContext(JobContext);
  const { key } = useContext(ModeContext);
  useEffect(() => {
    report(key, running ? label : null);
  }, [report, key, running, label]);
  useEffect(() => () => report(key, null), [report, key]);
}

export function ModeIntro({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="mode-intro">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {children}
    </div>
  );
}

export function Advanced({ summary, children, open }: { summary: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details className="advanced" open={open}>
      <summary>{summary}</summary>
      <div className="advanced-body">{children}</div>
    </details>
  );
}

export function Evidence({ title, summary, children, open = false }: { title: string; summary?: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details className="card" open={open}>
      <summary>
        {title}
        {summary && <span className="muted">· {summary}</span>}
      </summary>
      {children}
    </details>
  );
}

export function EvidenceLink() {
  return (
    <a className="to-evidence" href="#evidence">
      Detailed evidence below ↓
    </a>
  );
}

/** Backwards-compatible name used by the M6 components. */
export const Panel = ({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) => (
  <div className="mode">
    <ModeIntro eyebrow={eyebrow} title={title} />
    {children}
  </div>
);
