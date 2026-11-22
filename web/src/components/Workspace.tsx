// Shared workspace plumbing: every mode renders its controls in the left column and its long
// reports into the details area below the workspace through <Details>, so a mode keeps a single
// component (and a single state) while the 3D stage stays shared.
import { createContext, useContext, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface ModeSlot {
  active: boolean;
  details: HTMLElement | null;
}

export const ModeContext = createContext<ModeSlot>({ active: true, details: null });

export function Details({ children }: { children: ReactNode }) {
  const { active, details } = useContext(ModeContext);
  return active && details ? createPortal(children, details) : null;
}

export function useModeActive(): boolean {
  return useContext(ModeContext).active;
}

export function Panel({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <div className="panel">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {children}
    </div>
  );
}
