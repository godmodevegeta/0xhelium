// One palette for CSS (mirrored as custom properties in styles.css), SVG charts and the Three.js
// scene. Original values chosen for a dark, high-contrast simulation stage.
export const PALETTE = {
  bg: "#0a0a0a",
  surface: "#101010",
  surface2: "#161616",
  surface3: "#1d1d1c",
  border: "#232322",
  borderStrong: "#34332f",
  text: "#ededeb",
  muted: "#a3a19c",
  faint: "#6f6d68",
  accent: "#ff5b35", // attack cue and primary action
  ok: "#d7f25d", // survived
  fell: "#ff4f6e", // physical fall (distinct from the orange attack cue)
  candidate: "#8b93f8", // comparison robot / candidate accent
  warn: "#f2b45a",
  numerical: "#c58af9",
  // Three.js stage
  stageBg: "#0e0e0d",
  ground: "#111110",
  gridMajor: "#2b2a27",
  gridMinor: "#1b1b19",
  robot: "#c9c7c0",
  ghost: "#8b93f8",
  payload: "#f2b45a",
  lowFriction: "#0f2430",
  highFriction: "#2a2014",
} as const;
