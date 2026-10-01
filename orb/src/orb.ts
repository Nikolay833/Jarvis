import type { OrbState } from "./protocol";

// Accent: ice cyan (matches --accent in style.css).
const ACCENT = "111, 207, 227";
const LIGHT = "228, 246, 250";
const DEEP = "14, 74, 92";

const SIZE = 260; // css px, canvas is square
const R0 = 54; // resting radius -> ~108 px body, up to ~150 px when loud

const TAU = Math.PI * 2;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
/** Frame-rate independent exponential approach. */
const ease = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);

/**
 * Canvas orb. The animation loop only runs while the orb is visible or still
 * fading; when hidden it stops and the canvas is cleared. With
 * prefers-reduced-motion it never loops: it redraws on state/level events and
 * encodes loudness as brightness instead of size.
 */
export class Orb {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly reduceQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  private raf = 0;
  private last = 0;

  private state: OrbState = "idle";
  private forceVisible = false;
  private vis = 0;
  private w = { listening: 0, thinking: 0, speaking: 0 };
  private rawLevel = 0;
  private levelAt = 0;
  private level = 0; // fast envelope
  private echo = 0; // slow envelope (speaking ring)

  constructor(canvas: HTMLCanvasElement) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(SIZE * dpr);
    canvas.height = Math.round(SIZE * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas unavailable");
    ctx.scale(dpr, dpr);
    this.ctx = ctx;
    this.reduceQuery.addEventListener("change", () => this.kick());
  }

  private get reduced(): boolean {
    return this.reduceQuery.matches;
  }

  private get wantVisible(): boolean {
    return this.state !== "idle" || this.forceVisible;
  }

  setState(s: OrbState): void {
    this.state = s;
    if (s === "idle" || s === "thinking") this.rawLevel = 0;
    this.kick();
  }

  /** Keep the orb up while a confirmation is pending even if state says idle. */
  setForceVisible(v: boolean): void {
    this.forceVisible = v;
    this.kick();
  }

  setLevel(rms: number): void {
    if (this.state !== "listening" && this.state !== "speaking") return;
    // Speech sits around 0.02-0.2 rms; lift and soften so it fills the range.
    this.rawLevel = clamp(rms * 4.5) ** 0.7;
    this.levelAt = performance.now();
    if (this.reduced) this.frame(performance.now()); // event-driven redraw
  }

  private kick(): void {
    if (this.reduced) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.vis = this.wantVisible ? 1 : 0;
      this.settleWeights();
      this.draw(performance.now());
      return;
    }
    if (this.raf === 0) {
      this.last = performance.now();
      this.raf = requestAnimationFrame((t) => this.loop(t));
    }
  }

  private loop(t: number): void {
    this.raf = 0;
    this.frame(t);
    const settled = !this.wantVisible && this.vis < 0.002;
    if (settled) {
      this.ctx.clearRect(0, 0, SIZE, SIZE);
      return; // hidden: no loop, no GPU/CPU work
    }
    this.raf = requestAnimationFrame((n) => this.loop(n));
  }

  private frame(t: number): void {
    const dt = Math.min((t - this.last) / 1000 || 0.016, 0.1);
    this.last = t;

    // Visibility and per-state weights ease so looks cross-fade, never pop.
    this.vis += ((this.wantVisible ? 1 : 0) - this.vis) * ease(dt, this.wantVisible ? 0.16 : 0.22);
    const k = ease(dt, 0.14);
    this.w.listening += ((this.state === "listening" ? 1 : 0) - this.w.listening) * k;
    this.w.thinking += ((this.state === "thinking" ? 1 : 0) - this.w.thinking) * k;
    this.w.speaking += ((this.state === "speaking" ? 1 : 0) - this.w.speaking) * k;

    // Level: fast attack, slow release. Stale levels decay to zero.
    const target = t - this.levelAt > 250 ? 0 : this.rawLevel;
    this.level += (target - this.level) * ease(dt, target > this.level ? 0.05 : 0.2);
    this.echo += (this.level - this.echo) * ease(dt, 0.35);

    this.draw(t);
  }

  private settleWeights(): void {
    this.w.listening = this.state === "listening" ? 1 : 0;
    this.w.thinking = this.state === "thinking" ? 1 : 0;
    this.w.speaking = this.state === "speaking" ? 1 : 0;
  }

  private draw(t: number): void {
    const { ctx, w } = this;
    const reduced = this.reduced;
    ctx.clearRect(0, 0, SIZE, SIZE);
    if (this.vis < 0.002) return;

    const L = this.level;
    const E = this.echo;
    const breathe = reduced ? 0 : Math.sin((t / 1000) * 1.1 * TAU * 0.5);

    // Blend per-state parameters by weight.
    const wl = w.listening, wt = w.thinking, ws = w.speaking;
    const pulse = reduced ? 0 : 1;
    const radiusScale =
      wl * (1 + 0.22 * L * pulse) +
      wt * (0.93 + 0.02 * breathe) +
      ws * (1 + 0.15 * L * pulse) +
      (1 - wl - wt - ws) * 1;
    const glow = wl * (0.5 + 0.5 * L) + wt * 0.38 + ws * (0.55 + 0.45 * L);
    const white = wl * 0.5 + wt * 0.3 + ws * 0.78 + (reduced ? 0.25 * L : 0);

    const entry = reduced ? 1 : 0.82 + 0.18 * this.vis;
    const c = SIZE / 2;
    const r = R0 * radiusScale * entry;
    const a = this.vis;

    // 1. soft halo
    const halo = ctx.createRadialGradient(c, c, r * 0.7, c, c, r * 2.3);
    halo.addColorStop(0, `rgba(${ACCENT}, ${0.26 * glow * a})`);
    halo.addColorStop(1, `rgba(${ACCENT}, 0)`);
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, SIZE, SIZE);

    // 2. body
    const body = ctx.createRadialGradient(c - r * 0.28, c - r * 0.34, r * 0.05, c, c, r);
    body.addColorStop(0, `rgba(${LIGHT}, ${(0.55 + 0.4 * white) * a})`);
    body.addColorStop(0.4, `rgba(${ACCENT}, ${(0.5 + 0.3 * white) * a})`);
    body.addColorStop(0.85, `rgba(${DEEP}, ${0.7 * a})`);
    body.addColorStop(1, `rgba(${ACCENT}, ${0.55 * a})`);
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(c, c, r, 0, TAU);
    ctx.fill();

    // 3. hairline rim
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(${LIGHT}, ${0.35 * a})`;
    ctx.beginPath();
    ctx.arc(c, c, r - 0.5, 0, TAU);
    ctx.stroke();

    // 4. thinking: slow orbiting shimmer
    if (wt > 0.01) {
      const orbitR = r * 1.3;
      const angle = reduced ? -0.8 : (t / 3400) * TAU;
      this.comet(c, orbitR, angle, 0.85 * wt * a);
      this.comet(c, orbitR, angle + Math.PI, 0.35 * wt * a);
    }

    // 5. speaking: echo ring that trails the voice
    if (ws > 0.01) {
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = `rgba(${ACCENT}, ${(0.12 + 0.4 * E) * ws * a})`;
      ctx.beginPath();
      ctx.arc(c, c, r * (1.16 + 0.22 * E * pulse), 0, TAU);
      ctx.stroke();
    }
  }

  /** A short fading arc with a bright head, orbiting at radius `rad`. */
  private comet(c: number, rad: number, angle: number, alpha: number): void {
    const { ctx } = this;
    if (alpha < 0.005) return;
    const sweep = 0.22; // fraction of the circle covered by the tail
    if (typeof ctx.createConicGradient === "function") {
      const g = ctx.createConicGradient(angle - sweep * TAU, c, c);
      g.addColorStop(0, `rgba(${ACCENT}, 0)`);
      g.addColorStop(sweep, `rgba(${LIGHT}, ${alpha})`);
      g.addColorStop(sweep + 0.001, `rgba(${ACCENT}, 0)`);
      g.addColorStop(1, `rgba(${ACCENT}, 0)`);
      ctx.strokeStyle = g;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.arc(c, c, rad, 0, TAU);
      ctx.stroke();
    }
    const hx = c + Math.cos(angle) * rad;
    const hy = c + Math.sin(angle) * rad;
    const dot = ctx.createRadialGradient(hx, hy, 0, hx, hy, 7);
    dot.addColorStop(0, `rgba(${LIGHT}, ${alpha})`);
    dot.addColorStop(1, `rgba(${ACCENT}, 0)`);
    ctx.fillStyle = dot;
    ctx.fillRect(hx - 7, hy - 7, 14, 14);
  }
}
