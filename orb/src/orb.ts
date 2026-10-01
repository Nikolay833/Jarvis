import type { OrbState } from "./protocol";

// Look: one dense, grainy sphere of cyan light particles. Density is modulated
// by 3D noise so the surface shows dark gaps and bright clusters.
const HALO = "63, 216, 245";

const SIZE = 260; // css px, canvas is square
const R0 = 60; // resting sphere radius in css px (loud: wobbles out to ~78)
const N_PARTICLES = 6500;
const N_PARTICLES_2D = 1700;

const TAU = Math.PI * 2;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
/** Frame-rate independent exponential approach. */
const ease = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);

/** Everything a renderer needs for one frame. */
interface Frame {
  t: number; // seconds, drives twinkle / ripples (frozen when reduced motion)
  flow: number; // noise time
  rot: number; // y rotation, radians
  amp: number; // deformation amplitude (fraction of radius)
  rip: number; // speaking ripple amplitude
  think: number; // thinking weight 0..1
  band: number; // shimmer sweep phase
  bright: number; // overall brightness
  radius: number; // css px
  vis: number; // fade 0..1
  level: number; // audio level 0..1 (slight swell)
}

interface Renderer {
  render(f: Frame): void;
  clear(): void;
}

// ---------------------------------------------------------------------------
// Particle cloud

function hash3(x: number, y: number, z: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return h - Math.floor(h);
}

/** Smooth value noise in [0, 1]. */
function vnoise(x: number, y: number, z: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = x - xi, fy = y - yi, fz = z - zi;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), w = fz * fz * (3 - 2 * fz);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const c = (dx: number, dy: number, dz: number) => hash3(xi + dx, yi + dy, zi + dz);
  return l(
    l(l(c(0, 0, 0), c(1, 0, 0), u), l(c(0, 1, 0), c(1, 1, 0), u), v),
    l(l(c(0, 0, 1), c(1, 0, 1), u), l(c(0, 1, 1), c(1, 1, 1), u), v),
    w,
  );
}

/**
 * Random particles, mostly in a thin fuzzy shell plus some volume fill.
 * Candidates are accepted by noise-modulated probability, so clumps and gaps
 * are baked into the positions and rotate with the sphere. [x, y, z, rand].
 */
function particles(n: number): Float32Array {
  const out = new Float32Array(n * 4);
  const ox = Math.random() * 50, oy = Math.random() * 50, oz = Math.random() * 50;
  let i = 0;
  while (i < n) {
    const u = Math.random() * 2 - 1;
    const th = Math.random() * TAU;
    const s = Math.sqrt(1 - u * u);
    const dx = Math.cos(th) * s, dy = u, dz = Math.sin(th) * s;
    const dens = 0.6 * vnoise(dx * 3.2 + ox, dy * 3.2 + oy, dz * 3.2 + oz) + 0.4 * vnoise(dx * 7 + oy, dy * 7 + oz, dz * 7 + ox);
    if (Math.random() > clamp((dens - 0.2) * 3.0, 0.08, 1)) continue;
    const k = Math.random();
    let r: number;
    if (k < 0.1) r = Math.cbrt(Math.random()) * 0.85; // volume fill
    else if (k < 0.15) r = 1.03 + Math.random() * 0.1; // stray particles outside
    else r = 1.03 - 0.18 * Math.pow(Math.random(), 1.7); // shell
    out[i * 4] = dx * r;
    out[i * 4 + 1] = dy * r;
    out[i * 4 + 2] = dz * r;
    out[i * 4 + 3] = Math.random();
    i++;
  }
  return out;
}

// ---------------------------------------------------------------------------
// WebGL renderer: one draw call per frame.

const VERT = /* glsl */ `
attribute vec4 aP;
uniform float uT, uFlow, uRot, uAmp, uRip, uThink, uBand, uBright, uR, uDpr, uHalf, uLvl;
varying vec3 vCol;
varying float vA;

vec3 mod289(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 mod289(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 permute(vec4 x){return mod289(((x*34.)+1.)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1./6.,1./3.); const vec4 D=vec4(0.,.5,1.,2.);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.,i1.z,i2.z,1.))+i.y+vec4(0.,i1.y,i2.y,1.))+i.x+vec4(0.,i1.x,i2.x,1.));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.+1.; vec4 s1=floor(b1)*2.+1.; vec4 sh=-step(h,vec4(0.));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.); m=m*m;
  return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

vec3 rotY(vec3 v, float a){ float c=cos(a), s=sin(a); return vec3(c*v.x+s*v.z, v.y, -s*v.x+c*v.z); }
vec3 view(vec3 v){
  v = rotY(v, uRot);
  const float ct = 0.93, st = 0.37; // fixed axial tilt
  return vec3(v.x, ct*v.y - st*v.z, st*v.y + ct*v.z);
}

const vec3 TEAL = vec3(0.14, 0.62, 0.74);
const vec3 CYAN = vec3(0.18, 0.84, 0.96);   // ~#2fd6f5
const vec3 ICE  = vec3(0.48, 0.91, 1.0);    // ~#7ae8ff
const vec3 SPEC = vec3(0.78, 0.97, 1.0);
const float CAM = 3.4;

void main(){
  vec3 p = aP.xyz;
  float rnd = aP.w;
  float r2 = fract(rnd * 91.7);
  float r3 = fract(rnd * 37.3);
  vec3 dir = normalize(p + 1e-4);
  float rl = length(p);

  float n  = snoise(dir * 1.05 + vec3(0., 0., uFlow));
  float n2 = snoise(dir * 2.1 + vec3(uFlow * 1.7, 3.1, 0.));
  float d = uAmp * (0.75 * n + 0.25 * n2);
  d += uRip * sin(acos(clamp(dir.y, -1., 1.)) * 8.0 - uT * 5.0) * (0.55 + 0.45 * n);
  // tiny per-particle shimmer so the surface feels alive without moving the clumps
  vec3 jit = vec3(sin(uT * 0.7 + rnd * 60.), cos(uT * 0.6 + rnd * 90.), sin(uT * 0.8 + rnd * 40.)) * 0.006;
  vec3 pos = view((p + jit) * (1.0 + d + uLvl * 0.035));

  float f = view(dir).z;                       // + toward viewer
  float rim = 1.0 - abs(f);
  float inten = (0.7 + 0.7 * r3) * (0.7 + 0.7 * pow(rim, 2.0));
  if (f < 0.0) inten *= 0.6;
  inten *= mix(0.55, 1.0, smoothstep(0.5, 1.0, rl));  // volume fill is dimmer

  // thinking: soft brightness wave sweeping over the particles
  float bp = fract(uBand) * 3.2 - 1.6;
  float b = exp(-pow((pos.y + 0.45 * pos.x - bp) * 3.0, 2.0)) * uThink;
  inten += b * 0.55;

  vec3 col = mix(TEAL, CYAN, smoothstep(0.0, 0.5, r2));
  col = mix(col, ICE, smoothstep(0.55, 1.0, r2) * 0.8);
  if (rnd > 0.965) { col = SPEC; inten *= 1.15; }       // few near-white specks
  col = mix(col, ICE, clamp(b * 0.6, 0., 1.));

  float size = 1.45 + 1.4 * r3 * r3 + (rnd > 0.985 ? 1.1 : 0.0) + 0.25 * rim;
  float persp = CAM / (CAM - pos.z);
  gl_Position = vec4(pos.xy * persp * uR / uHalf, 0., 1.);
  gl_PointSize = max(size * persp * uDpr, 1.0);
  vCol = col;
  vA = inten * uBright;
}
`;

const FRAG = /* glsl */ `
precision mediump float;
varying vec3 vCol;
varying float vA;
void main(){
  vec2 q = gl_PointCoord - 0.5;
  float d = length(q) * 2.0;
  float s = (1.0 - smoothstep(0.35, 1.0, d));
  vec3 c = vCol * (s * vA);
  c = 1.0 - exp(-c * 1.7);   // soft tone curve: dense clusters stay cyan instead of clipping white
  // Premultiplied output: alpha must be >= every channel so additive blending
  // on a transparent window never turns grey at the edges.
  gl_FragColor = vec4(c, max(c.r, max(c.g, c.b)));
}
`;

class GLRenderer implements Renderer {
  private readonly gl: WebGLRenderingContext;
  private prog!: WebGLProgram;
  private buf!: WebGLBuffer;
  private loc!: { a: number; u: Record<string, WebGLUniformLocation | null> };
  private readonly data = particles(N_PARTICLES);
  private lost = false;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly dpr: number) {
    const gl = canvas.getContext("webgl", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: "low-power",
    });
    if (!gl) throw new Error("webgl unavailable");
    this.gl = gl;
    this.init();
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.lost = true;
    });
    canvas.addEventListener("webglcontextrestored", () => {
      this.init();
      this.lost = false;
    });
  }

  private compile(type: number, src: string): WebGLShader {
    const { gl } = this;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  private init(): void {
    const { gl } = this;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, this.compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, this.compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
    this.prog = prog;
    const u: Record<string, WebGLUniformLocation | null> = {};
    for (const n of ["uT", "uFlow", "uRot", "uAmp", "uRip", "uThink", "uBand", "uBright", "uR", "uDpr", "uHalf", "uLvl"]) {
      u[n] = gl.getUniformLocation(prog, n);
    }
    this.loc = { a: gl.getAttribLocation(prog, "aP"), u };
    this.buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.data, gl.STATIC_DRAW);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // additive, premultiplied
    gl.clearColor(0, 0, 0, 0);
  }

  clear(): void {
    if (this.lost) return;
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  render(f: Frame): void {
    if (this.lost) return;
    const { gl, loc } = this;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    const u = loc.u;
    gl.uniform1f(u.uT, f.t);
    gl.uniform1f(u.uFlow, f.flow);
    gl.uniform1f(u.uRot, f.rot);
    gl.uniform1f(u.uAmp, f.amp);
    gl.uniform1f(u.uRip, f.rip);
    gl.uniform1f(u.uThink, f.think);
    gl.uniform1f(u.uBand, f.band);
    gl.uniform1f(u.uBright, f.bright * f.vis);
    gl.uniform1f(u.uR, f.radius);
    gl.uniform1f(u.uDpr, this.dpr);
    gl.uniform1f(u.uHalf, SIZE / 2);
    gl.uniform1f(u.uLvl, f.level);
    gl.enableVertexAttribArray(loc.a);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.vertexAttribPointer(loc.a, 4, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, N_PARTICLES);
  }
}

// ---------------------------------------------------------------------------
// Canvas 2D fallback: same look, fewer particles, cheap sine "noise".

class Canvas2DRenderer implements Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly data = particles(N_PARTICLES_2D);

  constructor(canvas: HTMLCanvasElement, dpr: number) {
    canvas.width = Math.round(SIZE * dpr);
    canvas.height = Math.round(SIZE * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas unavailable");
    ctx.scale(dpr, dpr);
    this.ctx = ctx;
  }

  clear(): void {
    this.ctx.clearRect(0, 0, SIZE, SIZE);
  }

  private static noise(x: number, y: number, z: number, t: number): number {
    return (
      0.6 * Math.sin(x * 2.3 + t) * Math.cos(y * 2.1 - t * 0.8 + z) +
      0.4 * Math.sin(z * 3.1 - t * 1.3 + x) * Math.cos(y * 2.7 + t * 0.6)
    );
  }

  private proj(x: number, y: number, z: number, f: Frame): [number, number, number, number] {
    const c = Math.cos(f.rot), s = Math.sin(f.rot);
    const x1 = c * x + s * z, z1 = -s * x + c * z;
    const y2 = 0.93 * y - 0.37 * z1, z2 = 0.37 * y + 0.93 * z1;
    const persp = 3.4 / (3.4 - z2);
    return [SIZE / 2 + x1 * persp * f.radius, SIZE / 2 - y2 * persp * f.radius, z2, persp];
  }

  render(f: Frame): void {
    const { ctx } = this;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.globalCompositeOperation = "lighter";
    const A = f.bright * f.vis;
    const bp = (f.band % 1) * 3.2 - 1.6;
    for (let i = 0; i < N_PARTICLES_2D; i++) {
      const o = i * 4;
      const x = this.data[o], y = this.data[o + 1], z = this.data[o + 2], rnd = this.data[o + 3];
      const rl = Math.hypot(x, y, z);
      const dx = x / rl, dy = y / rl, dz = z / rl;
      const d = f.amp * Canvas2DRenderer.noise(dx, dy, dz, f.flow * 2) + f.rip * Math.sin(Math.acos(dy) * 8 - f.t * 5);
      const k = 1 + d + f.level * 0.035;
      const nz = this.proj(dx, dy, dz, f)[2];
      const [px, py, , persp] = this.proj(x * k, y * k, z * k, f);
      const rim = 1 - Math.abs(nz);
      const r3 = (rnd * 37.3) % 1;
      let a = (0.7 + 0.7 * r3) * (0.7 + 0.7 * rim ** 2);
      if (nz < 0) a *= 0.6;
      if (rl < 0.85) a *= 0.55;
      a += f.think * 0.55 * Math.exp(-(((SIZE / 2 - py) / f.radius + 0.45 * ((px - SIZE / 2) / f.radius) - bp) ** 2) * 9);
      const s = (1.7 + 1.6 * r3 * r3) * persp;
      ctx.fillStyle = rnd > 0.965 ? `rgba(199,247,255,${clamp(a * A)})` : `rgba(${r3 > 0.6 ? "122,232,255" : "47,214,245"},${clamp(a * A * 0.9)})`;
      ctx.fillRect(px - s / 2, py - s / 2, s, s);
    }
    ctx.globalCompositeOperation = "source-over";
  }
}

function makeRenderer(canvas: HTMLCanvasElement, dpr: number): { r: Renderer; canvas: HTMLCanvasElement } {
  canvas.width = Math.round(SIZE * dpr);
  canvas.height = Math.round(SIZE * dpr);
  try {
    return { r: new GLRenderer(canvas, dpr), canvas };
  } catch {
    // A canvas that already handed out a webgl context can't give a 2d one: swap it.
    const fresh = canvas.cloneNode(false) as HTMLCanvasElement;
    canvas.replaceWith(fresh);
    return { r: new Canvas2DRenderer(fresh, dpr), canvas: fresh };
  }
}

// ---------------------------------------------------------------------------

/**
 * Point-cloud orb. The animation loop only runs while the orb is visible or
 * still fading; when hidden it stops and the canvas is cleared. With
 * prefers-reduced-motion it never loops: it redraws on state/level events and
 * encodes loudness as brightness instead of motion.
 */
export class Orb {
  private readonly canvas: HTMLCanvasElement;
  private readonly renderer: Renderer;
  private readonly reduceQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  private raf = 0;
  private last = 0;
  private haloKey = "";

  private state: OrbState = "idle";
  private forceVisible = false;
  private vis = 0;
  private w = { listening: 0, thinking: 0, speaking: 0 };
  private rawLevel = 0;
  private levelAt = 0;
  private level = 0; // fast envelope
  private echo = 0; // slow envelope
  private rot = 0.6;
  private flow = 0;
  private time = 0;
  private bandPhase = 0;

  constructor(canvas: HTMLCanvasElement) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const made = makeRenderer(canvas, dpr);
    this.renderer = made.r;
    this.canvas = made.canvas;
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
      this.draw();
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
      this.renderer.clear();
      this.setHalo(0, 0);
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

    if (!this.reduced) {
      const wt = this.w.thinking;
      this.time += dt;
      this.rot += dt * (0.2 + 0.55 * wt + 0.12 * this.level);
      this.flow += dt * (0.3 + 0.55 * this.level + 0.15 * this.w.speaking);
      this.bandPhase += dt * 0.55 * wt;
    }

    this.draw();
  }

  private settleWeights(): void {
    this.w.listening = this.state === "listening" ? 1 : 0;
    this.w.thinking = this.state === "thinking" ? 1 : 0;
    this.w.speaking = this.state === "speaking" ? 1 : 0;
  }

  private draw(): void {
    const { w } = this;
    const reduced = this.reduced;
    if (this.vis < 0.002) {
      this.renderer.clear();
      this.setHalo(0, 0);
      return;
    }

    const L = this.level;
    const wl = w.listening, wt = w.thinking, ws = w.speaking;
    const idle = 1 - wl - wt - ws;

    // Quiet = near-perfect sphere, loud = organic blob. Thinking stays smooth.
    const amp = reduced ? 0 : 0.012 + (wl * 0.3 + ws * 0.27) * L + idle * 0.01;
    const rip = reduced ? 0 : ws * (0.012 + 0.07 * L);
    const bright = reduced
      ? 0.62 + 0.55 * L
      : wl * (0.7 + 0.3 * L) + wt * 0.72 + ws * (0.86 + 0.3 * L) + idle * 0.7;
    const radScale = reduced
      ? 1
      : wl * (1 + 0.05 * L) + wt * 0.94 + ws * (1 + 0.04 * L) + idle;
    const entry = reduced ? 1 : 0.84 + 0.16 * this.vis;

    const frame: Frame = {
      t: reduced ? 3 : this.time,
      flow: reduced ? 1.3 : this.flow,
      rot: this.rot,
      amp,
      rip,
      think: reduced ? wt * 0.5 : wt,
      band: reduced ? 0.55 : this.bandPhase,
      bright,
      radius: R0 * radScale * entry,
      vis: this.vis,
      level: reduced ? 0 : L,
    };
    this.renderer.render(frame);
    this.setHalo(this.vis * (0.1 + 0.07 * bright + (reduced ? 0 : 0.05 * L)), frame.radius);
  }

  /** Soft outer glow as a CSS gradient behind the points (cheap, resolution independent). */
  private setHalo(a: number, r: number): void {
    const key = a < 0.002 ? "off" : `${Math.round(a * 200)}|${Math.round(r)}`;
    if (key === this.haloKey) return;
    this.haloKey = key;
    this.canvas.style.background =
      key === "off"
        ? "none"
        : `radial-gradient(circle at 50% 50%, rgba(${HALO}, ${(a * 0.55).toFixed(3)}) 0px, ` +
          `rgba(${HALO}, ${(a * 0.8).toFixed(3)}) ${(r * 0.92).toFixed(0)}px, ` +
          `rgba(${HALO}, ${(a * 0.32).toFixed(3)}) ${(r * 1.3).toFixed(0)}px, ` +
          `rgba(${HALO}, ${(a * 0.08).toFixed(3)}) ${(r * 1.7).toFixed(0)}px, rgba(${HALO}, 0) ${(r * 2.0).toFixed(0)}px)`;
  }
}
