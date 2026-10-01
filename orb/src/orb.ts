import type { OrbState } from "./protocol";

// Palette: electric blue core, cyan highlights (matches --accent in style.css).
const CORE = [0.165, 0.482, 1.0]; // #2a7bff
const CYAN = [0.373, 0.831, 1.0]; // #5fd4ff
const HALO = "42, 123, 255";

const SIZE = 320; // css px, canvas is square
const R0 = 78; // resting sphere radius in css px (loud: wobbles out to ~100)
const N_SURFACE = 3200;
const N_INNER = 460;
const N_SURFACE_2D = 900;
const N_INNER_2D = 150;

const TAU = Math.PI * 2;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
/** Frame-rate independent exponential approach. */
const ease = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);
const glf = (n: number) => n.toFixed(4);

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
}

interface Renderer {
  render(f: Frame): void;
  clear(): void;
}

// ---------------------------------------------------------------------------
// Point clouds

/** Fibonacci sphere: [x, y, z, rand] per point. */
function fibonacci(n: number): Float32Array {
  const out = new Float32Array(n * 4);
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - ((i + 0.5) * 2) / n;
    const r = Math.sqrt(1 - y * y);
    const th = i * ga;
    out[i * 4] = Math.cos(th) * r;
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = Math.sin(th) * r;
    out[i * 4 + 3] = Math.random();
  }
  return out;
}

/** Uniform random points in a ball: [x, y, z, rand]. */
function ball(n: number): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 2 - 1;
    const th = Math.random() * TAU;
    const s = Math.sqrt(1 - u * u);
    const r = Math.cbrt(Math.random()) * 0.94;
    out[i * 4] = Math.cos(th) * s * r;
    out[i * 4 + 1] = u * r;
    out[i * 4 + 2] = Math.sin(th) * s * r;
    out[i * 4 + 3] = Math.random();
  }
  return out;
}

// ---------------------------------------------------------------------------
// WebGL renderer: two draws per frame (surface dots, inner sparkles).

const VERT = /* glsl */ `
attribute vec4 aP;
uniform float uT, uFlow, uRot, uAmp, uRip, uThink, uBand, uBright, uR, uDpr, uHalf, uInner;
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

const vec3 CORE = vec3(${CORE.map(glf).join(",")});
const vec3 CYAN = vec3(${CYAN.map(glf).join(",")});
const vec3 WHITE = vec3(0.86, 0.97, 1.0);
const float CAM = 3.4;

void main(){
  vec3 p = aP.xyz;
  float rnd = aP.w;
  float r2 = fract(rnd * 91.7);
  float r3 = fract(rnd * 37.3);
  vec3 pos; float size; float inten; vec3 col;

  if (uInner < 0.5) {
    // ---- surface dots
    bool streak = rnd < 0.09;
    vec3 q = p;
    if (streak) q = rotY(p, uT * 0.22 * (0.4 + r2));   // wisps slide over the surface
    float n  = snoise(q * 1.05 + vec3(0., 0., uFlow));
    float n2 = snoise(q * 2.1 + vec3(uFlow * 1.7, 3.1, 0.));
    float d = uAmp * (0.75 * n + 0.25 * n2);
    d += uRip * sin(acos(clamp(q.y, -1., 1.)) * 8.0 - uT * 5.0) * (0.55 + 0.45 * n);
    float rad = 1.0 + d + (streak ? 0.012 : 0.0);
    pos = view(q * rad);
    vec3 nrm = view(q);
    float f = nrm.z;                       // + toward viewer
    float rim = 1.0 - abs(f);
    float crest = smoothstep(0.02, 0.3, d);

    inten = 0.55 + 1.0 * pow(rim, 2.2) + 0.4 * crest;
    if (f < 0.0) inten *= 0.42;           // far side reads as a faint see-through lattice
    size = 2.0 + 1.0 * rim + 0.6 * crest;
    if (streak) { inten = pow(rim, 5.0) * 2.4; size *= 2.3; }
    size *= 0.85 + 0.35 * r3;

    // thinking: bright band sweeping across the dots
    float bp = fract(uBand) * 3.2 - 1.6;
    float b = exp(-pow((pos.y + 0.45 * pos.x - bp) * 3.4, 2.0)) * uThink;
    inten += b * 1.7;

    col = mix(CORE, CYAN, clamp(rim * 1.25 + 0.22 * n, 0., 1.));
    col = mix(col, WHITE, clamp(b * 0.9 + crest * 0.35 + pow(rim, 7.0) * 0.5, 0., 1.));
  } else {
    // ---- inner sparkles
    float rad = length(p);
    vec3 dir = p / max(rad, 1e-4);
    float prog = fract(rnd * 7.0 - uT * 0.16);     // inward travel while thinking
    float k = mix(1.0, 1.0 - prog * 0.88, uThink);
    vec3 pp = p * k;
    float ang = uThink * prog * 4.5 * (1.3 - rad);   // swirl, tighter near the core
    pp = rotY(pp, ang);
    pp += vec3(sin(uT * 0.31 + rnd * 40.), cos(uT * 0.27 + rnd * 70.), sin(uT * 0.33 + rnd * 23.)) * 0.045;
    float nn = snoise(dir * 1.35 + vec3(0., 0., uFlow));
    pp *= 1.0 + uAmp * 0.55 * nn;
    pos = view(pp);
    float tw = pow(0.5 + 0.5 * sin(uT * (1.2 + 3.0 * r2) + rnd * 80.), 4.0);
    float life = mix(1.0, sin(3.14159 * prog), uThink);
    inten = (0.22 + 0.95 * tw) * life;
    if (pos.z < 0.0) inten *= 0.6;
    size = (1.3 + 2.3 * r3 * r3) * (0.65 + 0.55 * tw);
    col = mix(vec3(0.2, 0.55, 1.0), vec3(0.72, 0.94, 1.0), clamp(tw * 0.75 + r3 * 0.2, 0., 1.));
  }

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
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float s = exp(-d * d * 4.2) * (1.0 - smoothstep(0.85, 1.0, d));
  vec3 c = vCol * (s * vA);
  // Premultiplied output: alpha must be >= every channel so additive blending
  // on a transparent window never turns grey at the edges.
  gl_FragColor = vec4(c, max(c.r, max(c.g, c.b)));
}
`;

class GLRenderer implements Renderer {
  private readonly gl: WebGLRenderingContext;
  private prog!: WebGLProgram;
  private surf!: WebGLBuffer;
  private inner!: WebGLBuffer;
  private loc!: { a: number; u: Record<string, WebGLUniformLocation | null> };
  private readonly surfData = fibonacci(N_SURFACE);
  private readonly innerData = ball(N_INNER);
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
    for (const n of ["uT", "uFlow", "uRot", "uAmp", "uRip", "uThink", "uBand", "uBright", "uR", "uDpr", "uHalf", "uInner"]) {
      u[n] = gl.getUniformLocation(prog, n);
    }
    this.loc = { a: gl.getAttribLocation(prog, "aP"), u };
    this.surf = this.buffer(this.surfData);
    this.inner = this.buffer(this.innerData);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // additive, premultiplied
    gl.clearColor(0, 0, 0, 0);
  }

  private buffer(data: Float32Array): WebGLBuffer {
    const { gl } = this;
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
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
    gl.enableVertexAttribArray(loc.a);

    gl.uniform1f(u.uInner, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.surf);
    gl.vertexAttribPointer(loc.a, 4, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, N_SURFACE);

    gl.uniform1f(u.uInner, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inner);
    gl.vertexAttribPointer(loc.a, 4, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, N_INNER);
  }
}

// ---------------------------------------------------------------------------
// Canvas 2D fallback: same look, fewer dots, cheap sine "noise".

class Canvas2DRenderer implements Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly surf = fibonacci(N_SURFACE_2D);
  private readonly inner = ball(N_INNER_2D);

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
    for (let i = 0; i < N_SURFACE_2D; i++) {
      const x = this.surf[i * 4], y = this.surf[i * 4 + 1], z = this.surf[i * 4 + 2];
      const n = Canvas2DRenderer.noise(x, y, z, f.flow * 2);
      const d = f.amp * n + f.rip * Math.sin(Math.acos(y) * 8 - f.t * 5);
      const k = 1 + d;
      const nz = this.proj(x, y, z, f)[2];
      const [px, py, , persp] = this.proj(x * k, y * k, z * k, f);
      const rim = 1 - Math.abs(nz);
      let a = 0.35 + 0.9 * rim ** 2.4;
      if (nz < 0) a *= 0.42;
      const bp = (f.band % 1) * 3.2 - 1.6;
      a += f.think * 1.5 * Math.exp(-(((py - SIZE / 2) / -f.radius + 0.45 * ((px - SIZE / 2) / f.radius) - bp) ** 2) * 11);
      const s = (1.9 + rim) * persp;
      ctx.fillStyle = `rgba(${rim > 0.6 ? "95,212,255" : "42,123,255"},${clamp(a * A * 0.8)})`;
      ctx.fillRect(px - s / 2, py - s / 2, s, s);
    }
    for (let i = 0; i < N_INNER_2D; i++) {
      const x = this.inner[i * 4], y = this.inner[i * 4 + 1], z = this.inner[i * 4 + 2], r = this.inner[i * 4 + 3];
      const tw = (0.5 + 0.5 * Math.sin(f.t * (1.2 + 3 * ((r * 91.7) % 1)) + r * 80)) ** 4;
      const [px, py, , persp] = this.proj(x, y, z, f);
      const s = (1.4 + 2 * ((r * 37.3) % 1) ** 2) * persp;
      ctx.fillStyle = `rgba(150,215,255,${clamp((0.2 + 0.9 * tw) * A)})`;
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
    };
    this.renderer.render(frame);
    this.setHalo(this.vis * (0.2 + 0.12 * bright + (reduced ? 0 : 0.08 * L)), frame.radius);
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
