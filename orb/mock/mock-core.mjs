// Dependency-free stand-in for the Python core, for developing/testing the orb.
//   node mock/mock-core.mjs          -> ws://127.0.0.1:8765, plays the demo loop
//   node mock/mock-core.mjs --manual -> no demo; POST/GET http://127.0.0.1:8766/send?m=<json> broadcasts
//   node mock/mock-core.mjs --map    -> like --manual, and sends the Sofia map scenario to every new client
//   GET http://127.0.0.1:8766/map[?mode=all|none|transit|london] broadcasts it; GET /maphide hides the map
import http from "node:http";
import crypto from "node:crypto";

const mapMode = process.argv.includes("--map");
const manual = process.argv.includes("--manual") || mapMode;
const clients = new Set();

function frame(text) {
  const p = Buffer.from(text);
  const h = p.length < 126 ? Buffer.from([0x81, p.length]) : Buffer.from([0x81, 126, p.length >> 8, p.length & 255]);
  return Buffer.concat([h, p]);
}
const broadcast = (o) => { const f = frame(JSON.stringify(o)); for (const s of clients) s.write(f); };

function readFrames(sock, buf) {
  // minimal parser for small masked client text frames
  while (buf.length >= 2) {
    let len = buf[1] & 127, off = 2;
    if (len === 126) { len = buf.readUInt16BE(2); off = 4; }
    const total = off + 4 + len;
    if (buf.length < total) break;
    const mask = buf.subarray(off, off + 4);
    const data = Buffer.from(buf.subarray(off + 4, total)).map((b, i) => b ^ mask[i % 4]);
    if ((buf[0] & 15) === 1) console.log("orb ->", data.toString());
    buf = buf.subarray(total);
  }
  return buf;
}

const ws = http.createServer();
ws.on("upgrade", (req, sock) => {
  const key = req.headers["sec-websocket-key"];
  const acc = crypto.createHash("sha1").update(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest("base64");
  sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${acc}\r\n\r\n`);
  clients.add(sock);
  if (mapMode) setTimeout(() => sock.write(frame(JSON.stringify(mapShow()))), 600);
  let buf = Buffer.alloc(0);
  sock.on("data", (d) => { buf = readFrames(sock, Buffer.concat([buf, d])); });
  sock.on("close", () => clients.delete(sock));
  sock.on("error", () => clients.delete(sock));
});
ws.listen(8765, "127.0.0.1", () => console.log("mock core on ws://127.0.0.1:8765"));

http.createServer((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname === "/map") broadcast(mapShow(u.searchParams.get("mode") ?? "all"));
  if (u.pathname === "/maphide") broadcast({ type: "map_hide" });
  if (u.pathname === "/send") { try { broadcast(JSON.parse(u.searchParams.get("m"))); } catch { /* ignore */ } }
  res.end("ok");
}).listen(8766, "127.0.0.1");

// ---- map scenario: Sofia, NDK area -> Sofia Airport (fake but plausible geometry) -------------------------
const ORIGIN = { lat: 42.6847, lon: 23.3188, label: "You are here", accuracy: 35, source: "windows" };
const DEST = { lat: 42.6966, lon: 23.4114, label: "Sofia Airport" };
const CAR_WP = [[23.3188, 42.6847], [23.3262, 42.6858], [23.3335, 42.6894], [23.3412, 42.6921], [23.3498, 42.6908],
  [23.3586, 42.6896], [23.3677, 42.6907], [23.3790, 42.6928], [23.3902, 42.6951], [23.4001, 42.6979], [23.4114, 42.6966]];
const WALK_WP = [[23.3188, 42.6847], [23.3226, 42.6866], [23.3289, 42.6880], [23.3365, 42.6905], [23.3470, 42.6925],
  [23.3566, 42.6910], [23.3690, 42.6921], [23.3805, 42.6944], [23.3935, 42.6961], [23.4040, 42.6990], [23.4114, 42.6966]];
function densify(wp, per) {
  const out = [];
  for (let i = 0; i < wp.length - 1; i++) {
    for (let k = 0; k < per; k++) {
      const t = k / per;
      const wob = Math.sin((i * per + k) * 0.9) * 0.00006;
      out.push([wp[i][0] + (wp[i + 1][0] - wp[i][0]) * t, wp[i][1] + (wp[i + 1][1] - wp[i][1]) * t + wob]);
    }
  }
  out.push(wp[wp.length - 1]);
  return out.map(([x, y]) => [+x.toFixed(5), +y.toFixed(5)]);
}
const carSteps = [["Head east on Bulevard Vitosha", 420], ["Turn left onto Bulevard Evlogi i Hristo Georgievi", 1180],
  ["At the roundabout take the second exit onto Bulevard Tsarigradsko Shose", 1650], ["Continue onto Bulevard Tsarigradsko Shose", 1320],
  ["Keep left at the fork onto Bulevard Brussels", 980], ["Take the ramp onto Sofia Ring Road", 310],
  ["Take the exit onto Bulevard Brussels", 240], ["Turn right onto Terminal 2 Access Road", 160], ["Arrive at your destination on the right", 0]];
const walkSteps = [["Head east on Bulevard Vitosha", 260], ["Turn right onto Ulitsa Pirotska", 540], ["Turn left onto Bulevard Evlogi i Hristo Georgievi", 1320],
  ["Continue onto Bulevard Tsarigradsko Shose", 2100], ["Turn right onto Bulevard Brussels", 1390], ["Continue straight", 210],
  ["Turn left onto Terminal 2 Access Road", 150], ["Arrive at your destination", 0]];
const sum = (a) => a.reduce((n, s) => n + s[1], 0);
function mapShow(mode = "all") {
  const car = { mode: "car", distance_m: sum(carSteps), duration_s: 14 * 60 + 20, geometry: { type: "LineString", coordinates: densify(CAR_WP, 18) },
    steps: carSteps.map(([text, d]) => ({ text, distance_m: d })) };
  const walk = { mode: "walk", distance_m: sum(walkSteps), duration_s: 69 * 60 + 40, geometry: { type: "LineString", coordinates: densify(WALK_WP, 18) },
    steps: walkSteps.map(([text, d]) => ({ text, distance_m: d })) };
  const url = "https://www.google.com/maps/dir/?api=1&origin=42.684700,23.318800&destination=42.696600,23.411400&travelmode=transit";
  if (mode === "london") return { type: "map_show", origin: null, destination: { lat: 51.5074, lon: -0.1278, label: "London", bbox: [51.28, 51.69, -0.51, 0.33] }, routes: [], transit_url: "", focus: "" };
  if (mode === "none") return { type: "map_show", origin: null, destination: DEST, routes: [], transit_url: "", focus: "" };
  if (mode === "transit") return { type: "map_show", origin: ORIGIN, destination: DEST, routes: [], transit_url: url, focus: "transit" };
  return { type: "map_show", origin: ORIGIN, destination: DEST, routes: [car, walk], transit_url: url, focus: "" };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function levels(ms, base) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const t = Date.now() / 1000;
    const rms = Math.max(0, base * (0.55 + 0.45 * Math.sin(t * 7) * Math.sin(t * 2.3)) + Math.random() * 0.02);
    broadcast({ type: "level", rms });
    await sleep(33);
  }
}
async function demo() {
  for (;;) {
    await sleep(2000);
    broadcast({ type: "state", state: "listening" });
    await levels(900, 0.12);
    broadcast({ type: "transcript", text: "open the project folder and", final: false });
    await levels(800, 0.1);
    broadcast({ type: "transcript", text: "open the project folder and run the tests", final: true });
    broadcast({ type: "state", state: "thinking" });
    await sleep(2600);
    broadcast({ type: "state", state: "speaking" });
    broadcast({ type: "reply", text: "Certainly, sir. Opening the project folder now." });
    await levels(2200, 0.15);
    broadcast({ type: "confirm", id: "c1", summary: "Sir, I'm about to run Remove-Item on C:\\build. Shall I proceed?" });
    await sleep(6000);
    broadcast({ type: "confirm_resolved", id: "c1", approved: true });
    broadcast({ type: "state", state: "idle" });
  }
}
if (!manual) demo();

// map_control from the core, e.g. GET /send?m={"type":"map_control","action":"zoom_in","amount":1}
