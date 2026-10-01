// Dependency-free stand-in for the Python core, for developing/testing the orb.
//   node mock/mock-core.mjs          -> ws://127.0.0.1:8765, plays the demo loop
//   node mock/mock-core.mjs --manual -> no demo; POST/GET http://127.0.0.1:8766/send?m=<json> broadcasts
import http from "node:http";
import crypto from "node:crypto";

const manual = process.argv.includes("--manual");
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
  let buf = Buffer.alloc(0);
  sock.on("data", (d) => { buf = readFrames(sock, Buffer.concat([buf, d])); });
  sock.on("close", () => clients.delete(sock));
  sock.on("error", () => clients.delete(sock));
});
ws.listen(8765, "127.0.0.1", () => console.log("mock core on ws://127.0.0.1:8765"));

http.createServer((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname === "/send") { try { broadcast(JSON.parse(u.searchParams.get("m"))); } catch { /* ignore */ } }
  res.end("ok");
}).listen(8766, "127.0.0.1");

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
