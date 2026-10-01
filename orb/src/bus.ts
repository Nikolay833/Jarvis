import { type CoreMessage, type OrbMessage, parseCoreMessage } from "./protocol";

const RETRY_MS = 2000;

/** WebSocket client that silently reconnects every 2 s while the core is down. */
export class Bus {
  private ws: WebSocket | null = null;
  private timer: number | undefined;

  constructor(
    private readonly url: string,
    private readonly onMessage: (m: CoreMessage) => void,
    private readonly onDisconnect: () => void,
  ) {}

  start(): void {
    this.connect();
  }

  send(msg: OrbMessage): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  private connect(): void {
    window.clearTimeout(this.timer);
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.retry();
      return;
    }
    this.ws = ws;
    ws.onmessage = (ev) => {
      const m = parseCoreMessage(ev.data);
      if (m) this.onMessage(m);
    };
    // error is always followed by close; handle reconnect there only.
    ws.onerror = () => {};
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.onDisconnect();
      this.retry();
    };
  }

  private retry(): void {
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.connect(), RETRY_MS);
  }
}
