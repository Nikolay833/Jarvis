// Caption line and confirm pills. Pure DOM, no animation loop.

const TAIL_CHARS = 96;
const SWAP_MS = 120;
const CLEAR_AFTER_MS = 700;

export class Caption {
  private clearTimer: number | undefined;
  private swapTimer: number | undefined;

  constructor(private readonly el: HTMLElement) {}

  /** Live words: show the tail so the newest words stay visible. */
  transcript(text: string, final: boolean): void {
    const t = text.trim();
    const shown = t.length > TAIL_CHARS ? "…" + t.slice(-TAIL_CHARS).trimStart() : t;
    this.set(shown, { dim: false, instant: !final });
  }

  /** Jarvis reply sentence: crossfade, CSS ellipsis truncates the tail. */
  reply(text: string): void {
    this.set(text.trim(), { dim: false, instant: false });
  }

  summary(text: string): void {
    this.set(text.trim(), { dim: false, instant: false });
  }

  dim(on: boolean): void {
    this.el.classList.toggle("dim", on);
  }

  fadeOut(): void {
    window.clearTimeout(this.swapTimer);
    this.el.classList.remove("on", "dim", "swap");
    window.clearTimeout(this.clearTimer);
    this.clearTimer = window.setTimeout(() => {
      this.el.textContent = "";
    }, CLEAR_AFTER_MS);
  }

  clear(): void {
    window.clearTimeout(this.clearTimer);
    window.clearTimeout(this.swapTimer);
    this.el.textContent = "";
    this.el.classList.remove("on", "dim", "swap");
  }

  private set(text: string, opts: { dim: boolean; instant: boolean }): void {
    window.clearTimeout(this.clearTimer);
    window.clearTimeout(this.swapTimer);
    this.el.classList.toggle("dim", opts.dim);
    if (!text) {
      this.fadeOut();
      return;
    }
    const apply = () => {
      this.el.textContent = text;
      this.el.title = text; // full text on hover if truncated
      this.el.classList.remove("swap");
      this.el.classList.add("on");
    };
    const visible = this.el.classList.contains("on") && this.el.textContent !== "";
    if (opts.instant || !visible) {
      apply();
    } else {
      this.el.classList.add("swap");
      this.swapTimer = window.setTimeout(apply, SWAP_MS);
    }
  }
}

export class ConfirmPills {
  private id: string | null = null;
  private timeout: number | undefined;

  constructor(
    private readonly box: HTMLElement,
    approve: HTMLButtonElement,
    deny: HTMLButtonElement,
    private readonly onAnswer: (id: string, approved: boolean) => void,
    private readonly onVisible: (visible: boolean) => void,
  ) {
    approve.addEventListener("click", () => this.answer(true));
    deny.addEventListener("click", () => this.answer(false));
  }

  get pending(): boolean {
    return this.id !== null;
  }

  show(id: string): void {
    this.id = id;
    this.box.removeAttribute("inert");
    this.box.classList.add("on");
    window.clearTimeout(this.timeout);
    // Core counts 30 s as "no"; hide locally a little later if it never tells us.
    this.timeout = window.setTimeout(() => this.hide(), 35_000);
    this.onVisible(true);
  }

  resolved(id: string): void {
    if (id === this.id) this.hide();
  }

  hide(): void {
    if (this.id === null) return;
    this.id = null;
    window.clearTimeout(this.timeout);
    this.box.classList.remove("on");
    this.box.setAttribute("inert", "");
    this.onVisible(false);
  }

  private answer(approved: boolean): void {
    const id = this.id;
    if (id === null) return;
    this.onAnswer(id, approved);
    this.hide();
  }
}
