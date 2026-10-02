"""Computer vision: look at the screen (or, only when enabled and asked, the webcam) with a small local VLM.

Flow: capture (mss, else PIL.ImageGrab) -> downscale -> base64 JPEG -> Ollama /api/chat with `images` ->
short spoken answer. The vision model is separate from the main model and unloads soon after use
([vision] keep_alive) so qwen3 keeps its VRAM. Ollama may evict the main model if both do not fit; the
next normal turn then reloads it (a few seconds).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from typing import Any

from ..llm import normalize_keep_alive, strip_think
from .context import IS_WINDOWS, ctx, emit
from .registry import ToolError, tool

log = logging.getLogger("jarvis.vision")

MAX_SPOKEN_CHARS = 1200
MIN_WINDOW_SIDE = 80  # a smaller foreground rect is not a real window: use the whole screen

SCREEN_PROMPT = (
    "You are the eyes of a voice assistant. The image is a screenshot of the user's computer screen. "
    "Answer the user's request about it concisely, in one to three plain spoken sentences: no markdown, "
    "no bullet points, no lists. If it asks about an error or a message, read the key text verbatim "
    "and say what it means. Only describe what is really visible; if something is unreadable, say so."
)
DESCRIBE_PROMPT = (
    "You are the eyes of a voice assistant. The image is a screenshot of the user's computer screen. "
    "Say in one to three plain spoken sentences what is on it: the main window or app and what the user "
    "seems to be doing. If an error message is visible, read it verbatim. No markdown, no lists."
)
TRANSCRIBE_PROMPT = (
    "Transcribe the main readable text in this screenshot exactly as written, in reading order. "
    "Skip menus, toolbars and decoration. Output only the text, no commentary, no markdown."
)
WEBCAM_PROMPT = (
    "You are the eyes of a voice assistant. The image is a single frame from the user's webcam. "
    "Answer the user's request about it concisely, in one to three plain spoken sentences, no markdown. "
    "If no question was asked, say what you see. Only describe what is really visible."
)

_THIS_WINDOW = re.compile(
    r"\b(?:this|the|current|active|that|my)\s+(?:window|error|dialog|popup|pop up|message|app|page|tab|"
    r"warning|exception|traceback|terminal|code)\b|\bwindow\b", re.IGNORECASE)


# ---- pure helpers ------------------------------------------------------------------------------------
def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Size scaled down (never up) so the longest side is at most `max_side`; aspect ratio kept."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image size {width}x{height}")
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def wants_active_window(question: str) -> bool:
    """True when the question is about the foreground window ("this error", "this window")."""
    return bool(question and _THIS_WINDOW.search(question))


def clamp_rect(rect: tuple[int, int, int, int], bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Intersect (left, top, right, bottom) rects; None if empty or too small to be a window."""
    l, t = max(rect[0], bounds[0]), max(rect[1], bounds[1])
    r, b = min(rect[2], bounds[2]), min(rect[3], bounds[3])
    if r - l < MIN_WINDOW_SIDE or b - t < MIN_WINDOW_SIDE:
        return None
    return l, t, r, b


def encode_image(img: Any, max_side: int = 1280, quality: int = 85) -> tuple[str, tuple[int, int]]:
    """Downscale a PIL image and return (base64 JPEG, (w, h) sent)."""
    w, h = img.size
    size = fit_size(w, h, max_side)
    if size != (w, h):
        try:
            from PIL import Image

            resample = Image.Resampling.LANCZOS
        except Exception:  # noqa: BLE001
            resample = None
        img = img.resize(size, resample) if resample is not None else img.resize(size)
    if getattr(img, "mode", "RGB") != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii"), size


def build_vision_payload(model: str, prompt: str, images_b64: list[str], keep_alive: int | str,
                         num_predict: int = 400) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "keep_alive": normalize_keep_alive(keep_alive),
        "options": {"num_ctx": 4096, "temperature": 0.2, "num_predict": num_predict},
    }


def clean_spoken(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Strip think blocks and markdown, collapse whitespace, cut at a sentence end near `limit`."""
    text = strip_think(text or "")
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = " ".join(text.split())
    if len(text) > limit:
        cut = text[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end > limit // 2 else cut.rstrip() + "..."
    return text


def parse_vision_response(data: dict[str, Any]) -> str:
    if data.get("error"):
        raise ToolError(f"vision model error: {str(data['error'])[:200]}")
    text = clean_spoken((data.get("message") or {}).get("content") or "")
    if not text:
        raise ToolError("the vision model returned no answer")
    return text


# ---- capture -----------------------------------------------------------------------------------------
def foreground_rect() -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of the foreground window in screen pixels (Windows only)."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd or user32.IsIconic(hwnd):
            return None
        rect = wintypes.RECT()
        # The extended frame excludes the invisible resize border Windows 10/11 adds to GetWindowRect.
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)) == 0
        if not ok and not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:  # noqa: BLE001
        log.debug("foreground window rect failed", exc_info=True)
        return None


def _have(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def choose_capture_backend(has_mss: bool | None = None, has_imagegrab: bool | None = None) -> str:
    """"mss" if installed, else "imagegrab" on Windows, else "" (no way to capture)."""
    if has_mss is None:
        has_mss = _have("mss")
    if has_mss:
        return "mss"
    if has_imagegrab is None:
        has_imagegrab = IS_WINDOWS and _have("PIL")
    return "imagegrab" if has_imagegrab else ""


def _grab_mss(monitor: int, region: tuple[int, int, int, int] | None) -> tuple[Any, tuple[int, int, int, int]]:
    import mss
    from PIL import Image

    with mss.mss() as sct:
        mons = sct.monitors  # [0] = all screens together, [1] = primary, [2...] = the others
        idx = monitor + 1
        if idx < 1 or idx >= len(mons):
            raise ToolError(f"there is no monitor {monitor}; I can see {len(mons) - 1}")
        m = mons[idx]
        bounds = (m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"])
        box = clamp_rect(region, bounds) if region else None
        area = ({"left": box[0], "top": box[1], "width": box[2] - box[0], "height": box[3] - box[1]}
                if box else m)
        shot = sct.grab(area)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"), box or bounds


def _grab_imagegrab(monitor: int, region: tuple[int, int, int, int] | None) -> tuple[Any, tuple[int, int, int, int]]:
    from PIL import ImageGrab

    if monitor != 0:
        raise ToolError("only the main monitor can be captured (install the mss package for more)")
    full = ImageGrab.grab()
    bounds = (0, 0, full.width, full.height)
    box = clamp_rect(region, bounds) if region else None
    return (full.crop(box) if box else full), box or bounds


def capture_screen(monitor: int = 0, active_window: bool = False, backend: str | None = None) -> Any:
    """Capture a monitor (0 = main) or, with `active_window`, the foreground window. Returns a PIL image."""
    backend = backend if backend is not None else choose_capture_backend()
    if not backend:
        raise ToolError("screen capture is not available here (pip install mss)")
    region = foreground_rect() if active_window else None
    grab = _grab_mss if backend == "mss" else _grab_imagegrab
    img, _ = grab(monitor, region)
    return img


def capture_webcam(index: int = 0) -> Any:
    """One frame from the camera as a PIL image. Needs opencv-python."""
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise ToolError('the camera needs OpenCV: pip install -e ".[webcam]"') from exc
    api = getattr(cv2, "CAP_DSHOW", 0) if IS_WINDOWS else 0
    cap = cv2.VideoCapture(index, api)
    try:
        if not cap.isOpened():
            raise ToolError("I cannot open the camera; it may be in use or blocked in Windows privacy settings")
        frame = None
        for _ in range(6):  # the first frames are dark while auto exposure settles
            ok, got = cap.read()
            if ok:
                frame = got
        if frame is None:
            raise ToolError("the camera gave no picture")
    finally:
        cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---- model call --------------------------------------------------------------------------------------
async def ask_vision(prompt: str, images_b64: list[str]) -> str:
    """Send images to the Ollama vision model and return the cleaned spoken answer."""
    import httpx

    cfg = ctx.config.vision
    url = (cfg.url or ctx.config.ollama.url).rstrip("/")
    payload = build_vision_payload(cfg.model, prompt, images_b64, cfg.keep_alive, cfg.max_reply_tokens)
    log.info("vision request to %s (%s, %d image(s); the first call loads the model and evicts qwen3 "
             "if VRAM is short)", url, cfg.model, len(images_b64))
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=url, timeout=cfg.timeout) as client:
            resp = await client.post("/api/chat", json=payload)
    except httpx.TimeoutException as exc:
        raise ToolError(f"the vision model took longer than {cfg.timeout:.0f} seconds") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"cannot reach Ollama at {url}: {exc!r}") from exc
    if resp.status_code == 404:
        raise ToolError(f"the vision model is not installed; run: ollama pull {cfg.model}")
    if resp.status_code != 200:
        raise ToolError(f"Ollama error {resp.status_code}: {resp.text[:200]}")
    text = parse_vision_response(resp.json())
    log.info("vision answer in %.1f s: %s", time.perf_counter() - t0, text[:120])
    return text


async def _look(kind: str, prompt: str, grab: Any) -> str:
    import asyncio

    cfg = ctx.config.vision
    emit("vision", active=True, source=kind)
    try:
        img = await asyncio.to_thread(grab)
        b64, size = await asyncio.to_thread(encode_image, img, cfg.max_side)
        log.info("vision %s capture %sx%s -> %sx%s (%d KB)", kind, *img.size, *size, len(b64) * 3 // 4096)
        return await ask_vision(prompt, [b64])
    finally:
        emit("vision", active=False, source=kind)


def _screen_prompt(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return DESCRIBE_PROMPT
    return f'{SCREEN_PROMPT}\n\nThe user said: "{q}"'


@tool("Look at the user's screen and answer a question about it, or describe what is on it. Use for "
      "\"what's on my screen\", \"what am I looking at\", \"read this error\", \"summarise this page\". "
      "Pass the user's own words as the question. Slow (a few seconds).")
async def look_at_screen(question: str = "", monitor: int = 0) -> str:
    """Take a screenshot and ask the vision model about it.

    Args:
        question: What the user wants to know, in their words; empty = describe the screen.
        monitor: 0 = main monitor, 1 = the next one, and so on.
    """
    active = wants_active_window(question)
    if active and foreground_rect() is None:
        active = False
    return await _look("screen", _screen_prompt(question),
                       lambda: capture_screen(monitor=max(0, monitor), active_window=active))


@tool("Read the text on the user's screen aloud, word for word. Use for \"read this to me\", "
      "\"read the screen\". Slow (a few seconds).")
async def read_screen_text() -> str:
    """Transcribe the main readable text of the screen."""
    return await _look("screen", TRANSCRIBE_PROMPT, lambda: capture_screen(active_window=False))


@tool("Take one picture with the webcam and answer a question about it. ONLY when the user explicitly asks "
      "to look at them or through the camera (\"look at me\", \"what do you see through the camera\"). "
      "Never use it for the screen.")
async def look_through_webcam(question: str = "") -> str:
    """Capture a single webcam frame and describe it.

    Args:
        question: What the user wants to know, in their words; empty = describe what is seen.
    """
    cfg = ctx.config.vision
    if not cfg.webcam_enabled:
        raise ToolError("the camera is switched off; set webcam_enabled = true under [vision] in config.toml")
    log.warning("WEBCAM in use: capturing one frame on explicit request (camera %d)", cfg.webcam_index)
    q = (question or "").strip()
    prompt = f'{WEBCAM_PROMPT}\n\nThe user said: "{q}"' if q else WEBCAM_PROMPT
    return await _look("webcam", prompt, lambda: capture_webcam(cfg.webcam_index))
