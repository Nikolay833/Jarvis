import asyncio
import base64
import json

import pytest

from jarvis.config import Config
from jarvis.fastpath import match
from jarvis.tools import load_all, vision
from jarvis.tools.context import ctx
from jarvis.tools.registry import ToolError


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(ctx, "config", Config())
    monkeypatch.setattr(ctx, "bus", None)


def run(coro):
    return asyncio.run(coro)


# ---- capture backend selection ----
def test_backend_prefers_mss():
    assert vision.choose_capture_backend(has_mss=True, has_imagegrab=True) == "mss"


def test_backend_imagegrab_fallback_and_none():
    assert vision.choose_capture_backend(has_mss=False, has_imagegrab=True) == "imagegrab"
    assert vision.choose_capture_backend(has_mss=False, has_imagegrab=False) == ""


def test_capture_without_backend_errors():
    with pytest.raises(ToolError):
        vision.capture_screen(backend="")


def test_capture_uses_chosen_grabber(monkeypatch):
    calls = []
    monkeypatch.setattr(vision, "_grab_mss", lambda m, r: (calls.append(("mss", m, r)) or "img", None))
    monkeypatch.setattr(vision, "_grab_imagegrab", lambda m, r: (calls.append(("ig", m, r)) or "img", None))
    monkeypatch.setattr(vision, "foreground_rect", lambda: (10, 20, 500, 400))
    assert vision.capture_screen(1, active_window=True, backend="mss") == "img"
    assert vision.capture_screen(0, backend="imagegrab") == "img"
    assert calls == [("mss", 1, (10, 20, 500, 400)), ("ig", 0, None)]


# ---- downscale math / window crop ----
@pytest.mark.parametrize("w,h,m,out", [
    (2560, 1440, 1280, (1280, 720)), (1440, 2560, 1280, (720, 1280)), (1000, 500, 1280, (1000, 500)),
    (3840, 2160, 1280, (1280, 720)), (1920, 1080, 0, (1920, 1080)), (5000, 10, 1280, (1280, 3)),
])
def test_fit_size(w, h, m, out):
    assert vision.fit_size(w, h, m) == out


def test_fit_size_bad():
    with pytest.raises(ValueError):
        vision.fit_size(0, 10, 100)


def test_clamp_rect():
    assert vision.clamp_rect((-10, -10, 800, 600), (0, 0, 1920, 1080)) == (0, 0, 800, 600)
    assert vision.clamp_rect((0, 0, 30, 30), (0, 0, 1920, 1080)) is None
    assert vision.clamp_rect((2000, 0, 2500, 500), (0, 0, 1920, 1080)) is None


@pytest.mark.parametrize("q,yes", [("read this error", True), ("what's in this window", True),
                                   ("summarise this page", True), ("what's on my screen", False), ("", False)])
def test_wants_active_window(q, yes):
    assert vision.wants_active_window(q) is yes


def test_encode_image_real_pillow():
    Image = pytest.importorskip("PIL.Image")
    b64, size = vision.encode_image(Image.new("RGBA", (2560, 1440), "white"), 1280)
    assert size == (1280, 720)
    raw = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"  # JPEG


def test_encode_image_fake_small_not_resized():
    class Fake:
        size = (800, 600)
        mode = "RGB"

        def resize(self, *a):
            raise AssertionError("must not resize")

        def save(self, buf, format, quality):
            assert format == "JPEG"
            buf.write(b"jpegbytes")

    b64, size = vision.encode_image(Fake(), 1280)
    assert size == (800, 600) and base64.b64decode(b64) == b"jpegbytes"


# ---- request shape / parsing ----
def test_payload_shape():
    p = vision.build_vision_payload("qwen2.5vl:3b", "describe", ["QUJD"], "2m", 300)
    assert p["model"] == "qwen2.5vl:3b" and p["stream"] is False and p["keep_alive"] == "2m"
    assert p["messages"] == [{"role": "user", "content": "describe", "images": ["QUJD"]}]
    assert p["options"]["num_predict"] == 300
    assert vision.build_vision_payload("m", "p", [], "-1")["keep_alive"] == -1


def test_parse_response_cleans_markdown_and_think():
    data = {"message": {"content": "<think>hm</think>**Error:** `File not found`.\n- second line"}}
    assert vision.parse_vision_response(data) == "Error: File not found. second line"


def test_parse_response_truncates_at_sentence():
    text = "One sentence here. " * 200
    out = vision.clean_spoken(text, 100)
    assert len(out) <= 100 and out.endswith(".")


def test_parse_response_errors():
    with pytest.raises(ToolError):
        vision.parse_vision_response({"message": {"content": "  "}})
    with pytest.raises(ToolError):
        vision.parse_vision_response({"error": "boom"})


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def fake_httpx(monkeypatch, resp, seen):
    import httpx

    class Client:
        def __init__(self, base_url, timeout):
            seen["base_url"], seen["timeout"] = base_url, timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, path, json):
            seen["path"], seen["json"] = path, json
            return resp

    monkeypatch.setattr(httpx, "AsyncClient", Client)


def test_ask_vision_request(monkeypatch):
    seen = {}
    fake_httpx(monkeypatch, FakeResp(200, {"message": {"content": "A terminal window."}}), seen)
    assert run(vision.ask_vision("p", ["QUJD"])) == "A terminal window."
    assert seen["path"] == "/api/chat" and seen["base_url"] == "http://127.0.0.1:11434"
    assert seen["json"]["model"] == "qwen2.5vl:3b" and seen["json"]["keep_alive"] == "2m"
    assert seen["json"]["messages"][0]["images"] == ["QUJD"]


def test_ask_vision_missing_model(monkeypatch):
    fake_httpx(monkeypatch, FakeResp(404, {"error": "not found"}), {})
    with pytest.raises(ToolError, match="ollama pull qwen2.5vl:3b"):
        run(vision.ask_vision("p", ["x"]))


# ---- tools ----
def test_tools_registered():
    reg = load_all()
    for name in ("look_at_screen", "read_screen_text", "look_through_webcam"):
        assert reg.get(name) is not None
    props = reg.get("look_at_screen").parameters
    assert set(props["properties"]) == {"question", "monitor"} and props["required"] == []
    assert reg.get("look_at_screen").risk == "safe"


def test_look_at_screen_flow(monkeypatch):
    events, seen = [], {}
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(vision, "capture_screen", lambda monitor=0, active_window=False:
                        seen.update(monitor=monitor, active=active_window) or Image.new("RGB", (100, 50)))
    monkeypatch.setattr(vision, "foreground_rect", lambda: (0, 0, 500, 500))

    async def fake_ask(prompt, images):
        seen["prompt"], seen["n"] = prompt, len(images)
        return "A dialog says disk full."

    monkeypatch.setattr(vision, "ask_vision", fake_ask)
    monkeypatch.setattr(vision, "emit", lambda t, **f: events.append((t, f)))
    out = load_all()
    assert run(out.call("look_at_screen", {"question": "read this error", "monitor": 1})) == "A dialog says disk full."
    assert seen["monitor"] == 1 and seen["active"] is True and seen["n"] == 1
    assert "read this error" in seen["prompt"]
    assert [e[1]["active"] for e in events] == [True, False]


def test_webcam_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(vision, "capture_webcam", lambda i=0: called.append(i))
    res = run(load_all().call("look_through_webcam", {"question": "look at me"}))
    assert res.startswith("Error:") and "switched off" in res and not called


def test_webcam_enabled_emits_and_logs(monkeypatch, caplog):
    Image = pytest.importorskip("PIL.Image")
    ctx.config.vision.webcam_enabled = True
    events = []
    monkeypatch.setattr(vision, "emit", lambda t, **f: events.append((t, f)))
    monkeypatch.setattr(vision, "capture_webcam", lambda i=0: Image.new("RGB", (64, 64)))

    async def fake_ask(prompt, images):
        return "A person."

    monkeypatch.setattr(vision, "ask_vision", fake_ask)
    with caplog.at_level("WARNING", logger="jarvis.vision"):
        assert run(load_all().call("look_through_webcam", {})) == "A person."
    assert "WEBCAM" in caplog.text
    assert [(e[0], e[1]["active"], e[1]["source"]) for e in events] == [
        ("vision", True, "webcam"), ("vision", False, "webcam")]


def test_vision_config_defaults():
    v = Config().vision
    assert v.model == "qwen2.5vl:3b" and v.keep_alive == "2m" and v.max_side == 1280 and v.webcam_enabled is False


# ---- fast paths ----
@pytest.mark.parametrize("text", ["what's on my screen", "What is on my screen?", "Jarvis, what's on my screen please",
                                  "what am I looking at", "what does this error say", "read this error",
                                  "can you tell me what do you see on my screen"])
def test_fast_look(text):
    fp = match(text)
    assert fp and fp.kind == "vision" and fp.speak_result
    assert fp.action[:2] == ("call", "look_at_screen")
    assert json.loads(fp.action[2])["question"]


def test_fast_question_is_utterance():
    fp = match("Jarvis, what does this error say?")
    assert json.loads(fp.action[2]) == {"question": "what does this error say"}


@pytest.mark.parametrize("text", ["read my screen", "read the screen to me"])
def test_fast_read(text):
    fp = match(text)
    assert fp and fp.action[:2] == ("call", "read_screen_text")


@pytest.mark.parametrize("text", ["what's on my calendar", "look at me", "what's on my screen saver settings",
                                  "what do you see through the camera"])
def test_not_vision(text):
    fp = match(text)
    assert fp is None or fp.kind != "vision"


def test_slow_notice():
    from jarvis.agent import SLOW_TOOL_NOTICE, SYSTEM_PROMPT

    assert SLOW_TOOL_NOTICE["look_at_screen"] == "One moment, sir."
    assert "look_at_screen" in SYSTEM_PROMPT and "look_through_webcam" in SYSTEM_PROMPT
