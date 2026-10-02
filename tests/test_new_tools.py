import asyncio
import json
import time

import pytest

from jarvis import safety
from jarvis.fastpath import match
from jarvis.tools import chrome, claude_chat, spotify, windows
from jarvis.tools.registry import ToolError
from jarvis.tools.windows import Win


# ---- fast paths -----------------------------------------------------------------
def call_of(text):
    fp = match(text)
    assert fp is not None, text
    assert fp.action[0] == "call"
    return fp.action[1], json.loads(fp.action[2]), fp


@pytest.mark.parametrize("text,tool,args", [
    ("minimise chrome", "window_action", {"app": "chrome", "action": "minimize"}),
    ("Jarvis, minimize the spotify window please", "window_action", {"app": "spotify", "action": "minimize"}),
    ("maximize vscode", "window_action", {"app": "vscode", "action": "maximize"}),
    ("maximise the file explorer", "window_action", {"app": "file explorer", "action": "maximize"}),
    ("minimise everything", "minimize_all", {}),
    ("show the desktop", "minimize_all", {}),
    ("show desktop", "minimize_all", {}),
    ("pause the music", "music_control", {"action": "play_pause"}),
    ("pause", "music_control", {"action": "play_pause"}),
    ("play the music", "music_control", {"action": "play_pause"}),
    ("resume", "music_control", {"action": "play_pause"}),
    ("skip song", "music_control", {"action": "next"}),
    ("next track", "music_control", {"action": "next"}),
    ("previous song", "music_control", {"action": "previous"}),
    ("stop the music", "music_control", {"action": "stop"}),
    ("what's playing", "spotify_now_playing", {}),
    ("what song is this", "spotify_now_playing", {}),
    ("play bohemian rhapsody on spotify", "spotify_play", {"query": "bohemian rhapsody"}),
    ("Jarvis play Blinding Lights by The Weeknd on Spotify.", "spotify_play", {"query": "blinding lights by the weeknd"}),
    ("search for cats online", "open_chrome", {"search": "cats"}),
    ("search for best pizza in chrome", "open_chrome", {"search": "best pizza"}),
    ("google python tutorials", "open_chrome", {"search": "python tutorials"}),
])
def test_new_fast_paths(text, tool, args):
    t, a, _ = call_of(text)
    assert (t, a) == (tool, args)


def test_speak_result_flags():
    assert call_of("what's playing")[2].speak_result
    assert call_of("play x on spotify")[2].speak_result
    assert not call_of("pause the music")[2].speak_result


@pytest.mark.parametrize("text", [
    "close chrome", "close all chrome windows", "close the window", "minimise the volume", "minimise my stress",
    "maximise profits", "play bohemian rhapsody", "play some music", "play it again on youtube",
    "search for cats on youtube", "google chrome", "search", "google", "what's playing at the cinema",
    "pause the meeting", "skip the intro", "next week", "open chrome with my work profile",
    "stop the music player from crashing today", "show me the desktop wallpapers",
])
def test_new_fast_paths_negative(text):
    assert match(text) is None, text


# ---- safety ---------------------------------------------------------------------------
def test_safety_window_and_chat():
    assert safety.classify_call("window_action", {"app": "chrome", "action": "close"}).is_risky
    assert safety.classify_call("window_action", {"app": "chrome", "action": " Close "}).is_risky
    for a in ("minimize", "maximize", "restore", "focus"):
        assert not safety.classify_call("window_action", {"app": "chrome", "action": a}).is_risky
    assert safety.describe_call("window_action", {"app": "Chrome", "action": "close"}) == "close all Chrome windows"
    assert safety.classify_call("claude_chat_delete", {"name": "x"}, safety.RISKY).is_risky
    assert not safety.classify_call("claude_chat", {"message": "hi"}).is_risky
    assert "Claude chat" in safety.describe_call("claude_chat_delete", {"name": "jokes"})


def test_registered_risks():
    from jarvis.tools import load_all

    reg = load_all()
    assert reg.get("claude_chat_delete").risk == "risky"
    for n in ("window_action", "claude_chat", "open_chrome", "spotify_play", "music_control", "list_windows"):
        assert reg.get(n).risk == "safe", n


# ---- windows matching -------------------------------------------------------------------
WINS = [
    Win(1, "Inbox - Gmail - Google Chrome", "chrome.exe"),
    Win(2, "Spotify Free", "spotify.exe"),
    Win(3, "main.py - jarvis - Visual Studio Code", "code.exe"),
    Win(4, "Docs - Google Chrome", "chrome.exe"),
    Win(5, "notes.txt - Notepad", "notepad.exe"),
    Win(6, "Barcode generator", "firefox.exe"),
    Win(7, "Downloads", "explorer.exe"),
    Win(8, "Terminal", "windowsterminal.exe"),
]


def ids(app):
    return [w.hwnd for w in windows.match_windows(WINS, app)]


def test_match_windows():
    assert ids("chrome") == [1, 4]
    assert ids("Google Chrome") == [1, 4]
    assert ids("vscode") == [3] and ids("vs code") == [3] and ids("code") == [3]  # not the "Barcode" title
    assert ids("spotify") == [2]
    assert ids("file explorer") == [7] and ids("explorer") == [7]
    assert ids("terminal") == [8]
    assert ids("gmail") == [1]  # title match
    assert ids("notep") == [5]  # exe substring
    assert ids("") == [] and ids("zoom") == []
    assert windows.alias_exe("vscode") == "code.exe"
    assert windows.alias_exe("unknown app") is None


def test_window_action_validates_without_windows():
    with pytest.raises(ToolError):
        windows.window_action("chrome", "explode")
    with pytest.raises(ToolError):
        windows.enum_windows()  # not Windows


# ---- chrome ---------------------------------------------------------------------------------
LOCAL_STATE = {"profile": {"info_cache": {
    "Profile 2": {"name": "Work", "gaia_name": "Nikolay Radni", "user_name": "niki@company.com"},
    "Default": {"name": "Personal", "gaia_name": "Nikolay", "user_name": "niki.radni@gmail.com"},
    "Profile 3": {"name": "Gaming", "gaia_name": "", "user_name": ""},
}}}


def test_parse_local_state():
    ps = chrome.parse_local_state(LOCAL_STATE)
    assert [p["dir"] for p in ps] == ["Default", "Profile 2", "Profile 3"]
    assert ps[0]["user_name"] == "niki.radni@gmail.com"
    assert chrome.parse_local_state({}) == []


@pytest.mark.parametrize("q,dirs", [
    ("work", ["Profile 2"]), ("Work", ["Profile 2"]), ("personal", ["Default"]), ("gmail", ["Default"]),
    ("niki@company.com", ["Profile 2"]), ("niki.radni", ["Default"]), ("profile 3", ["Profile 3"]),
    ("gamin", ["Profile 3"]), ("Workk", ["Profile 2"]), ("Radni", ["Default", "Profile 2"]),
    ("nikolay", ["Default"]),  # exact gaia name beats the substring in "Nikolay Radni"
    ("niki", ["Profile 2"]),  # exact email local part
    ("zzz", []), ("", []),
])
def test_match_profile(q, dirs):
    ps = chrome.parse_local_state(LOCAL_STATE)
    assert sorted(p["dir"] for p in chrome.match_profile(ps, q)) == sorted(dirs)


def test_launch_args():
    a = chrome.build_launch_args("C:/chrome.exe", "Profile 2", search="best pizza & beer")
    assert a == ["C:/chrome.exe", "--profile-directory=Profile 2",
                 "https://www.google.com/search?q=best+pizza+%26+beer"]
    assert chrome.build_launch_args("c", "", url="example.com") == ["c", "https://example.com"]
    assert chrome.build_launch_args("c") == ["c"]
    with pytest.raises(ToolError):
        chrome.build_launch_args("c", url="file:///etc/passwd")


def test_open_chrome_unknown_profile_lists_profiles(tmp_path, monkeypatch):
    p = tmp_path / "Google" / "Chrome" / "User Data"
    p.mkdir(parents=True)
    (p / "Local State").write_text(json.dumps(LOCAL_STATE))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    with pytest.raises(ToolError) as e:
        chrome.open_chrome(profile="nonesuch")
    assert "Work" in str(e.value) and "Personal" in str(e.value)
    assert "Work" in chrome.chrome_profiles()


# ---- spotify --------------------------------------------------------------------------------
@pytest.mark.parametrize("title,expected", [
    ("Daft Punk - One More Time", ("Daft Punk", "One More Time")),
    ("AC/DC - Back In Black - Remastered", ("AC/DC", "Back In Black - Remastered")),
    ("Spotify", None), ("Spotify Free", None), ("spotify premium", None), ("Advertisement", None),
    ("", None), ("Just a title", None), (" - x", None),
])
def test_parse_now_playing(title, expected):
    assert spotify.parse_now_playing(title) == expected


def test_now_playing_from_windows():
    wins = [Win(1, "Spotify Free", "spotify.exe"), Win(2, "Queen - Bohemian Rhapsody", "spotify.exe")]
    assert spotify.now_playing_from(wins) == ("Queen", "Bohemian Rhapsody")
    assert spotify.now_playing_from(wins[:1]) is None


def test_music_control_maps_keys(monkeypatch):
    pressed = []
    monkeypatch.setattr(spotify.system, "press_key", pressed.append)
    spotify.music_control("pause")
    spotify.music_control("next")
    spotify.music_control("previous")
    spotify.music_control("stop")
    assert pressed == [0xB3, 0xB0, 0xB1, 0xB2]
    with pytest.raises(ToolError):
        spotify.music_control("dance")


# ---- claude chat ------------------------------------------------------------------------------
@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return tmp_path


def test_build_command_new_vs_resume():
    new = claude_chat.build_command("claude", "sid-1", resume=False)
    assert new[:3] == ["claude", "-p", "--output-format"] and "json" in new
    assert new[new.index("--session-id") + 1] == "sid-1" and "--resume" not in new
    res = claude_chat.build_command("claude", "sid-1", resume=True)
    assert res[res.index("--resume") + 1] == "sid-1" and "--session-id" not in res
    for cmd in (new, res):
        assert cmd[cmd.index("--disallowedTools") + 1] == "Bash,Edit,Write,MultiEdit,NotebookEdit"
        assert "voice assistant" in cmd[cmd.index("--append-system-prompt") + 1]
        assert "--permission-mode" not in cmd and "WebSearch,WebFetch" in cmd


def test_parse_result():
    ok = json.dumps({"type": "result", "result": "Hello there.", "session_id": "s9", "is_error": False})
    assert claude_chat.parse_result(ok) == ("Hello there.", "s9", False)
    err = json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "boom"})
    assert claude_chat.parse_result(err)[2] is True
    lst = json.dumps([{"type": "system"}, {"type": "result", "result": "x", "session_id": "s"}])
    assert claude_chat.parse_result(lst) == ("x", "s", False)
    assert claude_chat.parse_result("plain text") == ("plain text", "", False)


def test_names_and_recent(appdata):
    assert claude_chat.name_from_message("Can you tell me a joke about the cats please") == "tell joke about cats"
    store = {"a": {"last_used": 1000.0}, "b": {"last_used": 2000.0}}
    assert claude_chat.pick_recent(store, now=2000 + 60) == "b"
    assert claude_chat.pick_recent(store, now=2000 + 31 * 60) is None
    assert claude_chat.pick_recent({}, now=1) is None
    assert claude_chat.find_chat({"Dinner plans": {}}, "dinner") == "Dinner plans"
    assert claude_chat.unique_name({"x": {}}, "x") == "x 2"


class FakeCli:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, cmd, cwd, message, timeout=0):
        self.calls.append((cmd, cwd, message))
        r = self.replies.pop(0)
        return r if isinstance(r, tuple) else (0, json.dumps({"type": "result", "result": r, "session_id": "ret-" + str(len(self.calls)), "is_error": False}), "")


def run(coro):
    return asyncio.run(coro)


def test_chat_flow_new_then_resume(appdata, monkeypatch):
    cli = FakeCli(["Paris is the capital.", "About two million people."])
    monkeypatch.setattr(claude_chat, "run_cli", cli)
    monkeypatch.setattr(claude_chat, "resolve_binary", lambda b: "claude")
    assert run(claude_chat.claude_chat("What is the capital of France")) == "Paris is the capital."
    store = claude_chat.load_store()
    assert list(store) == ["capital france"] or len(store) == 1
    name = next(iter(store))
    entry = store[name]
    assert entry["started"] and entry["session_id"] == "ret-1" and (appdata / "Jarvis" / "claude-chat").is_dir()
    first_cmd, cwd, msg = cli.calls[0]
    assert "--session-id" in first_cmd and msg == "What is the capital of France" and cwd.endswith("claude-chat")
    # follow-up within 30 min resumes the same chat
    assert run(claude_chat.claude_chat("And its population?")) == "About two million people."
    second_cmd = cli.calls[1][0]
    assert second_cmd[second_cmd.index("--resume") + 1] == "ret-1"
    assert len(claude_chat.load_store()) == 1
    md = (appdata / "Jarvis" / "claude_chats" / f"{name}.md").read_text()
    assert "**Q:** And its population?" in md and "**A:** About two million people." in md
    hist = claude_chat.claude_chat_history()
    assert "population" in hist and "Paris" in hist
    assert name in claude_chat.claude_chat_list()


def test_chat_old_session_starts_new_and_named(appdata, monkeypatch):
    cli = FakeCli(["one", "two", "three"])
    monkeypatch.setattr(claude_chat, "run_cli", cli)
    monkeypatch.setattr(claude_chat, "resolve_binary", lambda b: "claude")
    run(claude_chat.claude_chat("first topic here"))
    store = claude_chat.load_store()
    for e in store.values():
        e["last_used"] = time.time() - 3600
    claude_chat.save_store(store)
    run(claude_chat.claude_chat("something else entirely"))
    assert len(claude_chat.load_store()) == 2
    run(claude_chat.claude_chat("hello", session="dinner plans"))
    assert "dinner plans" in claude_chat.load_store()


def test_chat_errors_and_lost_session(appdata, monkeypatch):
    monkeypatch.setattr(claude_chat, "resolve_binary", lambda b: "claude")
    monkeypatch.setattr(claude_chat, "run_cli", FakeCli([(1, "", "auth failed")]))
    with pytest.raises(ToolError, match="auth failed"):
        run(claude_chat.claude_chat("hi there"))
    assert claude_chat.load_store() == {}
    # resume of a session Claude forgot -> retried once as a new session
    store = {"old": {**claude_chat.new_entry(), "started": True}}
    claude_chat.save_store(store)
    lost = (1, "", "No conversation found with session ID")
    cli = FakeCli([lost, "recovered"])
    monkeypatch.setattr(claude_chat, "run_cli", cli)
    assert run(claude_chat.claude_chat("hi", session="old")) == "recovered"
    assert "--session-id" in cli.calls[1][0]
    with pytest.raises(ToolError):
        run(claude_chat.claude_chat("   "))


def test_chat_new_and_delete(appdata, monkeypatch):
    out = run(claude_chat.claude_chat_new("jokes"))
    assert "jokes" in out and claude_chat.load_store()["jokes"]["started"] is False
    cli = FakeCli(["Ha."])
    monkeypatch.setattr(claude_chat, "run_cli", cli)
    monkeypatch.setattr(claude_chat, "resolve_binary", lambda b: "claude")
    run(claude_chat.claude_chat("tell a joke", session="jokes"))
    assert "--session-id" in cli.calls[0][0]  # first message into a pre-created chat is not a resume
    assert claude_chat.claude_chat_delete("joke").startswith("Deleted")
    assert claude_chat.load_store() == {}
    assert not (appdata / "Jarvis" / "claude_chats" / "jokes.md").exists()
    with pytest.raises(ToolError):
        claude_chat.claude_chat_delete("nope")
    assert claude_chat.claude_chat_list() == "No Claude chats yet"
