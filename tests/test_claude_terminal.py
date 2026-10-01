from jarvis.tools.claude_terminal import build_script, claude_invocation, launch_command


def test_script_new_conversation_reads_prompt_from_file():
    s = build_script("& 'claude'", r"C:\Users\Nik's\proj", r"C:\Temp\p.txt", "new", "")
    assert "Set-Location -LiteralPath 'C:\\Users\\Nik''s\\proj'" in s
    assert "$p = Get-Content -Raw -Encoding UTF8 -LiteralPath 'C:\\Temp\\p.txt'" in s
    assert s.rstrip().endswith("& 'claude' $p")


def test_script_continue_and_resume_flags():
    assert "& 'claude' --continue $p" in build_script("& 'claude'", "C:\\x", "C:\\p.txt", "continue", "")
    assert "& 'claude' --resume 'abc-123' $p" in build_script("& 'claude'", "C:\\x", "C:\\p.txt", "resume", "abc-123")


def test_script_without_prompt_just_opens_claude():
    s = build_script("& 'claude'", "C:\\x", None, "new", "")
    assert "$p" not in s and s.rstrip().endswith("& 'claude'")


def test_prefers_ps1_shim_next_to_cmd(tmp_path):
    cmd = tmp_path / "claude.cmd"
    cmd.write_text("")
    assert claude_invocation(str(cmd)) == f"& '{cmd}'"
    (tmp_path / "claude.ps1").write_text("")
    assert claude_invocation(str(cmd)) == f"& '{tmp_path / 'claude.ps1'}'"


def test_launch_falls_back_to_powershell_without_wt(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    cmd = launch_command("C:\\t\\s.ps1", "C:\\x")
    assert cmd[0] == "powershell.exe" and cmd[-2:] == ["-File", "C:\\t\\s.ps1"] and "-NoExit" in cmd
