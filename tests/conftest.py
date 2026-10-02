import pytest


@pytest.fixture(autouse=True)
def _isolated_appdata(tmp_path, monkeypatch):
    """Memory, reminders and state files go to a temp folder, never the real app folder."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
