# Jarvis setup for Windows 11 / 10. Run from the repo root or anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-Model qwen3:14b]
param(
    [string]$Model = "qwen3:14b",
    [string]$Python = "py -3.11"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Creating venv (.venv)"
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$Python -m venv .venv"
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip

# RTX 50-series (Blackwell) needs CUDA 12.8+ builds.
Write-Host "==> Installing PyTorch (cu128)"
& $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "==> Installing Jarvis"
& $Py -m pip install -e ".[dev]"
# faster-whisper needs ctranslate2 >= 4.5 (CUDA 12 + cuDNN 9).
& $Py -m pip install --upgrade "ctranslate2>=4.5"

Write-Host "==> Downloading wake word models"
& $Py -c "import openwakeword; openwakeword.utils.download_models(['hey_jarvis'])"

if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "==> Created config.toml from the example"
}

Write-Host "==> Pulling Ollama model $Model"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull $Model
} else {
    Write-Warning "Ollama not found. Install it from https://ollama.com/download, then run: ollama pull $Model"
}

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Warning "Claude Code CLI not found on PATH (only needed for the Claude Code tools). Set claude_code.binary in config.toml."
}

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Test without audio:   .venv\Scripts\python -m jarvis --text"
Write-Host "  2. Start everything:     powershell -File scripts\start_jarvis.ps1"
Write-Host "  3. Autostart at login:   powershell -File scripts\start_jarvis.ps1 -RegisterAutostart"
Write-Host "  4. Say 'Hey Jarvis'. Edit config.toml to tune the wake threshold, voice or model."
Write-Host "Note: the first run downloads Whisper and Kokoro models, so it takes a while."
