# Jarvis setup for Windows 11 / 10. Run from the repo root or anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-Model qwen3:14b] [-Parakeet] [-SkipVision] [-Webcam]
param(
    [string]$Model = "qwen3:14b",
    [string]$Python = "py -3.11",
    [switch]$Parakeet,  # also install the optional NVIDIA Parakeet STT engine (onnx-asr)
    [string]$VisionModel = "qwen2.5vl:3b",  # screen vision model (~3 GB); keep in sync with [vision] model
    [switch]$SkipVision,  # do not pull the vision model (look_at_screen then needs a manual ollama pull)
    [switch]$Webcam  # also install OpenCV for the optional webcam tool (still off until [vision] webcam_enabled)
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
if ($Parakeet) {
    Write-Host "==> Installing Parakeet STT (onnx-asr)"
    & $Py -m pip install "onnx-asr[gpu,hub]"
}

if ($Webcam) {
    Write-Host "==> Installing OpenCV (webcam tool)"
    & $Py -m pip install -e ".[webcam]"
}

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

if (-not $SkipVision) {
    # Pulled by default: "what's on my screen" needs it. The small 3b model fits next to qwen3:14b + Whisper;
    # it unloads 2 minutes after use ([vision] keep_alive).
    Write-Host "==> Pulling Ollama vision model $VisionModel"
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        ollama pull $VisionModel
    } else {
        Write-Warning "Ollama not found. After installing it run: ollama pull $VisionModel"
    }
}

Write-Host "==> Setting Ollama speed options (user environment variables)"
# Flash attention + q8 KV cache: faster prompt processing, about half the KV cache VRAM.
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
Write-Warning "Ollama must be restarted to pick these up: quit it from the system tray and reopen it."

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Warning "Claude Code CLI not found on PATH (only needed for the Claude Code tools). Set claude_code.binary in config.toml."
}

Write-Host "==> Installing Claude Code hooks (Jarvis announces when Claude finishes or needs permission)"
& $Py "scripts\install_claude_hooks.py"
Write-Host "    Edits ~/.claude/settings.json (backup written first). Undo: $Py scripts\install_claude_hooks.py --uninstall"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Test without audio:   .venv\Scripts\python -m jarvis --text"
Write-Host "  2. Start everything:     powershell -File scripts\start_jarvis.ps1"
Write-Host "  3. Autostart at login:   powershell -File scripts\start_jarvis.ps1 -RegisterAutostart"
Write-Host "  4. Say 'Hey Jarvis'. Edit config.toml to tune the wake threshold, voice or model."
Write-Host "Note: the first run downloads Whisper and Kokoro models, so it takes a while."
