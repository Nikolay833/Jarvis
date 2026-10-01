# Launch Jarvis core and the orb overlay.
#   powershell -File scripts\start_jarvis.ps1                      start now
#   powershell -File scripts\start_jarvis.ps1 -RegisterAutostart   also run at login
#   powershell -File scripts\start_jarvis.ps1 -UnregisterAutostart
param(
    [switch]$RegisterAutostart,
    [switch]$UnregisterAutostart,
    [switch]$NoOrb
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "Jarvis"

if ($UnregisterAutostart) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Autostart removed."
    return
}

if ($RegisterAutostart) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSCommandPath`"" `
        -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "Starts the Jarvis voice assistant at login" -Force | Out-Null
    Write-Host "Autostart registered (task '$TaskName'). Remove with -UnregisterAutostart."
    return
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\setup_windows.ps1 first." }

# Orb overlay: built release binary if present, else skip with a hint.
$orbProc = $null
if (-not $NoOrb) {
    $orbExe = Get-ChildItem -Path (Join-Path $Root "orb\src-tauri\target\release") -Filter "*.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "setup|uninstall" } | Select-Object -First 1
    if ($orbExe) {
        $orbProc = Start-Process -FilePath $orbExe.FullName -PassThru
    } else {
        Write-Warning "Orb not built yet (cd orb; npm install; npm run tauri build). Starting core only."
    }
}

try {
    Set-Location $Root
    & $Py -m jarvis
} finally {
    if ($orbProc -and -not $orbProc.HasExited) { Stop-Process -Id $orbProc.Id -Force }
}
