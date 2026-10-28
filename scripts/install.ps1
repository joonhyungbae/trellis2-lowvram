# One-step installer for Windows. Called by install.bat. Everything goes inside this folder
# (.tools, .venv, models, logs); delete the folder to uninstall.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$UvVersion = "0.11.8"
$UpstreamCommit = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$Log = "logs\install-$(Get-Date -Format yyyyMMdd-HHmmss).log"
Start-Transcript -Path $Log -Append | Out-Null

$Step = 0

# Translated text needs Python, which exists once uv is installed (step 2 on).
function Msg($key, $param) {
    $env:PYTHONPATH = "$Root\src"
    $out = & uv run -q --no-project --managed-python --python 3.11 python -m trellis2_lowvram.i18n $key $param 2>$null
    $ok = $LASTEXITCODE -eq 0
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if ($ok -and $out) { Write-Host $out; return $true }
    return $false
}

function Fail($msg) {
    Write-Host ""
    Write-Host "[FAIL] step ${Step}: $msg" -ForegroundColor Red
    if ($Step -le 1 -or -not (Msg "install.step$Step.fail")) {
        Write-Host "  [1단계: uv 받기] astral.sh 에서 설치 도구를 받지 못했습니다. 인터넷 연결과, 학교·회사 네트워크라면 astral.sh·github.com 이 막혀 있지 않은지 확인해 주세요. 백신 프로그램이 막았을 수도 있습니다."
        Write-Host "  [Step 1: uv] Could not download the installer tool from astral.sh. Check your internet connection and, on a school or office network, that astral.sh and github.com are not blocked. Your antivirus may also have blocked it."
    }
    Write-Host ""
    if (-not (Msg "install.retry")) { Write-Host "Run install.bat again after fixing it. / 고친 뒤 install.bat 을 다시 실행해 주세요." }
    if (-not (Msg "install.log" "log=$Root\$Log")) { Write-Host "log: $Root\$Log" }
    Stop-Transcript | Out-Null
    exit 1
}

# Native commands do not throw on a non-zero exit code; check every one.
function Run($what) {
    & $args[0] $args[1..($args.Length - 1)]
    if ($LASTEXITCODE -ne 0) { Fail "$what (exit $LASTEXITCODE)" }
}

$env:UV_INSTALL_DIR = "$Root\.tools\uv"
$env:UV_PYTHON_INSTALL_DIR = "$Root\.tools\python"
$env:UV_CACHE_DIR = "$Root\.tools\uv-cache"
$env:UV_NO_MODIFY_PATH = "1"
$env:PYTHONUTF8 = "1"
$env:PATH = "$env:UV_INSTALL_DIR;$env:PATH"

try {
    $Step = 1; Write-Host "== 1/6 uv $UvVersion"
    $have = ""
    if (Test-Path "$env:UV_INSTALL_DIR\uv.exe") { $have = & "$env:UV_INSTALL_DIR\uv.exe" --version }
    if ($have -notmatch [regex]::Escape($UvVersion)) {
        Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
    }
    Run "uv" uv --version

    $Step = 2; Write-Host "== 2/6 PC check (before the large download)"
    $env:PYTHONPATH = "$Root\src"
    & uv run --no-project --managed-python --python 3.11 --with nvidia-ml-py --with psutil python -m trellis2_lowvram doctor --pre
    if ($LASTEXITCODE -ne 0) { Fail "PC check" }
    Remove-Item Env:PYTHONPATH

    $Step = 3; Write-Host "== 3/6 TRELLIS.2 source @ $($UpstreamCommit.Substring(0,7))"
    if (-not (Test-Path "third_party\TRELLIS.2\trellis2\__init__.py")) {
        if ((Test-Path ".git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
            Run "git submodule" git submodule update --init third_party/TRELLIS.2
        } else {
            # No git needed: fetch the pinned commit as a zip.
            $zip = "$Root\.tools\trellis2.zip"
            New-Item -ItemType Directory -Force -Path .tools | Out-Null
            Invoke-WebRequest "https://github.com/microsoft/TRELLIS.2/archive/$UpstreamCommit.zip" -OutFile $zip
            Expand-Archive $zip -DestinationPath ".tools\trellis2-src" -Force
            New-Item -ItemType Directory -Force -Path third_party | Out-Null
            Remove-Item -Recurse -Force "third_party\TRELLIS.2" -ErrorAction SilentlyContinue
            Move-Item ".tools\trellis2-src\TRELLIS.2-$UpstreamCommit" "third_party\TRELLIS.2"
            Remove-Item -Recurse -Force ".tools\trellis2-src", $zip
        }
    }

    $Step = 4; Write-Host "== 4/6 Python packages (PyTorch etc., several GB)"
    # A running web UI / make3d window locks files in .venv and the update would fail halfway.
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='trellis2-lowvram.exe'" |
        Where-Object { $_.ExecutablePath -like "$Root\.venv\*" -or $_.CommandLine -like "*$Root\.venv\*" }
    if ($busy) {
        Msg "install.close_windows" | Out-Null
        Fail "trellis2-lowvram is still running (PID $($busy.ProcessId -join ', '))"
    }
    Run "uv sync" uv sync --frozen --no-dev --python 3.11

    $Step = 5; Write-Host "== 5/6 CUDA extensions"
    Run "install-extensions" uv run --frozen --no-dev trellis2-lowvram install-extensions

    $Step = 6; Write-Host "== 6/6 Hugging Face access (DINOv3) and models"
    & uv run --frozen --no-dev trellis2-lowvram download
    if ($LASTEXITCODE -ne 0) {
        # Usually: no token or not approved yet. login explains the steps; then try once more.
        & uv run --frozen --no-dev trellis2-lowvram login
        & uv run --frozen --no-dev trellis2-lowvram download
        if ($LASTEXITCODE -ne 0) { Write-Host ""; Msg "install.step6.fail" | Out-Null }
    }

    & uv run --frozen --no-dev trellis2-lowvram doctor --json "logs\doctor.json"
    Write-Host ""
    Write-Host "로그 / log: $Root\$Log"
} catch {
    Fail $_.Exception.Message
}
Stop-Transcript | Out-Null
