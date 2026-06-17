param(
    [string]$PythonVersion = "3.11",
    [string]$VenvPath = ".venv",
    [string]$CudaWheel = "cu124",
    [string]$LlamaCppVersion = "0.3.30",
    [switch]$SkipDoctor
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message"
}

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "This script is for Windows. On Linux, install llama-cpp-python with CMAKE_ARGS='-DGGML_CUDA=on'."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Step "Checking Python $PythonVersion"
$pythonInfo = & py "-$PythonVersion" -c "import sys; print(sys.executable); print(sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion was not found. Install Python 3.11 or 3.12, then rerun this script."
}
$pythonInfo | ForEach-Object { Write-Host "    $_" }

if (-not (Test-Path $VenvPath)) {
    Step "Creating virtual environment at $VenvPath"
    & py "-$PythonVersion" -m venv $VenvPath
}
else {
    Step "Reusing virtual environment at $VenvPath"
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python not found at $venvPython"
}

Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Step "Installing llama-cpp-python CUDA wheel ($CudaWheel)"
& $venvPython -m pip install `
    "llama-cpp-python==$LlamaCppVersion" `
    --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$CudaWheel" `
    --force-reinstall `
    --no-cache-dir

Step "Installing c3po in editable mode"
& $venvPython -m pip install -e ".[dev,cuda-win]"

if (-not $SkipDoctor) {
    Step "Running c3po doctor"
    & (Join-Path $VenvPath "Scripts\c3po.exe") doctor
}

Write-Host ""
Write-Host "Done. Use this shell command before running c3po:"
Write-Host "    .\$VenvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Or run c3po directly:"
Write-Host "    .\$VenvPath\Scripts\c3po.exe doctor"
