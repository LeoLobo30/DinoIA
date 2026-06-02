param(
    [string]$PythonExe = $null,
    [switch]$AllowUnsupportedPython,
    [string]$TorchIndexUrl = $null
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Candidate)

    if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $py314 = & $pyLauncher.Source -3.14 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $py314 -and (Test-Path -LiteralPath $py314)) {
                return (Resolve-Path -LiteralPath $py314).Path
            }
        } catch {
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch "WindowsApps") {
        return $pythonCommand.Source
    }

    throw "Could not find a usable Python executable. Pass -PythonExe or install Python 3.14.4 locally."
}

function Get-PythonVersion {
    param([string]$Executable)
    return (& $Executable -c "import platform; print(platform.python_version())").Trim()
}

$python = Resolve-PythonExe -Candidate $PythonExe
$pythonVersion = Get-PythonVersion -Executable $python
if ($pythonVersion -ne "3.14.4" -and -not $AllowUnsupportedPython) {
    throw "DinoIA is pinned to Python 3.14.4. Found Python $pythonVersion at '$python'. Install Python 3.14.4, pass -PythonExe, or rerun with -AllowUnsupportedPython."
}

$venvPath = Join-Path $PSScriptRoot ".venv"

Write-Host "Using Python: $python"
Write-Host "Python version: $pythonVersion"
Write-Host "Creating venv at: $venvPath"

& $python -m venv --clear $venvPath
if ($LASTEXITCODE -ne 0) {
    throw "venv creation failed with exit code $LASTEXITCODE."
}

$pip = Join-Path $venvPath "Scripts\python.exe"
Write-Host "Upgrading pip inside the venv..."
& $pip -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}

Write-Host "Installing project dependencies..."
& $pip -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "dependency installation failed with exit code $LASTEXITCODE."
}

if ($TorchIndexUrl) {
    Write-Host "Installing CUDA-enabled PyTorch from: $TorchIndexUrl"
    & $pip -m pip uninstall -y torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) {
        throw "torch uninstall failed with exit code $LASTEXITCODE."
    }
    & $pip -m pip install --no-cache-dir torch torchvision torchaudio --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Running a quick doctor check..."
try {
    & $pip -m dinoia doctor
} catch {
    Write-Host "Doctor check could not run automatically."
    Write-Host $_.Exception.Message
}

Write-Host ""
Write-Host "Done."
Write-Host "Next step: .\.venv\Scripts\Activate.ps1"
