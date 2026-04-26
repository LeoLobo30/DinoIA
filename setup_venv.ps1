param(
    [string]$PythonExe = $null,
    [switch]$UseCuda
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Candidate)

    if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch "WindowsApps") {
        return $pythonCommand.Source
    }

    $scoopPython = "C:\Users\leona\scoop\apps\python\3.14.4\python.exe"
    if (Test-Path -LiteralPath $scoopPython) {
        return $scoopPython
    }

    throw "Could not find a usable Python executable. Pass -PythonExe or install Python locally."
}

$python = Resolve-PythonExe -Candidate $PythonExe
$venvPath = Join-Path $PSScriptRoot ".venv"

Write-Host "Using Python: $python"
Write-Host "Creating venv at: $venvPath"

& $python -m venv $venvPath

$pip = Join-Path $venvPath "Scripts\python.exe"
Write-Host "Upgrading pip inside the venv..."
& $pip -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
& $pip -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

if ($UseCuda) {
    Write-Host "Installing PyTorch GPU build..."
    & $pip -m pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Installing PyTorch CPU build..."
    & $pip -m pip install torch torchvision torchaudio
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
