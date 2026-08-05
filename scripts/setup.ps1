param(
    [string]$Python = "py",
    [string[]]$PythonArgs = @("-3.11"),
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

$workspace = (Resolve-Path ".").Path
$localTemp = Join-Path $workspace ".tmp"
New-Item -ItemType Directory -Force -Path $localTemp | Out-Null
$env:TEMP = $localTemp
$env:TMP = $localTemp

if ($Recreate -and (Test-Path ".venv")) {
    $venvPath = (Resolve-Path ".venv").Path
    if (-not $venvPath.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove virtual environment outside the workspace: $venvPath"
    }
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path ".venv")) {
    & $Python @($PythonArgs + @("-m", "venv", ".venv"))
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Standard venv creation failed; retrying with --without-pip."
        Invoke-Checked $Python ($PythonArgs + @("-m", "venv", "--without-pip", ".venv"))
        $bundledPip = & $Python @($PythonArgs + @("-c", "import ensurepip, pathlib; print(pathlib.Path(ensurepip.__file__).parent / '_bundled')"))
        if ($LASTEXITCODE -ne 0) {
            throw "Could not locate bundled ensurepip wheels."
        }
        Invoke-Checked $Python ($PythonArgs + @("-m", "pip", "--python", ".venv", "install", "--no-index", "--find-links", $bundledPip, "pip"))
    }
}

Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "-e", ".[dev]")

Write-Output "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
