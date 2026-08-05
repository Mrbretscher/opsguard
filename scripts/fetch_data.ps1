$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m opsguard.cli fetch-data --output data/raw/ai4i2020.csv
if ($LASTEXITCODE -ne 0) {
    throw "Data fetch failed."
}
