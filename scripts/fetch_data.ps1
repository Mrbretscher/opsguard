$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python scripts/fetch_ai4i.py --output data/raw/ai4i2020.csv
if ($LASTEXITCODE -ne 0) {
    throw "Data fetch failed."
}
