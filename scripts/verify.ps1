$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "Ruff check failed." }
& $python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed." }
& $python -m mypy src/opsguard
if ($LASTEXITCODE -ne 0) { throw "Mypy failed." }
& $python -m pytest --cov=opsguard
if ($LASTEXITCODE -ne 0) { throw "Pytest failed." }
& $python -c "import opsguard; print(opsguard.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Import smoke test failed." }
