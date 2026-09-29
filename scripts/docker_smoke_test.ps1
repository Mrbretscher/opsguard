param(
    [string] $ImageName = "opsguard-gui-demo:local",
    [int] $Port = 8501
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install Docker Desktop or skip this smoke test."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running or is not reachable from this shell."
}

docker build -t $ImageName .
if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

$containerName = "opsguard-gui-smoke-$([Guid]::NewGuid().ToString('N').Substring(0, 12))"
$containerId = $null
$passed = $false

try {
    $containerId = docker run --rm -d --name $containerName -p "${Port}:8501" $ImageName
    if ($LASTEXITCODE -ne 0 -or -not $containerId) {
        throw "Docker container failed to start."
    }

    $deadline = (Get-Date).AddSeconds(60)
    do {
        try {
            $response = Invoke-WebRequest `
                -UseBasicParsing `
                -Uri "http://localhost:$Port/_stcore/health" `
                -TimeoutSec 3
            if ($response.Content.Trim() -eq "ok") {
                Write-Host "Docker GUI smoke test passed on http://localhost:$Port."
                $passed = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $deadline)

    if (-not $passed) {
        docker logs $containerName
        throw "Timed out waiting for the Streamlit health endpoint."
    }
}
finally {
    if ($containerId) {
        docker stop $containerName *> $null
    }
}
