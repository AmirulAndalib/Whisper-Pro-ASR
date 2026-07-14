# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in PowerShell from the project root.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $root

function Install-WithApt {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Packages
    )
    if (-not (Get-Command apt-get -ErrorAction SilentlyContinue)) {
        return $false
    }

    if (Get-Command sudo -ErrorAction SilentlyContinue) {
        & sudo -n apt-get update
        $updateExit = $LASTEXITCODE
        if ($updateExit -ne 0) {
            return $false
        }

        & sudo -n apt-get install -y @Packages
        $installExit = $LASTEXITCODE
        return ($installExit -eq 0)
    }

    & apt-get update
    $updateExit = $LASTEXITCODE
    if ($updateExit -ne 0) {
        return $false
    }

    & apt-get install -y @Packages
    $installExit = $LASTEXITCODE
    return ($installExit -eq 0)
}

function Ensure-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName,
        [Parameter(Mandatory = $true)]
        [string[]]$AptPackages
    )

    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "Dependency '$CommandName' is missing. Attempting auto-install..."
    $installed = Install-WithApt -Packages $AptPackages
    if (-not $installed -or -not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required dependency '$CommandName' is missing and could not be auto-installed. Install it manually and rerun."
    }
}

Ensure-Command -CommandName docker -AptPackages @("docker.io")

$hasHostNpm = $true
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "Dependency 'npm' is missing. Attempting auto-install..."
    try {
        $installed = Install-WithApt -Packages @("nodejs", "npm")
        if ($installed -and (Get-Command npm -ErrorAction SilentlyContinue)) {
            $hasHostNpm = $true
        } else {
            Write-Host "npm is not expected or available on the host. Will use transient Node container approach."
            $hasHostNpm = $false
        }
    } catch {
        Write-Host "npm is not expected or available on the host. Will use transient Node container approach."
        $hasHostNpm = $false
    }
}

# Auto-detect if sudo is needed for Docker commands on Unix platforms
# We check version to safely reference $IsLinux/$IsMacOS under Set-StrictMode
$isUnix = $false
if ($PSVersionTable.PSVersion.Major -ge 6) {
    $isUnix = $IsLinux -or $IsMacOS
}

$dockerExe = "docker"
if ($isUnix) {
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "docker"
    $processInfo.Arguments = "ps"
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    try {
        $process = [System.Diagnostics.Process]::Start($processInfo)
        # Read stderr before WaitForExit to prevent pipe-buffer deadlock
        $null = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) {
            if (Get-Command sudo -ErrorAction SilentlyContinue) {
                $sudoProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
                $sudoProcessInfo.FileName = "sudo"
                $sudoProcessInfo.Arguments = "-n docker ps"
                $sudoProcessInfo.RedirectStandardError = $true
                $sudoProcessInfo.UseShellExecute = $false
                $sudoProcess = [System.Diagnostics.Process]::Start($sudoProcessInfo)
                # Read stderr before WaitForExit to prevent pipe-buffer deadlock
                $null = $sudoProcess.StandardError.ReadToEnd()
                $sudoProcess.WaitForExit()
                if ($sudoProcess.ExitCode -eq 0) {
                    $dockerExe = "sudo"
                    Write-Host "Note: Prepended 'sudo' to docker commands because of permission check on /var/run/docker.sock"
                } else {
                    throw "Docker permission check failed (docker ps exited with non-zero, and sudo docker ps also failed)."
                }
            } else {
                throw "Docker permission check failed (docker ps exited with non-zero, and sudo is not available)."
            }
        }
    } catch {
        throw "Failed to verify Docker access: $_"
    }
}

function Invoke-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    if ($dockerExe -eq "sudo") {
        & sudo docker @Arguments
    } else {
        & docker @Arguments
    }

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Docker command failed with exit code ${exitCode}: docker $($Arguments -join ' ')"
    }

    return $exitCode
}

function Update-PoetryLock {
    Write-Host "`n--- Verifying Poetry Lock File ---"

    $arguments = @("run", "--rm")
    if ($isUnix) {
        $uid = (& id -u).Trim()
        $gid = (& id -g).Trim()
        if ($LASTEXITCODE -eq 0 -and $uid -and $gid) {
            $arguments += @("--user", "${uid}:${gid}")
        }
    }

    $arguments += @(
        "-v", "${root}:/workspace",
        "-w", "/workspace",
        "python:3.12-slim",
        "/bin/bash",
        "-lc",
        'export HOME=/tmp && python -m pip install --quiet --user poetry && if [ ! -f poetry.lock ] || ! python -m poetry check --lock >/dev/null 2>&1; then python -m poetry lock --no-interaction; fi'
    )

    Invoke-Docker -- @arguments
}

function Ensure-FrontendTooling {
    Write-Host "`n--- Ensuring Playwright CLI + MCP Tooling Dependencies ---"

    $nodeMajor = [int]((& node -p "process.versions.node.split('.')[0]").Trim())
    if ($nodeMajor -lt 20) {
        throw "Node.js 20+ is required for Playwright MCP CLI. Current version: $((& node -v).Trim())"
    }

    $nodeModulesPath = Join-Path $root "node_modules"
    $needsNpmInstall = -not (Test-Path $nodeModulesPath)
    if (-not $needsNpmInstall) {
        & npm ls --depth=0 *> $null
        $needsNpmInstall = ($LASTEXITCODE -ne 0)
    }

    if ($needsNpmInstall) {
        & npm ci
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install npm dependencies required for Playwright CLI and MCP tooling."
        }
    }

    & npx playwright --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Playwright CLI is unavailable after npm dependency install."
    }

    # Idempotent browser bootstrap for local Playwright CLI execution.
    & npx playwright install chromium *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Playwright Chromium browser dependency."
    }

    & npx @playwright/mcp --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Playwright MCP CLI is unavailable. Ensure '@playwright/mcp' exists in devDependencies."
    }
}

Update-PoetryLock

Write-Host "`n--- Running Frontend Validation ---"
if ($hasHostNpm) {
    & npm ci
    if ($LASTEXITCODE -ne 0) {
        throw "npm ci failed"
    }
    & npm audit --audit-level=low
    if ($LASTEXITCODE -ne 0) {
        throw "npm audit failed: vulnerabilities detected"
    }
    Ensure-FrontendTooling
} else {
    $arguments = @("run", "--rm")
    if ($isUnix) {
        $uid = (& id -u).Trim()
        $gid = (& id -g).Trim()
        if ($LASTEXITCODE -eq 0 -and $uid -and $gid) {
            $arguments += @("--user", "${uid}:${gid}")
        }
    }
    $arguments += @(
        "-v", "${root}:/workspace",
        "-w", "/workspace",
        "-e", "HOME=/tmp",
        "node:20-slim",
        "/bin/bash",
        "-lc",
        "npm ci && npm audit --audit-level=low"
    )
    Invoke-Docker -- @arguments
}

Write-Host "`n--- Running Static Analysis (Linting) ---"
# Build the lint stage specifically. This uses Docker layer caching for both dependencies and results.
Invoke-Docker -- build --progress=plain -f Dockerfile.test --target lint -t whisper-pro-asr-lint .

Write-Host "`n--- Building Test Image ---"
# Build the final test image.
Invoke-Docker -- build --progress=plain -f Dockerfile.test --target test -t whisper-pro-asr-test .

Write-Host "`n--- Execute Test Suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
$reportsDir = Join-Path $root "reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
Invoke-Docker -- run --rm `
    -e "CI=true" `
  -v "${root}/assets:/app/assets" `
    -v "${reportsDir}:/reports" `
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp complexity_output.txt /reports/complexity_output.txt && cp pytest.xml /reports/pytest.xml"

$badgePath = Join-Path $root "assets/coverage.svg"
if (-not (Test-Path $badgePath)) {
        throw "Mandatory coverage badge is missing at assets/coverage.svg"
}

$badgeInfo = Get-Item $badgePath
if ($badgeInfo.Length -le 0) {
        throw "Mandatory coverage badge is empty at assets/coverage.svg"
}

Write-Host "`n--- Code Coverage Summary ---"
$covFile = Join-Path $reportsDir "coverage_output.txt"
if (Test-Path $covFile) {
    $covContent = Get-Content $covFile
    $printing = $false
    foreach ($line in $covContent) {
        if ($line -match "---------- coverage") { $printing = $true }
        if ($printing) { Write-Host $line }
        if ($line -match "TOTAL" -and $printing) { $printing = $false }
    }
}

Write-Host "`n--- Cyclomatic Complexity Summary (Radon cc) ---"
$complexityFile = Join-Path $reportsDir "complexity_output.txt"
if (Test-Path $complexityFile) {
    Get-Content $complexityFile | Write-Host
}

Write-Host "`n--- Done ---"
