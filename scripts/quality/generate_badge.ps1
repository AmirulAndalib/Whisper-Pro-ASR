# Local Test Runner with Coverage Badge Generation
# Usage: .\scripts\quality\generate_badge.ps1

$ErrorActionPreference = "Stop"

# Anchor execution to repository root (two levels up from this script)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot
Write-Host "Running Tests with Coverage" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Run pytest with coverage
python -m pytest --cov=. --cov-report=term-missing --cov-report=xml --cov-fail-under=90

if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed or coverage below 90%!" -ForegroundColor Red
    exit 1
}

# Verify per-file coverage threshold
python tests/check_coverage.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Per-file coverage check failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generating Coverage Badge" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Ensure required badge dependency is available.
$badgeDepOk = $true
python -c "import coverage_badge, pkg_resources" 2>$null
if ($LASTEXITCODE -ne 0) {
    $badgeDepOk = $false
}

if (-not $badgeDepOk) {
    Write-Host "coverage_badge/pkg_resources dependency is missing. Attempting auto-install..." -ForegroundColor Yellow
    python -m pip install --disable-pip-version-check "coverage-badge==1.1.2" "setuptools==80.9.0"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to auto-install badge dependencies (coverage_badge, setuptools)." -ForegroundColor Red
        exit 1
    }
}

python -c "import coverage_badge, pkg_resources" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Badge dependencies are still unavailable after auto-install." -ForegroundColor Red
    exit 1
}

# Generate the SVG badge
python -m coverage_badge -o assets/coverage.svg -f

if ($LASTEXITCODE -eq 0 -and (Test-Path "assets/coverage.svg") -and ((Get-Item "assets/coverage.svg").Length -gt 0)) {
    Write-Host "Coverage badge generated: assets/coverage.svg" -ForegroundColor Green
}
else {
    Write-Host "Failed to generate coverage badge!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done! Coverage badge saved to assets/coverage.svg" -ForegroundColor Green
