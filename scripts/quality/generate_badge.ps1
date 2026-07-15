# Local Test Runner with Coverage Badge Generation
# Usage: .\scripts\quality\generate_badge.ps1

$ErrorActionPreference = "Stop"

# Anchor execution to repository root (two levels up from this script)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot
Write-Output "Running Tests with Coverage"
Write-Output "========================================"

# Run pytest with coverage
python -m pytest --cov=. --cov-report=term-missing --cov-report=xml --cov-fail-under=90

if ($LASTEXITCODE -ne 0) {
    Write-Output "Tests failed or coverage below 90%!"
    exit 1
}

# Verify per-file coverage threshold
python tests/check_coverage.py

if ($LASTEXITCODE -ne 0) {
    Write-Output "Per-file coverage check failed!"
    exit 1
}

Write-Output ""
Write-Output "========================================"
Write-Output "Generating Coverage Badge"
Write-Output "========================================"

# Generate the SVG badge from coverage.xml using genbadge.
genbadge coverage -i coverage.xml -o assets/coverage.svg

if ($LASTEXITCODE -eq 0 -and (Test-Path "assets/coverage.svg") -and ((Get-Item "assets/coverage.svg").Length -gt 0)) {
    Write-Output "Coverage badge generated: assets/coverage.svg"
}
else {
    Write-Output "Failed to generate coverage badge!"
    exit 1
}

Write-Output ""
Write-Output "Done! Coverage badge saved to assets/coverage.svg"
