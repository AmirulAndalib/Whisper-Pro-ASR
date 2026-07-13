#!/bin/bash
# Script to run the test suite and linting in Docker
set -e
set -o pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment if running locally and it exists
if [ "$CI" != "true" ]; then
  if [ -d ".venv" ]; then
    source .venv/bin/activate
  elif [ -d "venv" ]; then
    source venv/bin/activate
  fi
fi


if [ "$SKIP_LINT" != "1" ]; then
  echo "--- Running Markdown Lint ---"
  npm run lint:md

  echo "--- Running Frontend Dependency Integrity ---"
  npm audit --audit-level=low

  echo "--- Running Frontend Quality Gates ---"
  npm run quality:frontend
else
  echo "--- Skipping Frontend Gates (SKIP_LINT=1) ---"
fi

echo ""
echo "--- Running Pytest with Coverage ---"
# We output XML (for PR display) and terminal report
python3 -m pytest -ra --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing --junitxml=pytest.xml | tee coverage_output.txt

echo ""
echo "--- Verifying Per-File Coverage (Threshold: 90%) ---"
python3 tests/check_coverage.py
echo ""
echo "--- Generating Coverage Badge ---"
# Ensure mandatory badge-generation dependencies are present.
if ! PYTHONWARNINGS="ignore::DeprecationWarning" python3 -c "import coverage_badge, pkg_resources" >/dev/null 2>&1; then
  echo "coverage_badge/pkg_resources dependency is missing. Attempting auto-install..."
  python3 -m pip install --disable-pip-version-check "coverage-badge==1.1.2" "setuptools==80.9.0"
fi

if ! PYTHONWARNINGS="ignore::DeprecationWarning" python3 -c "import coverage_badge, pkg_resources" >/dev/null 2>&1; then
  echo "Error: Failed to auto-install required badge dependencies (coverage_badge, pkg_resources)."
  exit 1
fi

# Create assets directory if it doesn't exist
mkdir -p assets
python3 -m coverage_badge -o assets/coverage.svg -f

if [ ! -s assets/coverage.svg ]; then
  echo "Error: Coverage badge was not generated or is empty at assets/coverage.svg"
  exit 1
fi

echo ""
echo "--- Running Cyclomatic Complexity Check (Radon) ---"
python3 -m radon cc -s -a modules whisper_pro_asr.py | tee complexity_output.txt

echo ""
echo "--- Test Suite Completed Successfully ---"

