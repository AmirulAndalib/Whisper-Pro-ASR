#!/bin/bash

# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in bash from the project root.

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve script directory in a POSIX-safe way (works with sh and bash).
SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$(pwd)/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

install_with_apt() {
  packages="$1"
  if command -v sudo >/dev/null 2>&1; then
    sudo -n apt-get update && sudo -n apt-get install -y $packages
    return $?
  fi

  if [ "$(id -u)" -eq 0 ]; then
    apt-get update && apt-get install -y $packages
    return $?
  fi

  return 1
}

ensure_command() {
  cmd_name="$1"
  apt_packages="$2"

  if command -v "$cmd_name" >/dev/null 2>&1; then
    return 0
  fi

  echo "Dependency '$cmd_name' is missing. Attempting auto-install..."
  if command -v apt-get >/dev/null 2>&1 && install_with_apt "$apt_packages"; then
    if command -v "$cmd_name" >/dev/null 2>&1; then
      return 0
    fi
  fi

  echo "Error: '$cmd_name' is required and could not be auto-installed."
  echo "Install it manually, then re-run this script."
  exit 1
}

ensure_command docker "docker.io"

HAS_HOST_NPM=true
if ! command -v npm >/dev/null 2>&1; then
  echo "Dependency 'npm' is missing. Attempting auto-install..."
  if command -v apt-get >/dev/null 2>&1 && install_with_apt "nodejs npm" && command -v npm >/dev/null 2>&1; then
    HAS_HOST_NPM=true
  else
    echo "npm is not expected or available on the host. Will use transient Node container approach."
    HAS_HOST_NPM=false
  fi
fi

ensure_frontend_tooling() {
  printf "\n--- Ensuring Playwright CLI + MCP Tooling Dependencies ---\n"

  node_major="$(node -p 'process.versions.node.split(".")[0]')"
  if [ "${node_major}" -lt 20 ]; then
    echo "Error: Node.js 20+ is required for Playwright MCP CLI. Current version: $(node -v)"
    exit 1
  fi

  if [ ! -d "${PROJECT_ROOT}/node_modules" ]; then
    npm ci
  elif ! npm ls --depth=0 >/dev/null 2>&1; then
    npm ci
  fi

  if ! npx playwright --version >/dev/null 2>&1; then
    echo "Error: Playwright CLI is unavailable after npm dependency install."
    exit 1
  fi

  # Idempotent browser bootstrap for local Playwright CLI execution.
  npx playwright install chromium >/dev/null

  if ! npx @playwright/mcp --help >/dev/null 2>&1; then
    echo "Error: Playwright MCP CLI is unavailable."
    echo "Run 'npm install' and ensure '@playwright/mcp' is present in devDependencies."
    exit 1
  fi
}

# Pick Docker command: direct docker when allowed, otherwise sudo docker.
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker ps >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
  else
    echo "Error: Docker is installed but not accessible for the current user."
    echo "Add your user to the docker group or enable sudo access to docker."
    exit 1
  fi
fi

ensure_poetry_lock() {
  printf "\n--- Verifying Poetry Lock File ---\n"
  user_args=""
  if command -v id >/dev/null 2>&1; then
    user_args="--user $(id -u):$(id -g)"
  fi

  # Refresh poetry.lock only when it is missing or invalid before Docker builds
  # so local/CI parity stays stable without churning the lock file unnecessarily.
  # shellcheck disable=SC2086
  $DOCKER_CMD run --rm \
    $user_args \
    -v "${PROJECT_ROOT}:/workspace" \
    -w /workspace \
    python:3.12-slim \
    /bin/bash -lc "export HOME=/tmp && python -m pip install --quiet --user poetry && if [ ! -f poetry.lock ] || ! python -m poetry check --lock >/dev/null 2>&1; then python -m poetry lock --no-interaction; fi"
}

ensure_poetry_lock

printf "\n--- Running Frontend Validation ---\n"
if [ "$HAS_HOST_NPM" = true ]; then
  npm ci
  npm audit --audit-level=low
  ensure_frontend_tooling
else
  user_args=""
  if command -v id >/dev/null 2>&1; then
    user_args="--user $(id -u):$(id -g)"
  fi
  # shellcheck disable=SC2086
  $DOCKER_CMD run --rm \
    $user_args \
    -v "${PROJECT_ROOT}:/workspace" \
    -w /workspace \
    node:20-slim \
    /bin/bash -lc "export HOME=/tmp && npm ci && npm audit --audit-level=low"
fi

printf "\n--- Running Static Analysis (Linting) ---\n"
$DOCKER_CMD build -f Dockerfile.test --target lint -t whisper-pro-asr-lint .

printf "\n--- Building Test Image ---\n"
$DOCKER_CMD build -f Dockerfile.test --target test -t whisper-pro-asr-test .

printf "\n--- Execute Test Suite ---\n"
REPORTS_DIR="${PROJECT_ROOT}/reports"
mkdir -p assets "$REPORTS_DIR"
$DOCKER_CMD run --rm \
  -e CI=true \
  -v "${PROJECT_ROOT}/assets:/app/assets" \
  -v "${REPORTS_DIR}:/reports" \
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp complexity_output.txt /reports/complexity_output.txt && cp pytest.xml /reports/pytest.xml"

if [ ! -s "${PROJECT_ROOT}/assets/coverage.svg" ]; then
  echo "Error: Mandatory coverage badge is missing or empty at assets/coverage.svg"
  exit 1
fi

printf "\n--- Code Coverage Summary ---\n"
sed -n '/---------- coverage/,/TOTAL/p' "${REPORTS_DIR}/coverage_output.txt"

printf "\n--- Cyclomatic Complexity Summary (Radon cc) ---\n"
cat "${REPORTS_DIR}/complexity_output.txt"

printf "\n--- Done ---\n"

