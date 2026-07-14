# CI Quality Gates Skill

Use this skill before merge/release to enforce repository standards.

## Objective

Maintain a zero-regression quality baseline.

## Required Gates

1. Pylint score: `10.00/10` for project command scope.
2. Flake8 gate pass for Python sources (`modules`, `whisper_pro_asr.py`, `tests`, `tests/check_coverage.py`) with zero Flake8 ignore directives.
3. Markdown lint pass (`npm run lint:md`) via the repo-configured `markdownlint-cli2` gate.
4. Test coverage: `>= 90%` (current baseline higher).
5. Full test suite pass.
6. No lint suppressions added as workaround.
7. Frontend quality gate pass (`npm run quality:frontend`), including HTML lint, executed from the Docker test image.
8. Frontend security audit pass: `npm audit --audit-level=low`, executed from the Docker test image.
9. Frontend Playwright E2E pass (`npm run test:e2e`), executed from the Docker test image.
10. JS per-file coverage threshold enforced at `>= 90%` for lines/statements on monitored dashboard files.
11. Concurrency-affecting changes must include liveness tests (pause/resume, queued waiting behavior, acquisition behavior) and pass related scheduler suites.
12. Concurrency-affecting changes must include synchronized documentation updates (`README.md`, `docs/CONCURRENCY.md`, and relevant `.agent/skills` files).
13. Local parity scripts (`scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`) must stay aligned with CI gates and execute lint/tests via `Dockerfile.test` test-image pipeline.
14. Frontend verification and debugging must use Playwright CLI and MCP tooling when available; do not rely on manual-only browser validation.
15. Local parity scripts must auto-bootstrap frontend CLI dependencies (`npm ci` when needed), install Playwright Chromium for CLI runs, enforce Node.js 20+, and fail fast when `@playwright/mcp` CLI is unavailable.

## Verification Commands

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/workspace" -w /workspace python:3.12-slim /bin/bash -lc "export HOME=/tmp && python -m pip install --quiet --user poetry && if [ ! -f poetry.lock ] || ! python -m poetry check --lock; then echo 'poetry.lock is missing or out of sync. Please run poetry lock locally and commit the changes.' && exit 1; fi"
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
npm run lint:md
mkdir -p reports
docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp pytest.xml /reports/pytest.xml"
```

Lock-file contract: CI verifies `poetry.lock` before Docker builds and fails when the lockfile is missing or out of sync. Local parity scripts may still regenerate when missing or stale.

Frontend tooling contract: local parity scripts must verify Playwright CLI availability, perform idempotent browser bootstrap (`npx playwright install chromium`), and verify MCP CLI availability via `npx @playwright/mcp --help`.

## Done Criteria

- All listed verification commands pass in local environment.
- Any changed behavior has matching tests.
