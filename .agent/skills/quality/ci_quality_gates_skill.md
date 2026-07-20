# CI Quality Gates Skill

Use this skill before merge/release to enforce repository standards.

## Objective

Maintain a zero-regression quality baseline.

## Required Gates

1. Pylint score: `10.00/10` for project command scope (executed inside the Docker test image via `tests/run_suite.sh`).
2. Flake8 gate pass for Python sources (`modules`, `whisper_pro_asr.py`, `tests`, `tests/check_coverage.py`) with zero Flake8 ignore directives (executed inside the Docker test image).
3. Markdown lint pass (`npm run lint:md`) via the repo-configured `markdownlint-cli2` gate (executed inside the Docker test image).
4. Test coverage: `>= 90%` (current baseline higher, verified inside the Docker test image).
5. Full test suite pass (executed inside the Docker test image).
6. No lint suppressions added as workaround.
7. Frontend quality gate pass (`npm run quality:frontend`), including HTML lint, executed exclusively inside the Docker image.
8. Frontend security audit pass: `npm audit --audit-level=low`, executed inside the Docker image.
9. Frontend Playwright E2E pass (`npm run test:e2e`), executed inside the Docker image.
10. JS per-file coverage threshold enforced at `>= 90%` for lines/statements on monitored dashboard files.
11. Concurrency-affecting changes must include liveness tests (pause/resume, queued waiting behavior, acquisition behavior) and pass related scheduler suites.
12. Concurrency-affecting changes must include synchronized documentation updates (`README.md`, `docs/CONCURRENCY.md`, and relevant `.agent/skills` files).
13. **Docker-Only execution policy**: Local parity scripts (`scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`) must not run eslint, stylelint, hadolint, shellcheck, taplo, bandit, or pip-audit on the host. They must build the Docker test image and run all tests/check output exclusively via that image.
14. Frontend verification and debugging must use Playwright CLI and MCP tooling when available; do not rely on manual-only browser validation.
15. Node.js dependencies and Playwright Chromium must be bootstrapped inside the Docker image for testing.
16. Python source code cyclomatic complexity must have 100% A-grade ranks (Radon cc rank A, score <= 5) for all functions, methods, and blocks.
17. The Docker test image pipeline must fail immediately when any rank-B-or-worse Radon output is detected.
18. `tests/run_suite.sh` inside the Docker test image must execute Radon complexity summary and rank-A enforcement before pytest and coverage generation.
19. In Docker test images, Radon source enumeration must be filesystem-based (e.g., `find ... -name '*.py'`) and must not depend on `.git` metadata.
20. Dockerfile lint gate must pass with Hadolint (`hadolint --failure-threshold warning --disable-ignore-pragma Dockerfile Dockerfile.test`) inside the Docker test image.
21. PowerShell script lint gate must pass with PSScriptAnalyzer inside the Docker test image.
22. Shell script lint gate must pass with ShellCheck inside the Docker test image.
23. CSS lint gate must run explicitly (`npm run lint:css`) inside the Docker test image.
24. HTML lint gate must run explicitly (`npm run lint:html`) inside the Docker test image.
25. Python formatter checks must run in the Docker test image using `black --check .` and `isort --check-only .`.

## Verification Commands

```bash
./scripts/ci/build-and-test.sh

# Or on Windows
powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1

# Optional explicit image invocation (still Docker-only)
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
mkdir -p reports assets
docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" whisper-pro-asr-test /bin/bash -lc "tests/run_suite.sh"
```

Lock-file contract: CI verifies `poetry.lock` before Docker builds and fails when the lockfile is missing or out of sync. Local parity scripts may still regenerate when missing or stale.

Frontend tooling contract: local parity scripts must verify Playwright CLI availability, perform idempotent browser bootstrap (`npx playwright install chromium`), and verify MCP CLI availability via `npx @playwright/mcp --help`.

GitHub Actions cache contract: the Docker test image build must use a dedicated GHA cache scope so its environment layers remain reusable across CI runs, and the production image build may read from that test-image scope without overwriting it.

## Done Criteria

- All listed verification commands pass in local environment.
- Any changed behavior has matching tests.
