# Frontend Quality Gates Skill

Use this skill whenever dashboard HTML, JavaScript, or CSS files are added/changed.

## Objective

Keep frontend quality gates deterministic and enforceable in local runs and CI.

## Scope

- HTML in `modules/monitoring/templates/*.html`
- JavaScript in `modules/monitoring/templates/*.js`
- CSS in `static/**/*.css` and `modules/monitoring/templates/**/*.css`
- JS tests in `tests/js/**/*.test.js`

## Required Gates

1. HTML lint: `npm run lint:html`
2. JavaScript lint: `npm run lint:js`
3. CSS lint: `npm run lint:css`
4. JS tests + coverage: `npm run test:js`
5. Playwright E2E: `npm run test:e2e`
6. Frontend security audit: `npm audit --audit-level=low`
7. Aggregate gate: `npm run quality:frontend`

## Tooling Policy

- Use Playwright via CLI (`npx playwright ...`) or npm scripts that wrap the Playwright CLI.
- Use MCP browser tooling to inspect DOM state, selectors, and runtime page data when diagnosing flaky or unexpected frontend behavior.
- Do not treat manual browser clicks/visual checks as a substitute for Playwright CLI and MCP-backed validation.

## Coverage Policy

- Enforce per-file coverage for monitored JS files.
- Minimum threshold: 90% for `lines` and `statements` per file.
- CI must fail if any monitored JS file drops below the threshold.

## CI Integration

- Ensure `.github/workflows/ci.yml` executes frontend gates through the `Dockerfile.test` test image (`tests/run_suite.sh`), not host Node steps.
- Keep local parity scripts (`scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`) aligned with the same Docker test-image frontend gate path.
- The Docker test image must include Node/npm dependencies and Playwright Chromium required for frontend checks.

## Test Strategy Guidance

- Favor deterministic unit tests with mocked DOM, fetch, timers, and charting APIs.
- Keep template HTML structurally valid and lint-clean alongside JS/CSS changes.
- Add branch-targeted tests for queue/task rendering, telemetry chart updates, and export/download paths.
- Avoid disabling lint rules or lowering thresholds to bypass regressions.

## Done Criteria

- `npm run quality:frontend` passes locally.
- `npm run test:e2e` passes locally.
- `npm audit --audit-level=low` passes locally.
- CI job executes same frontend gates.
- README and relevant `.agent` docs reflect any changes to commands/policies.
