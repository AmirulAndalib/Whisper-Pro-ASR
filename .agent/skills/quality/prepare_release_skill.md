# Prepare Release Skill

This skill defines the release preparation workflow for Whisper Pro ASR from an active working branch.

## Objective

Determine the target release version, verify quality gates, review documentation/release-note drift across the full change set, update docs, generate release notes, and produce a clean release-ready commit.

---

## Procedure

### 1. Identify Release Version

Determine the release version in this order:

1. If branch name matches `feature/vX.Y.Z`, use `X.Y.Z`.
2. Otherwise, if branch name matches `release/vX.Y.Z`, `feature/X.Y.Z`, or `release/X.Y.Z`, use `X.Y.Z`.
3. Otherwise, use an explicitly provided release version.
4. If neither is available, stop and request the target version before publishing release notes.

```bash
# Display active branch name
git branch --show-current
```

*Example*: On branch `feature/v1.1.6`, release tag/docs target version is `v1.1.6`.

### 2. Verify Pipeline Quality Gates

Ensure tests and lint gates are passing before release finalization:

- Run Docker-backed parity wrapper (preferred):

  ```bash
  ./scripts/ci/build-and-test.sh
  ```

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1
  ```

- Or run the explicit Docker test image/entrypoint (still Docker-only):

  ```bash
  docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
  docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" whisper-pro-asr-test /bin/bash -lc "tests/run_suite.sh"
  ```

- Required gates remain unchanged: backend tests, coverage >= 90%, flake8/pylint quality, frontend quality + npm audit, markdown lint (when markdown changed).

### 3. Update Project Documentation

Ensure all documentation files are synchronized with the actual shipped behavior:

- Review the full current commit diff first so docs and release notes are checked against every changed file, not only the most obvious feature file.
- **README.md**: Update user-visible capabilities, command snippets, and template/module tree references.
- **docs/ARCHITECTURE.md**: Keep pipeline, locking, lifecycle, and monitoring structure details accurate.
- **docs/API.md**: Keep endpoint classes, parameters, and outputs accurate.
- **docs/DOCKERHUB_DESCRIPTION.md**: Keep feature summary aligned with README.
- **.agent/** skills/workflows: Update when behavior, quality gates, or execution policy changed.

### 4. Generate GitHub Release Notes

Create or update the version-specific markdown release file at `docs/releases/GITHUB_RELEASE_v<VERSION>.md`:

- Highlight key features and structural improvements.
- Document optimizations, bug fixes, and security enhancements.
- Include verification results (backend/frontend tests, Playwright E2E, coverage, lint, audit status).

### 5. Consolidate Git Commit

A release change set should be staged and committed with a descriptive message:

- Stage all modified files and the new release markdown file:

  ```bash
  git add -A
  ```

- Create a release commit with a descriptive title and concise summary bullets:

  ```bash
  git commit -m "v1.x.y: Short Summary of Main Features

  - Detailed bullet point 1
  - Detailed bullet point 2"
  ```

- Confirm the git tree is completely clean (fail-fast check):

  ```bash
  # Check if there are any untracked, unstaged, or staged changes remaining
  if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean. Stage and commit all changes first."
    exit 1
  fi
  ```

Do not amend commits unless explicitly requested.
