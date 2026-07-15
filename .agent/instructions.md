# Agent Instructions

This file is the canonical pre-task instruction entrypoint for `.agent` assets.

## Mandatory Pre-Task Review

Before implementation:

1. Read `.agent/instructions.md`.
2. Read `.agent/skills/SKILLS_CATALOG.md`.
3. Read all directly relevant skill/workflow files for the task domain.

## Global Execution Rules

- Concurrency correctness (deadlock/livelock safety and bounded progress) takes priority over throughput optimizations.
- If behavior, architecture, APIs, quality gates, or operations change, update impacted markdown docs in the same task (`README.md`, `docs/*.md`, and relevant `.agent/*.md`).
- Keep endpoint taxonomy contract aligned everywhere:
  - Standard ASR class: `/asr`, `/v1/audio/transcriptions`, `/v1/audio/translations`
  - Priority language-ID class: `/detect-language`, `/detectlang`
- For dashboard/frontend changes, keep frontend quality-gate docs and commands synchronized, including Playwright browser prerequisites and npm audit enforcement.

## Agent Asset Maintenance

When code/process changes impact agent guidance:

- Update affected files in `.agent/skills/`.
- Update `.agent/workflows/` when flow/commands change.
- Update `.agent/skills/SKILLS_CATALOG.md` for add/remove/rename changes.
- Keep redirects valid and workspace-relative for moved skills.

## Documentation Completion Rule

Do not close a task that changes user-visible behavior, APIs, quality gates, or architecture until all impacted documentation and agent assets are synchronized.
