#!/usr/bin/env python3
import logging
import os
import re
import sys
from collections.abc import Iterator

type Violation = tuple[int, str, str]

# Directory exclusions
EXCLUDE_DIRS = {
    "node_modules",
    ".venv",
    "venv",
    ".git",
    ".ruff_cache",
    "state",
    "model_cache",
    "reports",
    "coverage-js",
    "test-results",
    "__pycache__",
}

# File extensions to scan
EXTENSIONS = {".py", ".js", ".cjs", ".mjs", ".css", ".html", ".md", ".sh", ".ps1", ".yml", ".yaml", ".toml"}
SPECIAL_FILENAMES = {"Dockerfile", "Dockerfile.test"}

# Regex patterns for suppression comments
PATTERNS = {
    "eslint-disable": re.compile(r"eslint-disable"),
    "stylelint-disable": re.compile(r"stylelint-disable"),
    "markdownlint-disable": re.compile(r"markdownlint-disable"),
    "shellcheck-disable": re.compile(r"shellcheck\s+disable"),
    "type-ignore": re.compile(r"#\s*type:\s*ignore"),
    "nosec": re.compile(r"#\s*nosec\b"),
    "noqa": re.compile(r"#\s*noqa\b"),
    "pylint-disable": re.compile(r"pylint:\s*disable"),
    "coverage-ignore": re.compile(r"pragma:\s*no\s*cover|istanbul\s+ignore|c8\s+ignore"),
    "isort-skip": re.compile(r"#\s*isort:\s*(skip|off)"),
    "formatter-ignore": re.compile(r"prettier-ignore|ruff:\s*noqa"),
    "html-validate-disable": re.compile(r"html-validate-disable"),
}

logger = logging.getLogger(__name__)


def should_scan_file(filepath: str) -> bool:
    if os.path.basename(filepath) == "check-inline-ignores.py":
        return False
    if os.path.basename(filepath) in SPECIAL_FILENAMES:
        return True
    return os.path.splitext(filepath)[1] in EXTENSIONS


def _should_scan_file(filepath: str) -> bool:
    """Backward-compatible alias for internal callers."""
    return should_scan_file(filepath)


def _scan_files_in_dir(root: str, files: list[str]) -> list[str]:
    return [os.path.join(root, filename) for filename in files if should_scan_file(os.path.join(root, filename))]


def _iter_scan_targets(root_dir: str) -> Iterator[str]:
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        yield from _scan_files_in_dir(root, files)


def _report_violations(root_dir: str, filepath: str, violations: list[Violation]) -> None:
    rel_path = os.path.relpath(filepath, root_dir)
    logger.error("Violations in %s:", rel_path)
    for line_num, name, line in violations:
        logger.error("  Line %d: [%s] %s", line_num, name, line)


def scan_file(filepath: str) -> list[Violation]:
    violations: list[Violation] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for name, pattern in PATTERNS.items():
                    if pattern.search(line):
                        violations.append((line_num, name, line.strip()))
    except Exception as exc:
        logger.error("Error reading %s: %s", filepath, exc)
        raise RuntimeError(f"Failed to scan file: {filepath}") from exc
    return violations


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    has_violations = False

    for filepath in _iter_scan_targets(root_dir):
        violations = scan_file(filepath)
        if violations:
            has_violations = True
            _report_violations(root_dir, filepath, violations)

    if has_violations:
        logger.error("Error: Inline suppressions / ignores are disallowed under zero-suppression policy.")
        sys.exit(1)
    logger.info("Success: No inline suppressions found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
