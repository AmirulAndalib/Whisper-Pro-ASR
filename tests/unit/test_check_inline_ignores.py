"""Tests for scripts/ci/check-inline-ignores.py."""

import importlib.util
from pathlib import Path


def _load_module():
    """Load the inline-ignore checker module from scripts/ci."""

    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check-inline-ignores.py"
    spec = importlib.util.spec_from_file_location("check_inline_ignores", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detects_dockerfile_without_extension(tmp_path):
    """Ensure Dockerfile names are scanned despite lacking extensions."""

    module = _load_module()
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("# py" + "lint: disable=unused-variable\n", encoding="utf-8")

    assert module.should_scan_file(str(dockerfile)) is True
    violations = module.scan_file(str(dockerfile))
    assert any(v[1] == "pylint-disable" for v in violations)


def test_detects_new_suppression_patterns(tmp_path):
    """Ensure newly tracked suppression patterns are reported."""

    module = _load_module()
    sample = tmp_path / "sample.py"
    noqa_line = "# " + "no" + "qa: E501"
    pylint_disable_line = "# py" + "lint: disable=too-many-branches"
    coverage_line = "# pragma:" + " no cover"
    isort_line = "# " + "isort:" + " skip"
    istanbul_line = "// istanbul " + "ignore next"
    html_validate_line = "<!-- html-" + "validate-disable -->"
    sample.write_text(
        "\n".join(
            [
                noqa_line,
                pylint_disable_line,
                coverage_line,
                isort_line,
                istanbul_line,
                html_validate_line,
            ]
        ),
        encoding="utf-8",
    )

    violations = module.scan_file(str(sample))
    names = {name for _, name, _ in violations}
    expected = {"noqa", "pylint-disable", "coverage-ignore", "isort-skip", "html-" + "validate-disable"}
    assert expected.issubset(names)


def test_scan_file_raises_on_read_failure(monkeypatch, tmp_path):
    """Ensure unreadable files raise a stable RuntimeError contract."""

    module = _load_module()
    target = tmp_path / "bad.py"
    target.write_text("print('x')", encoding="utf-8")

    def _raise_open(*_args, **_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr("builtins.open", _raise_open)

    try:
        module.scan_file(str(target))
    except RuntimeError as exc:
        assert "Failed to scan file" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unreadable file")
