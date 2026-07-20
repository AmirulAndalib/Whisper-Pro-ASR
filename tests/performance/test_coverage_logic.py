"""Comprehensive coverage for the coverage check utility."""

import xml.etree.ElementTree as ET
from unittest import mock

import pytest

from tests import check_coverage


def test_check_coverage_success(caplog):
    """Check utility prints PASS and success summary when all files meet threshold."""
    xml_content = """<coverage line-rate="0.95"><packages>
        <package name="modules"><classes>
            <class filename="modules/mod1.py" line-rate="0.95" name="mod1.py"></class>
        </classes></package>
    </packages></coverage>"""
    root = ET.fromstring(xml_content)

    caplog.set_level("INFO", logger="tests.check_coverage")
    with mock.patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = mock.MagicMock()
        mock_tree.getroot.return_value = root
        mock_parse.return_value = mock_tree
        check_coverage.check_coverage("coverage.xml", threshold=0.9, use_color=False)
        mock_parse.assert_called_once_with("coverage.xml")

    out = caplog.text
    assert "modules/mod1.py" in out
    assert "PASS" in out
    assert "SUCCESS" in out


def test_check_coverage_failure_exits(caplog):
    """Check utility exits with code 1 and failure summary when a file is below threshold."""
    xml_content = """<coverage line-rate="0.50"><packages>
        <package name="modules"><classes>
            <class filename="modules/mod1.py" line-rate="0.50" name="mod1.py"></class>
        </classes></package>
    </packages></coverage>"""
    root = ET.fromstring(xml_content)

    caplog.set_level("INFO", logger="tests.check_coverage")
    with mock.patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = mock.MagicMock()
        mock_tree.getroot.return_value = root
        mock_parse.return_value = mock_tree

        with pytest.raises(SystemExit) as exc:
            check_coverage.check_coverage("coverage.xml", threshold=0.9, use_color=False)
        mock_parse.assert_called_once_with("coverage.xml")

    assert exc.value.code == 1
    out = caplog.text
    assert "FAIL" in out
    assert "CRITICAL" in out
