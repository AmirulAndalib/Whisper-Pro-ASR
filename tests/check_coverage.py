"""Enforce per-file coverage threshold from coverage.xml."""

import logging
import sys
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _format_status(line_rate: float, threshold: float, colors: dict[str, str]) -> tuple[str, bool]:
    """Return the status label and whether the file failed coverage."""
    if line_rate < threshold:
        return f"{colors['red']}FAIL{colors['reset']}", True
    return f"{colors['green']}PASS{colors['reset']}", False


def _iter_coverage_rows(root: ET.Element):
    """Yield coverage rows from the XML tree."""
    for package in root.findall(".//package"):
        for cls in package.findall(".//class"):
            yield cls.get("filename"), float(cls.get("line-rate"))


def _print_coverage_rows(root: ET.Element, threshold: float, colors: dict[str, str]) -> tuple[bool, int]:
    """Print coverage rows and return whether any file failed."""
    failed = False
    row_count = 0
    for filename, line_rate in _iter_coverage_rows(root):
        row_count += 1
        pct = line_rate * 100
        status, row_failed = _format_status(line_rate, threshold, colors)
        failed = failed or row_failed
        logger.info("%s | %8.2f%% | %s", f"{filename:<40}", pct, status)
    return failed, row_count


def _report_result(failed: bool, threshold: float, colors: dict[str, str], row_count: int) -> None:
    """Print the final summary and exit on failure."""
    if row_count == 0:
        logger.error(
            "%s%sCRITICAL: No file-level coverage rows were found in coverage.xml.%s", colors["bold"], colors["red"], colors["reset"]
        )
        sys.exit(1)

    if failed:
        msg = f"CRITICAL: One or more files are below the {threshold * 100:.0f}% threshold!"
        logger.error("%s%s%s%s", colors["bold"], colors["red"], msg, colors["reset"])
        sys.exit(1)

    msg = f"SUCCESS: All files meet or exceed the {threshold * 100:.0f}% threshold."
    logger.info("%s%s%s%s", colors["bold"], colors["green"], msg, colors["reset"])


def check_coverage(xml_file: str, threshold: float = 0.9, use_color: bool = True) -> None:
    """Check if all files in the coverage.xml meet the threshold."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    logger.info("\n%s", "=" * 60)
    logger.info("%s | %s | %s", f"{'FILE':<40}", f"{'COVERAGE':<10}", f"{'STATUS':<6}")
    logger.info("%s", "-" * 60)

    # Style
    clr = {
        "green": "\033[92m" if use_color else "",
        "red": "\033[91m" if use_color else "",
        "reset": "\033[0m" if use_color else "",
        "bold": "\033[1m" if use_color else "",
    }

    failed, row_count = _print_coverage_rows(root, threshold, clr)

    logger.info("%s", "-" * 60)
    _report_result(failed, threshold, clr, row_count)
    logger.info("%s\n", "=" * 60)


if __name__ == "__main__":
    # Check if --no-color is passed
    no_color = "--no-color" in sys.argv
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_coverage("coverage.xml", use_color=not no_color)
