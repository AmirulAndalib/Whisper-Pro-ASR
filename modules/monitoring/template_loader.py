"""Shared template loading helpers for monitoring UI pages."""

import os


def read_text_file(file_path: str) -> str:
    """Read and return UTF-8 text from a file path."""
    with open(file_path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def read_manifest_lines(file_path: str) -> list[str]:
    """Read newline-delimited manifest entries, ignoring blanks and comments."""
    return [line.strip() for line in read_text_file(file_path).splitlines() if line.strip() and not line.lstrip().startswith("#")]


def load_page_html_and_css(base_dir: str, template_name: str, css_name: str) -> tuple[str, str]:
    """Load page HTML and merged shared+page CSS from templates directory."""
    templates_dir = os.path.join(base_dir, "templates")

    template_path = os.path.join(templates_dir, template_name)
    shared_css_path = os.path.join(templates_dir, "responsive_shared.css")
    css_path = os.path.join(templates_dir, css_name)

    html = read_text_file(template_path)
    shared_css = read_text_file(shared_css_path)
    css = read_text_file(css_path)

    return html, f"{shared_css}\n\n{css}"


def load_javascript_bundle(base_dir: str, manifest_name: str) -> str:
    """Load and concatenate JavaScript assets listed in a templates manifest."""
    templates_dir = os.path.join(base_dir, "templates")
    manifest_path = os.path.join(templates_dir, manifest_name)
    js_files = read_manifest_lines(manifest_path)

    return "\n\n".join(read_text_file(os.path.join(templates_dir, js_file)) for js_file in js_files)
