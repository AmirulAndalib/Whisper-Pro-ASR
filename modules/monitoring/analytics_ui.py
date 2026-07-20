"""
HTML template for the Whisper Pro Analytics Dashboard.
"""

import os

from modules.monitoring import template_loader


def get_analytics_html() -> str:
    """Returns the rendered HTML for the analytics page."""
    base_dir = os.path.dirname(__file__)
    html, css = template_loader.load_page_html_and_css(base_dir, "analytics.html", "analytics.css")
    js = template_loader.load_javascript_bundle(base_dir, "analytics_js_files.txt")

    return html.replace("/* {{ANALYTICS_CSS}} */", css).replace("// {{ANALYTICS_JS}}", js)
