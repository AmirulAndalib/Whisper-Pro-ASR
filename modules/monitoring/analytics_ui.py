"""
HTML template for the Whisper Pro Analytics Dashboard.
"""

import os

from modules.monitoring import template_loader


def get_analytics_html() -> str:
    """Returns the rendered HTML for the analytics page."""
    base_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(base_dir, "templates")

    html, css = template_loader.load_page_html_and_css(base_dir, "analytics.html", "analytics.css")
    js_path = os.path.join(templates_dir, "analytics.js")
    js = template_loader.read_text_file(js_path)

    return html.replace("/* {{ANALYTICS_CSS}} */", css).replace("// {{ANALYTICS_JS}}", js)
