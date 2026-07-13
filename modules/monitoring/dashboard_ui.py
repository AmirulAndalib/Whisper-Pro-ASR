"""
Dashboard UI Components
"""

import os

from modules.monitoring import template_loader


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    base_dir = os.path.dirname(__file__)

    html, css = template_loader.load_page_html_and_css(base_dir, "dashboard.html", "dashboard.css")
    combined_js = template_loader.load_javascript_bundle(base_dir, "dashboard_js_files.txt")

    return html.replace("/* {{DASHBOARD_CSS}} */", css).replace("// {{DASHBOARD_JS}}", combined_js)
