"""
Thin client onto the real, deployed pipeline -- the live HF Space, via
gradio_client, exactly the way a real user (or the 2026-09-09 session)
reaches it. No pipeline internals are called directly here, and no
parameters beyond what the real UI exposes (files, source_lang) are sent,
because the real API does not expose anything else -- see graph.py's
module docstring for what that constraint means for "retry with
adjusted parameters."
"""

import html
import re


def call_live_space(image_path, source_lang, space_name="NerdCoderZero/manga-translator"):
    from gradio_client import Client, handle_file

    client = Client(space_name)
    status, gallery, region_html, zip_info = client.predict(
        files=[handle_file(image_path)], source_lang=source_lang, api_name="/translate"
    )
    regions = _parse_region_table(region_html)
    return {"status": status, "regions": regions, "gallery": gallery, "raw_html": region_html}


_ROW_RE = re.compile(
    r"<tr><td>(\d+)</td><td>(.*?)</td><td>(.*?)</td><td><code>(.*?)</code></td></tr>"
)


def _parse_region_table(region_html):
    import json

    regions = []
    for page, source, translation, bounds in _ROW_RE.findall(region_html):
        regions.append({
            "page": int(page),
            "source_text": html.unescape(source),
            "translation": html.unescape(translation),
            "bounds": json.loads(bounds),
        })
    return regions
