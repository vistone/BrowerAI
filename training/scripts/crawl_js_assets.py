#!/usr/bin/env python3
"""
Fetch JS assets from target websites and store plain JS for obfuscation pairing.
- Input: list of URLs (default includes a few large sites; replace as needed).
- Output: writes JS files into training/data/raw_js/<host>/script_#.js
- Notes: Respect robots.txt and only fetch publicly accessible URLs. Use responsibly.
"""

import os
import re
import sys
import time
import json
import urllib.parse
from pathlib import Path
from typing import List

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_js"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SITES = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.youtube.com",
]

HEADERS = {
    "User-Agent": "BrowerAI-Collector/1.0"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def fetch_text(url: str) -> str:
    resp = SESSION.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def resolve_url(base: str, link: str) -> str:
    return urllib.parse.urljoin(base, link)


def extract_js_links(html: str) -> List[str]:
    links = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    return links


def save_js(content: str, host: str, idx: int) -> Path:
    out_dir = RAW_DIR / host
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"script_{idx}.js"
    path.write_text(content, encoding="utf-8", errors="ignore")
    return path


def fetch_site(site: str, max_scripts: int = 5) -> List[Path]:
    saved = []
    try:
        html = fetch_text(site)
    except Exception as e:
        print(f"[warn] fetch page failed {site}: {e}")
        return saved

    links = extract_js_links(html)
    if not links:
        print(f"[info] no script tags found for {site}")
        return saved

    host = urllib.parse.urlparse(site).netloc or "unknown"
    for i, link in enumerate(links[:max_scripts], 1):
        js_url = resolve_url(site, link)
        try:
            js_text = fetch_text(js_url)
            path = save_js(js_text, host, i)
            saved.append(path)
            print(f"[ok] {host} #{i}: {js_url} -> {path.relative_to(ROOT)}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[warn] skip {js_url}: {e}")
    return saved


def main():
    sites = DEFAULT_SITES
    if len(sys.argv) > 1:
        sites = sys.argv[1:]
    print(f"Targets: {sites}")

    all_saved = []
    for site in sites:
        all_saved.extend(fetch_site(site))

    if not all_saved:
        print("❌ No JS fetched. Provide reachable URLs or increase limits.")
        return 1

    print(f"✅ Saved {len(all_saved)} JS files under {RAW_DIR}")
    print("下一步: python scripts/generate_obfuscation_pairs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
