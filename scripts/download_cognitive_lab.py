"""Download and stage the public Cognitive Lab Figshare project.

The script intentionally downloads only metadata-listed files and never invents labels.
It searches the public project for cognitive-state/HCI files and stores the raw files
under data/cognitive_lab/. The exact final CSV is created by prepare_cognitive_lab.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

BASE = "https://api.figshare.com/v2"
PROJECT_ID = 233810
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "cognitive_lab"


def get(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    articles = get(f"{BASE}/projects/{PROJECT_ID}/articles?page=1&page_size=100")
    manifest = []
    for article in articles:
        aid = article["id"]
        detail = get(f"{BASE}/articles/{aid}")
        folder = RAW / str(aid)
        folder.mkdir(exist_ok=True)
        for f in detail.get("files", []):
            name = f["name"]
            target = folder / name
            manifest.append({"article_id": aid, "name": name, "url": f.get("download_url"), "size": f.get("size")})
            if target.exists() and target.stat().st_size == f.get("size", -1):
                continue
            url = f.get("download_url")
            if not url or f.get("is_link_only"):
                continue
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with target.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged {len(manifest)} files under {RAW}")


if __name__ == "__main__":
    main()
