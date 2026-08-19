"""Fetch the public SWELL-KW minute-level behavioural feature table.

Source: DANS Data Station, DOI 10.17026/DANS-X55-69ZP.
The public dataset contains 25 participants and minute-level multimodal/behavioural
features. This setup script deliberately uses the 4 MB behavioural table rather than
the 7 GB raw archive.
"""
from __future__ import annotations

from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "swell_kw_behavioral.tab"
API = "https://ssh.datastations.nl/api"
DOI = "doi:10.17026/DANS-X55-69ZP"


def main():
    ROOT.joinpath("data").mkdir(exist_ok=True)
    meta_url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    r = requests.get(meta_url, timeout=60)
    r.raise_for_status()
    meta = r.json()["data"]

    candidates = []
    def walk(node):
        if isinstance(node, dict):
            name = str(node.get("label") or node.get("name") or "")
            if name.lower() == "behavioral-features - per minute.tab":
                candidates.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(meta)
    if not candidates:
        raise RuntimeError("Could not locate the public behavioural feature file in DANS metadata.")
    item = candidates[0]
    file_id = item.get("dataFile", {}).get("id") or item.get("dataFileId") or item.get("id")
    if not file_id:
        raise RuntimeError("DANS metadata did not expose a data-file id.")

    url = f"{API}/access/datafile/{file_id}?format=original"
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with OUT.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
