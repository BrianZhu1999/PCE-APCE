#!/usr/bin/env python3
"""Inventory Baoding documents and visual assets on the authoritative server."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DOC_SUFFIXES = {".docx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_info(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            return {"width_px": image.width, "height_px": image.height, "mode": image.mode, "format": image.format}
    except Exception as exc:
        return {"read_error": str(exc)}


def docx_info(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            media = [name for name in archive.namelist() if name.startswith("word/media/") and not name.endswith("/")]
            return {"embedded_media_count": len(media), "embedded_media": media}
    except Exception as exc:
        return {"read_error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.root:
        if not root.exists():
            rows.append({"root": str(root), "path": str(root), "kind": "missing"})
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES | DOC_SUFFIXES:
                continue
            row = {"root": str(root), "path": str(path), "relative_path": str(path.relative_to(root)), "kind": "image" if suffix in IMAGE_SUFFIXES else "docx", "suffix": suffix, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            row.update(image_info(path) if suffix in IMAGE_SUFFIXES else docx_info(path))
            rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "baoding_visual_asset_inventory.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output / "baoding_visual_asset_inventory.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "embedded_media": json.dumps(row.get("embedded_media", []), ensure_ascii=False)})
    print(json.dumps({"assets": len(rows), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
