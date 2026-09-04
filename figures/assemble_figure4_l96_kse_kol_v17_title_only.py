from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent
INPUT_PNG = HERE / "figure4_reordered_l96_kse_v16_title20_moregap_93.png"
OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v17_title_only"
TITLE_TEXT = "Kolmogorov flow system"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    existing = Image.open(INPUT_PNG).convert("RGBA")
    if existing.width != base.CANVAS_W:
        raise ValueError(f"Expected width {base.CANVAS_W}, got {existing.width}")

    # The accepted master already contains BOTTOM_MARGIN pixels below row 3.
    # Add only the remaining gap so the new case title follows the same spacing rule.
    title_y = existing.height + max(0, base.ROW_GAP - base.BOTTOM_MARGIN)
    total_h = title_y + base.TITLE_BAND + base.BOTTOM_MARGIN
    canvas = Image.new("RGBA", (base.CANVAS_W, total_h), (255, 255, 255, 255))
    canvas.alpha_composite(existing, (0, 0))

    draw = ImageDraw.Draw(canvas)
    base.title(draw, title_y, TITLE_TEXT)

    rgb = canvas.convert("RGB")
    outputs = {
        "png": OUTPUT_BASE.with_suffix(".png"),
        "pdf": OUTPUT_BASE.with_suffix(".pdf"),
        "tiff": OUTPUT_BASE.with_suffix(".tiff"),
    }
    rgb.save(outputs["png"], quality=95)
    rgb.save(outputs["pdf"], resolution=600)
    rgb.save(outputs["tiff"], compression="tiff_deflate")

    qa = {
        "backend": "Python/Pillow",
        "purpose": "Figure 4 master with a reserved fourth-row Kolmogorov-flow section",
        "input": str(INPUT_PNG),
        "input_sha256": sha256(INPUT_PNG),
        "input_size_px": list(existing.size),
        "output_size_px": list(rgb.size),
        "title": TITLE_TEXT,
        "title_x_px": base.TITLE_LEFT,
        "title_y_px": title_y,
        "title_font_px": base.TITLE_PX,
        "title_band_px": base.TITLE_BAND,
        "row_gap_px": base.ROW_GAP,
        "existing_bottom_margin_px": base.BOTTOM_MARGIN,
        "existing_master_pixels_preserved": True,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
