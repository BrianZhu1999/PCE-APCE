from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent
INPUT_PNG = HERE / "figure4_reordered_l96_kse_v16_title20_moregap_93.png"
KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_2x4_v1.png"
OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v18_vorticity_left"
TITLE_TEXT = "Kolmogorov turbulent flow system"
KOL_PANEL_X = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    existing = Image.open(INPUT_PNG).convert("RGBA")
    panel = Image.open(KOL_PANEL).convert("RGBA")
    if existing.width != base.CANVAS_W:
        raise ValueError(f"Expected width {base.CANVAS_W}, got {existing.width}")
    if panel.width > base.CANVAS_W // 2:
        raise ValueError(f"KOL panel width {panel.width} exceeds half-width {base.CANVAS_W // 2}")

    title_y = existing.height + max(0, base.ROW_GAP - base.BOTTOM_MARGIN)
    panel_y = title_y + base.TITLE_BAND
    total_h = panel_y + panel.height + base.BOTTOM_MARGIN
    canvas = Image.new("RGBA", (base.CANVAS_W, total_h), (255, 255, 255, 255))
    canvas.alpha_composite(existing, (0, 0))

    draw = ImageDraw.Draw(canvas)
    base.title(draw, title_y, TITLE_TEXT)
    canvas.alpha_composite(panel, (KOL_PANEL_X, panel_y))

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
        "input_master": str(INPUT_PNG),
        "input_master_sha256": sha256(INPUT_PNG),
        "kol_panel": str(KOL_PANEL),
        "kol_panel_sha256": sha256(KOL_PANEL),
        "input_size_px": list(existing.size),
        "kol_panel_size_px": list(panel.size),
        "output_size_px": list(rgb.size),
        "title": TITLE_TEXT,
        "title_x_px": base.TITLE_LEFT,
        "title_y_px": title_y,
        "panel_x_px": KOL_PANEL_X,
        "panel_y_px": panel_y,
        "left_half_width_px": base.CANVAS_W // 2,
        "existing_master_pixels_preserved": True,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
