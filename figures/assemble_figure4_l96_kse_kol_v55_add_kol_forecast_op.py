from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
INPUT_FIGURE = HERE / "figure4_reordered_l96_kse_kol_v54_attachment_palettes_alpha093.png"
KOL_FORECAST_ROW = HERE / "figure4_kol_forecast_panels_op_v1.png"
OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v55_add_kol_forecast_op"

FORECAST_ROW_GAP = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    top = Image.open(INPUT_FIGURE).convert("RGBA")
    forecast = Image.open(KOL_FORECAST_ROW).convert("RGBA")
    if forecast.width != top.width:
        raise ValueError(f"Forecast row width {forecast.width} differs from base figure width {top.width}")
    total_h = top.height + FORECAST_ROW_GAP + forecast.height
    canvas = Image.new("RGBA", (top.width, total_h), (255, 255, 255, 255))
    canvas.alpha_composite(top, (0, 0))
    canvas.alpha_composite(forecast, (0, top.height + FORECAST_ROW_GAP))
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
        "input_figure": str(INPUT_FIGURE),
        "input_figure_sha256": sha256(INPUT_FIGURE),
        "kol_forecast_row": str(KOL_FORECAST_ROW),
        "kol_forecast_row_sha256": sha256(KOL_FORECAST_ROW),
        "forecast_row_gap_px": FORECAST_ROW_GAP,
        "output_size_px": list(rgb.size),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
