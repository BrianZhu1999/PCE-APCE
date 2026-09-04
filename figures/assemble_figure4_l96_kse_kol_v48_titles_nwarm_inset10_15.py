from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent

L96_PATH = HERE / "figure4_l96_row_v6_full_titles.png"
KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08_chrome.png"
KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v16_ref_label.png"
KOL_PANEL_N = HERE / "figure4_kol_quant_panel_n_v5_inset10_15_warm.png"

OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v48_titles_nwarm_inset10_15"

CANVAS_W = base.CANVAS_W
TOP_MARGIN = base.TOP_MARGIN
ROW_GAP = 0
KSE_INNER_GAP = -40
BOTTOM_MARGIN = base.BOTTOM_MARGIN
KOL_PANEL_X = 3

# Chosen from v38 measured title-bottom -> next panel-label-top gaps:
# Lorenz 169 px, KSE 173 px, KOL 187 px.  Move the content blocks upward by
# 69/73/87 px respectively, targeting 100 px for all three case headers.
L96_TITLE_BAND = base.TITLE_BAND - 69
KSE_TITLE_BAND = base.TITLE_BAND - 73
KOL_TITLE_BAND = base.TITLE_BAND - 87


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_to_width(im: Image.Image, width: int = CANVAS_W) -> Image.Image:
    if im.width == width:
        return im.convert("RGBA")
    height = round(im.height * width / im.width)
    return im.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    l96_full = resize_to_width(Image.open(L96_PATH))
    l96 = l96_full.crop((0, base.L96_CROP_TOP, l96_full.width, l96_full.height))

    kse = resize_to_width(Image.open(KSE_PATH))
    kse_row_h = kse.height // 2
    kse_recon = kse.crop((0, 0, kse.width, kse_row_h))
    kse_fore = kse.crop((0, kse_row_h, kse.width, kse.height))

    kol = Image.open(KOL_PANEL).convert("RGBA")
    if KOL_PANEL_X + kol.width > CANVAS_W:
        raise ValueError(f"KOL panel right edge {KOL_PANEL_X + kol.width} exceeds canvas width {CANVAS_W}")
    kol_n = Image.open(KOL_PANEL_N).convert("RGBA")
    if kol_n.width != CANVAS_W:
        raise ValueError(f"KOL panel n width {kol_n.width} differs from canvas width {CANVAS_W}")

    total_h = (
        TOP_MARGIN
        + L96_TITLE_BAND
        + l96.height
        + ROW_GAP
        + KSE_TITLE_BAND
        + kse_recon.height
        + KSE_INNER_GAP
        + kse_fore.height
        + ROW_GAP
        + KOL_TITLE_BAND
        + kol.height
        + BOTTOM_MARGIN
    )
    canvas = Image.new("RGBA", (CANVAS_W, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = TOP_MARGIN
    base.title(draw, y, "Lorenz–96 system")
    y += L96_TITLE_BAND
    canvas.alpha_composite(l96, (0, y))
    y += l96.height + ROW_GAP

    base.title(draw, y, "Kuramoto–Sivashinsky system")
    y += KSE_TITLE_BAND
    canvas.alpha_composite(kse_recon, (0, y))
    y += kse_recon.height + KSE_INNER_GAP
    canvas.alpha_composite(kse_fore, (0, y))
    y += kse_fore.height + ROW_GAP

    base.title(draw, y, "Kolmogorov turbulent flow system")
    y += KOL_TITLE_BAND
    canvas.alpha_composite(kol, (KOL_PANEL_X, y))
    canvas.alpha_composite(kol_n, (0, y))

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
        "input_l96": str(L96_PATH),
        "input_l96_sha256": sha256(L96_PATH),
        "input_kse": str(KSE_PATH),
        "input_kse_sha256": sha256(KSE_PATH),
        "input_kol": str(KOL_PANEL),
        "input_kol_sha256": sha256(KOL_PANEL),
        "input_kol_n": str(KOL_PANEL_N),
        "input_kol_n_sha256": sha256(KOL_PANEL_N),
        "output_size_px": list(rgb.size),
        "target_title_to_panel_label_gap_px": 100,
        "title_bands_px": {
            "lorenz96": L96_TITLE_BAND,
            "kse": KSE_TITLE_BAND,
            "kol": KOL_TITLE_BAND,
        },
        "kse_inner_gap_px": KSE_INNER_GAP,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
