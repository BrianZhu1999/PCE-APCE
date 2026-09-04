from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

import assemble_figure4_l96_kse_kol_v18_vorticity_left as kol_assembly
import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent

SVG_IN = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl.svg"
PNG_IN = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl.png"

SVG_OUT = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08.svg"
PNG_OUT = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08.png"

MASTER_OUT = HERE / "figure4_reordered_l96_kse_v17_title20_kse_titles4p08.png"
MASTER_PDF = HERE / "figure4_reordered_l96_kse_v17_title20_kse_titles4p08.pdf"
MASTER_TIFF = HERE / "figure4_reordered_l96_kse_v17_title20_kse_titles4p08.tiff"

FINAL_BASE = HERE / "figure4_reordered_l96_kse_kol_v33_kse_titles4p08"

SHIFT_PT = 6.01
SVG_WIDTH_PT = 1277.148
LABEL_SAFE_START_PT = 21.0


def patch_svg() -> None:
    src = SVG_IN.read_text(encoding="utf-8")
    patched = re.sub(
        r'<g id="axes_(\d+)">',
        lambda match: f'<g id="axes_{match.group(1)}" transform="translate(0 -{SHIFT_PT})">',
        src,
    )
    SVG_OUT.write_text(patched, encoding="utf-8")


def patch_png() -> dict[str, object]:
    image = Image.open(PNG_IN).convert("RGBA")
    scale = image.width / SVG_WIDTH_PT
    shift_px = round(SHIFT_PT * scale)
    start_px = round(LABEL_SAFE_START_PT * scale)
    row_h = image.height // 2

    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    for row in (0, 1):
        y0 = row * row_h
        y1 = y0 + row_h

        # Keep panel labels g--l fixed.
        canvas.alpha_composite(image.crop((0, y0, image.width, y0 + start_px)), (0, y0))

        # Move titles and plot bodies upward by 6.01 pt.
        src_top = y0 + start_px
        dst_top = y0 + start_px - shift_px
        if dst_top < y0:
            src_top += y0 - dst_top
            dst_top = y0
        canvas.alpha_composite(image.crop((0, src_top, image.width, y1)), (0, dst_top))

    if image.height > 2 * row_h:
        canvas.alpha_composite(image.crop((0, 2 * row_h, image.width, image.height)), (0, 2 * row_h))

    canvas.convert("RGB").save(PNG_OUT, quality=95)
    return {
        "shift_pt": SHIFT_PT,
        "scale_px_per_pt": scale,
        "shift_px": shift_px,
        "label_safe_start_px": start_px,
        "row_h_px": row_h,
        "png_in": str(PNG_IN),
        "png_out": str(PNG_OUT),
        "png_size_px": list(image.size),
    }


def assemble_l96_kse_master() -> None:
    original_kse = base.KSE_PATH
    original_outs = (base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF)
    try:
        base.KSE_PATH = PNG_OUT
        base.OUT_PNG = MASTER_OUT
        base.OUT_PDF = MASTER_PDF
        base.OUT_TIFF = MASTER_TIFF
        base.main()
    finally:
        base.KSE_PATH = original_kse
        base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF = original_outs


def assemble_final() -> None:
    old_input = kol_assembly.INPUT_PNG
    old_panel = kol_assembly.KOL_PANEL
    old_output = kol_assembly.OUTPUT_BASE
    old_x = kol_assembly.KOL_PANEL_X
    try:
        kol_assembly.INPUT_PNG = MASTER_OUT
        kol_assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v12_titlegap_up_tlabels.png"
        kol_assembly.OUTPUT_BASE = FINAL_BASE
        kol_assembly.KOL_PANEL_X = 3
        kol_assembly.main()
    finally:
        kol_assembly.INPUT_PNG = old_input
        kol_assembly.KOL_PANEL = old_panel
        kol_assembly.OUTPUT_BASE = old_output
        kol_assembly.KOL_PANEL_X = old_x


def verify_svg_baselines() -> dict[str, object]:
    patched = SVG_OUT.read_text(encoding="utf-8")
    labels = {}
    for match in re.finditer(r'<text[^>]* y="([^"]+)"[^>]*>([ghijkl])</text>', patched):
        labels[match.group(2)] = float(match.group(1))

    # The moved title baselines equal the original explicit y minus SHIFT_PT.
    title_top = 28.0 - SHIFT_PT
    title_bottom = 262.0 - SHIFT_PT
    return {
        "g_h_i_label_y_pt": {k: labels[k] for k in ("g", "h", "i")},
        "j_k_l_label_y_pt": {k: labels[k] for k in ("j", "k", "l")},
        "top_title_y_after_transform_pt": title_top,
        "bottom_title_y_after_transform_pt": title_bottom,
        "top_label_to_title_pt": title_top - labels["g"],
        "bottom_label_to_title_pt": title_bottom - labels["j"],
    }


def main() -> None:
    patch_svg()
    raster_qa = patch_png()
    assemble_l96_kse_master()
    assemble_final()
    qa = {
        **raster_qa,
        **verify_svg_baselines(),
        "master_png": str(MASTER_OUT),
        "master_pdf": str(MASTER_PDF),
        "master_tiff": str(MASTER_TIFF),
        "final_png": str(FINAL_BASE.with_suffix(".png")),
        "final_pdf": str(FINAL_BASE.with_suffix(".pdf")),
        "final_tiff": str(FINAL_BASE.with_suffix(".tiff")),
    }
    qa_path = FINAL_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
