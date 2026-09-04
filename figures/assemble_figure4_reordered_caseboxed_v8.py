from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl.png"
L96_PATH = HERE / "figure4_l96_row_v4_trace_phase.png"
OUT_PNG = HERE / "figure4_reordered_l96_kse_v16_title20_moregap_93.png"
OUT_PDF = HERE / "figure4_reordered_l96_kse_v16_title20_moregap_93.pdf"
OUT_TIFF = HERE / "figure4_reordered_l96_kse_v16_title20_moregap_93.tiff"

CANVAS_W = 11532
TOP_MARGIN = 26
TITLE_BAND = 306
ROW_GAP = 30
KSE_INNER_GAP = 24
BOTTOM_MARGIN = 30
TITLE_LEFT = 62
L96_CROP_TOP = 282

# 18-pt bold title, scaled to the accepted high-DPI raster.
TITLE_PX = 169


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


TITLE_FONT = font("arialbd.ttf", TITLE_PX)


def resize_to_width(im: Image.Image, width: int = CANVAS_W) -> Image.Image:
    if im.width == width:
        return im.convert("RGBA")
    height = round(im.height * width / im.width)
    return im.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)


def title(draw: ImageDraw.ImageDraw, y: int, text: str) -> None:
    draw.text((TITLE_LEFT, y), text, font=TITLE_FONT, fill=(25, 25, 25, 255))


def main() -> None:
    l96_full = resize_to_width(Image.open(L96_PATH))
    l96 = l96_full.crop((0, L96_CROP_TOP, l96_full.width, l96_full.height))
    kse = resize_to_width(Image.open(KSE_PATH))
    kse_row_h = kse.height // 2
    kse_recon = kse.crop((0, 0, kse.width, kse_row_h))
    kse_fore = kse.crop((0, kse_row_h, kse.width, kse.height))

    total_h = (
        TOP_MARGIN
        + TITLE_BAND
        + l96.height
        + ROW_GAP
        + TITLE_BAND
        + kse_recon.height
        + KSE_INNER_GAP
        + kse_fore.height
        + BOTTOM_MARGIN
    )

    canvas = Image.new("RGBA", (CANVAS_W, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = TOP_MARGIN
    title(draw, y, "Lorenz–96 system")
    y += TITLE_BAND
    canvas.alpha_composite(l96, (0, y))
    y += l96.height + ROW_GAP

    title(draw, y, "Kuramoto–Sivashinsky system")
    y += TITLE_BAND
    canvas.alpha_composite(kse_recon, (0, y))
    y += kse_recon.height + KSE_INNER_GAP
    canvas.alpha_composite(kse_fore, (0, y))

    rgb = canvas.convert("RGB")
    rgb.save(OUT_PNG, quality=95)
    rgb.save(OUT_PDF, resolution=600)
    rgb.save(OUT_TIFF, compression="tiff_deflate")
    print(OUT_PNG.resolve())
    print(OUT_PDF.resolve())
    print(OUT_TIFF.resolve())
    print(rgb.size)


if __name__ == "__main__":
    main()
