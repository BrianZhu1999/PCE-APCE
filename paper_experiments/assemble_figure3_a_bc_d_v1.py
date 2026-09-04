from __future__ import annotations

from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


FIG_DIR = Path(r"figures")
PANELS = [
    FIG_DIR / "figure3a_selected5_ode_panels_row_no_sir_v21_whitebg_fontrule16.png",
    FIG_DIR / "figure3bc_delta_lollipop_v19_height130.png",
    FIG_DIR / "figure3d_radar_summary_v10_labelup.png",
]
OUT_STEM = FIG_DIR / "figure3_assembled_v15_bc_uncropped_gap80"


def fit_width(im: Image.Image, target_w: int) -> Image.Image:
    if im.width == target_w:
        return im
    new_h = round(im.height * target_w / im.width)
    return im.resize((target_w, new_h), Image.Resampling.LANCZOS)


def crop_bottom(im: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return im
    return im.crop((0, 0, im.width, max(1, im.height - px)))


def crop_top(im: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return im
    return im.crop((0, min(px, im.height - 1), im.width, im.height))


def main() -> None:
    imgs = [Image.open(p).convert("RGB") for p in PANELS]
    target_w = max(im.width for im in imgs)
    imgs = [fit_width(im, target_w) for im in imgs]
    # Preserve the complete b/c row, including x-axis tick labels and titles.
    # Cropping this row created the false appearance that panel d touched it.
    # Keep d's own top margin intact; the requested separation is created by
    # an explicit inter-row white gap rather than by cropping the radar panel.
    gaps = [6, 80]
    total_h = sum(im.height for im in imgs) + sum(gaps)
    canvas = Image.new("RGB", (target_w, total_h), "white")
    y = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (0, y))
        y += im.height
        if i < len(gaps):
            y += gaps[i]

    png_path = OUT_STEM.with_suffix(".png")
    tiff_path = OUT_STEM.with_suffix(".tiff")
    pdf_path = OUT_STEM.with_suffix(".pdf")
    canvas.save(png_path, dpi=(600, 600))
    canvas.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")

    fig = plt.figure(figsize=(canvas.width / 600, canvas.height / 600), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(pdf_path, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
