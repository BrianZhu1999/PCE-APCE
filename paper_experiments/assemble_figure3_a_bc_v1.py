from __future__ import annotations

from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


FIG_DIR = Path(r"figures")
A_PATH = FIG_DIR / "figure3a_selected5_ode_panels_row_no_sir_v19_uniformcrop.png"
BC_PATH = FIG_DIR / "figure3bc_delta_lollipop_v15_zero_aligned.png"
OUT_STEM = FIG_DIR / "figure3_assembled_v3_a_bc_tightest_gap"


def main() -> None:
    a = Image.open(A_PATH).convert("RGB")
    bc = Image.open(BC_PATH).convert("RGB")

    target_w = max(a.width, bc.width)
    if a.width != target_w:
        new_h = round(a.height * target_w / a.width)
        a = a.resize((target_w, new_h), Image.Resampling.LANCZOS)
    if bc.width != target_w:
        new_h = round(bc.height * target_w / bc.width)
        bc = bc.resize((target_w, new_h), Image.Resampling.LANCZOS)

    # Tight but breathable row gap; white background keeps b/c panels distinct from
    # the beige hero strip without adding a visible frame.
    gap = 12
    canvas = Image.new("RGB", (target_w, a.height + gap + bc.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(bc, (0, a.height + gap))

    png_path = OUT_STEM.with_suffix(".png")
    tiff_path = OUT_STEM.with_suffix(".tiff")
    pdf_path = OUT_STEM.with_suffix(".pdf")

    canvas.save(png_path, dpi=(600, 600))
    canvas.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")

    fig_w = target_w / 600.0
    fig_h = canvas.height / 600.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(pdf_path, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
