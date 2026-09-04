"""Compose the approved Figure 5 b-e, f-g and h-j rows without rescaling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


HERE = Path(__file__).resolve().parent
BCDE = HERE / "panels_bcde" / "outputs_gradient_glow" / "figure5_panels_bcde_x40y20_0803_best_mse_gradient_glow.png"
FG = HERE / "fg_row" / "outputs" / "figure5_fg_row_10553.png"
HIJ = HERE / "hij_row" / "outputs" / "figure5_hij_row_10553.png"
OUT = HERE / "combined_bcde_fg" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FG_ROW_GAP_PX = 220
HIJ_ROW_GAP_PX = 120


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcde", type=Path, default=BCDE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stem", default="figure5_bcde_fg_combined")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bcde = Image.open(args.bcde).convert("RGBA")
    fg = Image.open(FG).convert("RGBA")
    hij = Image.open(HIJ).convert("RGBA")
    if len({bcde.width, fg.width, hij.width}) != 1:
        raise ValueError(f"Width mismatch: bcde={bcde.width}, fg={fg.width}, hij={hij.width}")

    width = bcde.width
    height = bcde.height + FG_ROW_GAP_PX + fg.height + HIJ_ROW_GAP_PX + hij.height
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    canvas.alpha_composite(bcde, (0, 0))
    canvas.alpha_composite(fg, (0, bcde.height + FG_ROW_GAP_PX))
    canvas.alpha_composite(hij, (0, bcde.height + FG_ROW_GAP_PX + fg.height + HIJ_ROW_GAP_PX))
    png = args.output / f"{args.stem}.png"
    tiff = args.output / f"{args.stem}.tiff"
    canvas.save(png, dpi=(650, 650))
    canvas.convert("RGB").save(tiff, compression="tiff_lzw", dpi=(650, 650))

    # Vector containers preserve the source raster panels without changing pixels.
    fig = plt.figure(figsize=(width / 650, height / 650), dpi=650, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_axis_off()
    fig.savefig(args.output / f"{args.stem}.pdf", dpi=650, facecolor="white", pad_inches=0)
    fig.savefig(args.output / f"{args.stem}.svg", dpi=650, facecolor="white", pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": args.stem,
        "composition": "vertical concatenation; bcde, fg and hij rows from approved source assets with explicit inter-row gaps; no rescaling or cropping",
        "canvas": {"width_px": width, "height_px": height, "dpi": 650, "fg_row_gap_px": FG_ROW_GAP_PX, "hij_row_gap_px": HIJ_ROW_GAP_PX},
        "panels": {
            "bcde": {"path": str(args.bcde), "width_px": bcde.width, "height_px": bcde.height},
            "fg": {"path": str(FG), "width_px": fg.width, "height_px": fg.height},
            "hij": {"path": str(HIJ), "width_px": hij.width, "height_px": hij.height},
        },
        "outputs": {ext: str(args.output / f"{args.stem}{ext}") for ext in [".png", ".tiff", ".pdf", ".svg"]},
    }
    metadata["output_sha256"] = {
        ext: hashlib.sha256((args.output / f"{args.stem}{ext}").read_bytes()).hexdigest()
        for ext in [".png", ".tiff", ".pdf", ".svg"]
    }
    (args.output / f"{args.stem}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
