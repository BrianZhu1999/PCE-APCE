"""Create a Figure 5 variant with a cool darkening gradient behind f--j.

The existing approved composites remain untouched.  This variant starts from
the plain-bcde candidate and replaces only registered near-background colours
in the fg and hij source rows; field heatmaps, curves, labels and colourbars
are preserved pixel-for-pixel outside that mask.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_BCDE = HERE / "panels_bcde" / "outputs_half_tint" / "figure5_panels_bcde_x40y20_0803_best_mse_gradient_glow.png"
FG = HERE / "fg_row" / "outputs" / "figure5_fg_row_10553.png"
HIJ = HERE / "hij_row" / "outputs" / "figure5_hij_row_10553.png"

FG_ROW_GAP_PX = 220
HIJ_ROW_GAP_PX = 120
DPI = 650
THEMES = {
    "cool-gray": {
        "gradient_top": [248, 250, 252],
        "gradient_bottom": [230, 236, 240],
        "description": "half-strength registered background-only cool gradient",
    },
    "coordinated-mist": {
        "gradient_top": [240, 246, 244],
        "gradient_bottom": [229, 239, 237],
        "description": "low-chroma blue-to-sage continuation of the coordinated b-e stage palette",
    },
    "soft-continuation": {
        "gradient_top": [248, 251, 250],
        "gradient_bottom": [239, 245, 244],
        "description": "very-light neutral continuation matched to the b-e lower-stage export, with only a gentle depth cue through f-j",
    },
    "pearl-blue": {
        "gradient_top": [250, 251, 253],
        "gradient_bottom": [237, 242, 248],
        "description": "green-free pearl white to restrained cool blue-gray continuation",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertical_gradient(
    height: int,
    start_y: int,
    end_y: int,
    width: int,
    gradient_top: np.ndarray,
    gradient_bottom: np.ndarray,
) -> np.ndarray:
    y = np.arange(height, dtype=np.float64)[:, None]
    denom = max(1.0, float(end_y - start_y - 1))
    t = np.clip((y - float(start_y)) / denom, 0.0, 1.0)
    rgb = (1.0 - t)[..., None] * gradient_top + t[..., None] * gradient_bottom
    return np.repeat(np.rint(rgb).astype(np.uint8), width, axis=1)


def replace_registered_backgrounds(
    image: Image.Image,
    background_colours: list[tuple[int, int, int]],
    gradient_rgb: np.ndarray,
    row_y0: int,
    tolerance: float = 2.5,
) -> tuple[Image.Image, int]:
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.float64)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for colour in background_colours:
        target = np.asarray(colour, dtype=np.float64)
        mask |= np.sqrt(np.sum((rgb - target) ** 2, axis=2)) <= tolerance
    if np.any(mask):
        local_gradient = gradient_rgb[row_y0:row_y0 + rgba.shape[0]]
        rgba[mask, :3] = local_gradient[mask, :3]
    return Image.fromarray(rgba, mode="RGBA"), int(mask.sum())


def replace_registered_gradient(
    image: Image.Image,
    old_gradient_rgb: np.ndarray,
    new_gradient_rgb: np.ndarray,
    row_y0: int,
    tolerance: float = 2.5,
) -> tuple[Image.Image, int]:
    """Map only pixels matching a registered old gradient to a new one."""
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.float64)
    old_local = old_gradient_rgb[row_y0:row_y0 + image.height, : image.width].astype(np.float64)
    new_local = new_gradient_rgb[row_y0:row_y0 + image.height, : image.width]
    mask = np.sqrt(np.sum((rgb - old_local) ** 2, axis=2)) <= tolerance
    rgba[mask, :3] = new_local[mask, :3]
    return Image.fromarray(rgba, mode="RGBA"), int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", choices=tuple(THEMES), default="cool-gray")
    parser.add_argument("--bcde", type=Path, default=DEFAULT_BCDE)
    parser.add_argument("--hij", type=Path, default=HIJ)
    parser.add_argument("--output", type=Path, default=HERE / "combined_bcde_fg_dark" / "outputs")
    parser.add_argument("--stem", default="figure5_bcde_fg_combined_dark_fghij")
    parser.add_argument(
        "--previous-bj",
        type=Path,
        default=None,
        help="Approved b-j source used to retain pixels outside revised panels.",
    )
    parser.add_argument(
        "--preserve-fg-left-px",
        type=int,
        default=0,
        help="When --previous-bj is given, retain this many left pixels of the f-g row.",
    )
    parser.add_argument(
        "--revise-hij-left-px",
        type=int,
        default=0,
        help="When --previous-bj is given, replace only this left width of the h-i-j row; zero replaces the full row.",
    )
    parser.add_argument(
        "--previous-theme",
        choices=tuple(THEMES),
        default=None,
        help="Registered theme of --previous-bj, used to recolor preserved background pixels only.",
    )
    args = parser.parse_args()
    theme = THEMES[args.theme]
    gradient_top = np.asarray(theme["gradient_top"], dtype=np.float64)
    gradient_bottom = np.asarray(theme["gradient_bottom"], dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=True)

    bcde = Image.open(args.bcde).convert("RGBA")
    fg = Image.open(FG).convert("RGBA")
    hij = Image.open(args.hij).convert("RGBA")
    if len({bcde.width, fg.width, hij.width}) != 1:
        raise ValueError(f"Width mismatch: bcde={bcde.width}, fg={fg.width}, hij={hij.width}")

    width = bcde.width
    fg_y0 = bcde.height + FG_ROW_GAP_PX
    hij_y0 = fg_y0 + fg.height + HIJ_ROW_GAP_PX
    height = hij_y0 + hij.height
    gradient = vertical_gradient(height, fg_y0, hij_y0 + hij.height, width, gradient_top, gradient_bottom)

    # These are the registered solid backgrounds of the source rows.  The
    # field colormaps and line colours are intentionally not included.
    fg, fg_masked = replace_registered_backgrounds(
        fg,
        [(255, 255, 255), (245, 247, 248)],
        gradient,
        fg_y0,
    )
    hij, hij_masked = replace_registered_backgrounds(
        hij,
        [(255, 255, 255), (244, 248, 251), (246, 247, 247)],
        gradient,
        hij_y0,
    )

    preserved_fg_left_px = 0
    preserved_fg_background_pixels_recolored = 0
    if args.previous_bj is None:
        canvas = Image.fromarray(np.full((height, width, 4), 255, dtype=np.uint8), mode="RGBA")
        # Extend the gradient through the f-g/h-i-j rows and both inter-row gaps.
        canvas.paste(Image.fromarray(np.dstack([gradient, np.full((height, width), 255, dtype=np.uint8)]), mode="RGBA"), (0, 0))
        canvas.alpha_composite(bcde, (0, 0))
        canvas.alpha_composite(fg, (0, fg_y0))
    else:
        canvas = Image.open(args.previous_bj).convert("RGBA")
        if canvas.size != (width, height):
            raise ValueError(
                f"Registered b-j size mismatch: {canvas.size} != {(width, height)}"
            )
        preserved_fg_left_px = int(args.preserve_fg_left_px)
        if not 0 <= preserved_fg_left_px <= width:
            raise ValueError(f"Invalid preserve-fg-left-px: {preserved_fg_left_px}")
        if preserved_fg_left_px:
            if args.previous_theme is not None and args.previous_theme != args.theme:
                previous = THEMES[args.previous_theme]
                previous_gradient = vertical_gradient(
                    height, fg_y0, hij_y0 + hij.height, width,
                    np.asarray(previous["gradient_top"], dtype=np.float64),
                    np.asarray(previous["gradient_bottom"], dtype=np.float64),
                )
                preserved = canvas.crop(
                    (0, fg_y0, preserved_fg_left_px, fg_y0 + fg.height)
                )
                preserved, preserved_fg_background_pixels_recolored = replace_registered_gradient(
                    preserved,
                    previous_gradient,
                    gradient,
                    fg_y0,
                )
                canvas.alpha_composite(preserved, (0, fg_y0))
            canvas.alpha_composite(
                fg.crop((preserved_fg_left_px, 0, width, fg.height)),
                (preserved_fg_left_px, fg_y0),
            )
        else:
            canvas.alpha_composite(fg, (0, fg_y0))
    revised_hij_left_px = 0
    if args.previous_bj is not None and args.revise_hij_left_px:
        revised_hij_left_px = int(args.revise_hij_left_px)
        if not 0 < revised_hij_left_px <= width:
            raise ValueError(f"Invalid revise-hij-left-px: {revised_hij_left_px}")
        canvas.alpha_composite(
            hij.crop((0, 0, revised_hij_left_px, hij.height)),
            (0, hij_y0),
        )
    else:
        canvas.alpha_composite(hij, (0, hij_y0))

    # Keep the inter-row spacing continuous with the selected background
    # without touching any plotted or labelled pixels.
    if args.previous_bj is not None:
        gradient_rgba = Image.fromarray(
            np.dstack([gradient, np.full((height, width), 255, dtype=np.uint8)]),
            mode="RGBA",
        )
        upper_gap = gradient_rgba.crop((0, bcde.height, width, fg_y0))
        middle_gap = gradient_rgba.crop((0, fg_y0 + fg.height, width, hij_y0))
        canvas.paste(upper_gap, (0, bcde.height))
        canvas.paste(middle_gap, (0, fg_y0 + fg.height))

    stem = args.output / args.stem
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    canvas.save(png, dpi=(DPI, DPI))
    canvas.convert("RGB").save(tiff, compression="tiff_lzw", dpi=(DPI, DPI))

    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_axis_off()
    fig.savefig(pdf, dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(svg, dpi=DPI, facecolor="white", pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": args.stem,
        "composition": (
            f"half-strength stage-tinted bcde source plus fg/hij rows with {theme['description']}"
            if args.previous_bj is None
            else "registered b-j base preserved outside revised g and h-j source panels"
        ),
        "canvas": {
            "width_px": width,
            "height_px": height,
            "dpi": DPI,
            "fg_row_gap_px": FG_ROW_GAP_PX,
            "hij_row_gap_px": HIJ_ROW_GAP_PX,
        },
        "gradient": {
            "theme": args.theme,
            "top_rgb": gradient_top.astype(int).tolist(),
            "bottom_rgb": gradient_bottom.astype(int).tolist(),
            "tint_scale_relative_to_previous": 0.5,
            "scope": "fg and hij row regions plus their inter-row gaps",
            "scientific_pixels_changed": "only registered background-colour pixels in fg/hij; bcde unchanged",
        },
        "sources": {
            "bcde": str(args.bcde),
            "fg": str(FG),
            "hij": str(args.hij),
            "previous_bj": str(args.previous_bj) if args.previous_bj is not None else None,
        },
        "pixel_preservation": {
            "enabled": args.previous_bj is not None,
            "fg_left_pixels_preserved": preserved_fg_left_px,
            "revised_fg_x_range": [preserved_fg_left_px, width],
            "revised_hij_full_row": not bool(revised_hij_left_px),
            "revised_hij_left_pixels": revised_hij_left_px,
            "hij_right_preserved": bool(revised_hij_left_px),
            "preserved_fg_background_pixels_recolored": preserved_fg_background_pixels_recolored,
            "previous_theme": args.previous_theme,
        },
        "registered_background_colours": {
            "fg": [(255, 255, 255), (245, 247, 248)],
            "hij": [(255, 255, 255), (244, 248, 251), (246, 247, 247)],
        },
        "masked_pixel_counts": {"fg": fg_masked, "hij": hij_masked},
        "outputs": {suffix: str(path) for suffix, path in {".png": png, ".tiff": tiff, ".pdf": pdf, ".svg": svg}.items()},
    }
    metadata["output_sha256"] = {suffix: sha256(path) for suffix, path in {".png": png, ".tiff": tiff, ".pdf": pdf, ".svg": svg}.items()}
    (args.output / f"{args.stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
