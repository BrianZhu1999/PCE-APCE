"""Compose Figure 5 with a cropped panel a and a unified mist-continuum background.

The approved panel-a PDF and approved b--j PNG remain the scientific sources.
This compositor only:
  1) trims the registered blank margin below panel a;
  2) replaces edge-connected, low-chroma background pixels with one restrained
     blue-white -> neutral -> warm-mist -> sage-mist vertical gradient;
  3) leaves scientific pixels, text, curves, cloud maps and legends untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compose_figure5_full_with_panel_a import (
    BJ_METADATA_DEFAULT,
    BJ_PNG_DEFAULT,
    DPI,
    HERE,
    LOCAL_BJ_MIRROR,
    PANEL_A_PDF_DEFAULT,
    PANEL_A_SCRIPT_DEFAULT,
    match_edge_dimension_without_scaling,
    sha256_file,
)


OUTDIR_DEFAULT = HERE / "combined_abcde_fg_coordinated_mist_soft_gradient_source_layout" / "outputs"
STEM_DEFAULT = "figure5_abcde_fg_combined_coordinated_mist_soft_gradient_source_layout_fghij"
PANEL_A_TOP_MARGIN_PX = 120
PANEL_A_BOTTOM_MARGIN_PX = 60
# Retained for the explicitly requested aligned-expanded export; the
# source-layout default itself does not invoke anchor-based raster scaling.
ANCHOR_DARK_THRESHOLD = 80
ANCHOR_SEARCH_ROWS = 320

# Very light, low-chroma and continuous: cool blue-white -> neutral mist ->
# warm ivory mist -> sage mist.  The colors are backgrounds, not data colors.
GRADIENT_STOPS = [
    (0.00, (247, 250, 251)),
    (0.30, (249, 249, 247)),
    (0.58, (249, 247, 243)),
    (1.00, (238, 245, 242)),
]


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def render_pdf_first_page(pdf_path: Path, dpi: int) -> Image.Image:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required to render the approved panel-a PDF.")
    with tempfile.TemporaryDirectory(prefix="figure5a_render_") as tmp:
        prefix = Path(tmp) / "panel_a"
        subprocess.run(
            [pdftoppm, "-f", "1", "-singlefile", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True,
        )
        rendered = prefix.with_suffix(".png")
        with Image.open(rendered) as img:
            return img.convert("RGBA").copy()


def save_pdf_svg_from_canvas(canvas: Image.Image, pdf: Path, svg: Path, dpi: int) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    width, height = canvas.size
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(canvas)
    ax.set_axis_off()
    fig.savefig(pdf, dpi=dpi, facecolor="white", pad_inches=0)
    fig.savefig(svg, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def crop_panel_a_blank_margin(
    image: Image.Image,
    *,
    top_margin_px: int,
    bottom_margin_px: int,
    nonwhite_threshold: int = 3,
) -> tuple[Image.Image, dict[str, int]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    deviation = np.max(np.abs(rgb - 255), axis=2)
    row_has_content = np.any(deviation > nonwhite_threshold, axis=1)
    ys = np.flatnonzero(row_has_content)
    if ys.size == 0:
        raise ValueError("Panel-a crop failed: no non-white content detected.")
    y_min = max(0, int(ys[0]) - int(top_margin_px))
    y_max_exclusive = min(image.height, int(ys[-1]) + 1 + int(bottom_margin_px))
    if y_max_exclusive <= y_min:
        raise ValueError(f"Invalid panel-a crop: {y_min}:{y_max_exclusive}")
    cropped = image.crop((0, y_min, image.width, y_max_exclusive))
    return cropped, {
        "source_height_px": image.height,
        "content_y_min_px": int(ys[0]),
        "content_y_max_px": int(ys[-1]),
        "crop_top_px": y_min,
        "crop_bottom_exclusive_px": y_max_exclusive,
        "cropped_height_px": cropped.height,
        "top_margin_px": int(top_margin_px),
        "bottom_margin_px": int(bottom_margin_px),
    }


def find_top_left_dark_anchor(
    image: Image.Image,
    *,
    search_rows: int = ANCHOR_SEARCH_ROWS,
    dark_threshold: int = ANCHOR_DARK_THRESHOLD,
) -> dict[str, int]:
    """Find the first dark mark in the title/label band.

    In the registered source images this is the panel letter (a or b).  The
    search is deliberately restricted to the top band so that plot axes and
    colourbars cannot become the alignment anchor.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    band = rgb[: min(int(search_rows), image.height)]
    dark = np.max(band, axis=2) < int(dark_threshold)
    ys, xs = np.where(dark)
    if xs.size == 0:
        raise ValueError("Could not find a dark panel-label anchor.")
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
        "search_rows": int(search_rows),
        "dark_threshold": int(dark_threshold),
    }


def find_leftmost_dark_anchor_in_band(
    image: Image.Image,
    *,
    y_min: int,
    y_max: int,
    dark_threshold: int = ANCHOR_DARK_THRESHOLD,
) -> dict[str, int]:
    """Find the leftmost dark panel label in a specified vertical band."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    y0 = max(0, int(y_min))
    y1 = min(image.height, int(y_max))
    if y1 <= y0:
        raise ValueError(f"Invalid anchor search band: {y0}:{y1}")
    band = rgb[y0:y1]
    dark = np.max(band, axis=2) < int(dark_threshold)
    ys, xs = np.where(dark)
    if xs.size == 0:
        raise ValueError(f"Could not find a dark panel-label anchor in band {y0}:{y1}.")
    return {
        "x": int(xs.min()),
        "y": int(ys.min() + y0),
        "x_max": int(xs.max()),
        "y_max": int(ys.max() + y0),
        "search_y_min": int(y0),
        "search_y_max": int(y1),
        "dark_threshold": int(dark_threshold),
    }


def scale_and_align_panel_a(
    image: Image.Image,
    *,
    source_anchor: dict[str, int],
    target_anchor: dict[str, int],
    scale: float,
) -> tuple[Image.Image, dict[str, object]]:
    """Expand panel a from its label anchor and align it to panel b.

    The affine map is applied to the complete raster rather than cropping to
    the anchor.  This preserves anti-aliased parts of the axis glyphs and
    labels that can extend slightly above or left of the first dark pixel.
    """
    if scale <= 1.0:
        raise ValueError(f"Panel-a scale must be > 1, got {scale}")

    sx = int(source_anchor["x"])
    sy = int(source_anchor["y"])
    tx = int(target_anchor["x"])
    ty = sy  # preserve the current vertical placement; only x alignment changes
    if sx < 0 or sy < 0 or sx >= image.width or sy >= image.height:
        raise ValueError(f"Invalid source anchor {source_anchor} for image {image.size}")
    if tx < 0 or tx >= image.width:
        raise ValueError(f"Invalid target anchor {target_anchor} for image {image.size}")

    output_height = ty + max(1, int(round((image.height - sy) * float(scale))))
    # output -> source inverse map for:
    #   x' = tx + scale * (x - sx)
    #   y' = ty + scale * (y - sy)
    inverse = (
        1.0 / float(scale),
        0.0,
        float(sx) - float(tx) / float(scale),
        0.0,
        1.0 / float(scale),
        float(sy) - float(ty) / float(scale),
    )
    transformed = image.transform(
        (image.width, output_height),
        Image.Transform.AFFINE,
        inverse,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255, 255),
    )

    metadata = {
        "operation": "anchor_scale_then_left_align",
        "scale": float(scale),
        "source_anchor": dict(source_anchor),
        "target_anchor": dict(target_anchor),
        "target_anchor_y_preserved_from_source": int(ty),
        "source_roi_size_px": [int(image.width), int(image.height)],
        "scaled_roi_size_px": [
            int(round(image.width * float(scale))),
            int(round(image.height * float(scale))),
        ],
        "output_size_px": [int(transformed.width), int(transformed.height)],
        "right_edge_clipped_px": max(
            0,
            int(round(tx + (image.width - sx) * float(scale) - transformed.width)),
        ),
    }
    return transformed, metadata


def make_gradient(width: int, height: int) -> np.ndarray:
    positions = np.linspace(0.0, 1.0, height, dtype=np.float64)
    stop_x = np.asarray([item[0] for item in GRADIENT_STOPS], dtype=np.float64)
    stop_rgb = np.asarray([item[1] for item in GRADIENT_STOPS], dtype=np.float64)
    rows = np.column_stack([np.interp(positions, stop_x, stop_rgb[:, channel]) for channel in range(3)])
    rgb = np.repeat(np.rint(rows).astype(np.uint8)[:, None, :], width, axis=1)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def edge_connected_background_mask(
    image: Image.Image,
    *,
    mean_threshold: float,
    chroma_threshold: float,
    downsample: int = 8,
) -> np.ndarray:
    """Find broad background regions without recoloring enclosed white data areas.

    The connectivity is evaluated on a small image, which is sufficient for
    broad page backgrounds and keeps the 10553 x 10730 source tractable.
    """
    width, height = image.size
    small_w = max(1, int(np.ceil(width / downsample)))
    small_h = max(1, int(np.ceil(height / downsample)))
    small = np.asarray(
        image.convert("RGB").resize((small_w, small_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    mean = small.mean(axis=2)
    chroma = small.max(axis=2).astype(np.float32) - small.min(axis=2).astype(np.float32)
    candidate = (mean >= mean_threshold) & (chroma <= chroma_threshold)

    connected = np.zeros(candidate.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    edge_points: list[tuple[int, int]] = []
    edge_points.extend((0, x) for x in np.flatnonzero(candidate[0]))
    if small_h > 1:
        edge_points.extend((small_h - 1, x) for x in np.flatnonzero(candidate[-1]))
    if small_w > 1:
        edge_points.extend((y, 0) for y in np.flatnonzero(candidate[:, 0]))
        edge_points.extend((y, small_w - 1) for y in np.flatnonzero(candidate[:, -1]))
    for y, x in edge_points:
        if not connected[y, x]:
            connected[y, x] = True
            queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        if y > 0 and candidate[y - 1, x] and not connected[y - 1, x]:
            connected[y - 1, x] = True
            queue.append((y - 1, x))
        if y + 1 < small_h and candidate[y + 1, x] and not connected[y + 1, x]:
            connected[y + 1, x] = True
            queue.append((y + 1, x))
        if x > 0 and candidate[y, x - 1] and not connected[y, x - 1]:
            connected[y, x - 1] = True
            queue.append((y, x - 1))
        if x + 1 < small_w and candidate[y, x + 1] and not connected[y, x + 1]:
            connected[y, x + 1] = True
            queue.append((y, x + 1))

    mask = Image.fromarray((connected.astype(np.uint8) * 255), mode="L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    return np.asarray(mask, dtype=np.uint8) > 0


def apply_gradient_to_background(
    image: Image.Image,
    gradient_rgba: np.ndarray,
    *,
    mean_threshold: float,
    chroma_threshold: float,
) -> tuple[Image.Image, int]:
    rgba = np.asarray(image.convert("RGBA")).copy()
    mask = edge_connected_background_mask(
        image,
        mean_threshold=mean_threshold,
        chroma_threshold=chroma_threshold,
        downsample=8,
    )
    rgba[mask, :3] = gradient_rgba[mask, :3]
    return Image.fromarray(rgba, mode="RGBA"), int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop panel-a whitespace and apply a unified Figure 5 mist gradient.")
    parser.add_argument("--panel-a-pdf", type=Path, default=PANEL_A_PDF_DEFAULT)
    parser.add_argument("--panel-a-script", type=Path, default=PANEL_A_SCRIPT_DEFAULT)
    parser.add_argument("--bj-png", type=Path, default=BJ_PNG_DEFAULT)
    parser.add_argument("--bj-metadata", type=Path, default=BJ_METADATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--stem", default=STEM_DEFAULT)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--top-margin-px", type=int, default=PANEL_A_TOP_MARGIN_PX)
    parser.add_argument("--bottom-margin-px", type=int, default=PANEL_A_BOTTOM_MARGIN_PX)
    parser.add_argument(
        "--panel-a-scale",
        type=float,
        default=1.0,
        help="Optional legacy raster expansion; keep 1.0 for the source-layout Figure 5 output.",
    )
    parser.add_argument("--local-bj-mirror", default=LOCAL_BJ_MIRROR)
    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    args.output.mkdir(parents=True, exist_ok=True)

    panel_a_raw = render_pdf_first_page(args.panel_a_pdf, args.dpi)
    bj = Image.open(args.bj_png).convert("RGBA")
    panel_a_width_matched, width_adjustment = match_edge_dimension_without_scaling(
        panel_a_raw,
        target_width=bj.width,
        target_height=None,
    )
    panel_a, crop_metadata = crop_panel_a_blank_margin(
        panel_a_width_matched,
        top_margin_px=args.top_margin_px,
        bottom_margin_px=args.bottom_margin_px,
    )
    if panel_a.width != bj.width:
        raise ValueError(f"Width mismatch after panel-a crop: a={panel_a.width}, b-j={bj.width}")

    panel_a_expansion_alignment = None
    if args.panel_a_scale > 1.0:
        source_anchor = find_top_left_dark_anchor(panel_a)
        target_anchor = find_leftmost_dark_anchor_in_band(
            bj,
            y_min=250,
            y_max=650,
        )
        panel_a, panel_a_expansion_alignment = scale_and_align_panel_a(
            panel_a,
            source_anchor=source_anchor,
            target_anchor=target_anchor,
            scale=args.panel_a_scale,
        )

    width = panel_a.width
    height = panel_a.height + bj.height
    gradient = make_gradient(width, height)

    # A uses a stricter background threshold because its source background is
    # white.  The registered b-j composite is kept pixel-identical: applying
    # a broad low-chroma mask to it can recolor borderless scientific
    # colorbars whose neutral midpoint is connected to the page background.
    panel_a_gradient, panel_a_mask_count = apply_gradient_to_background(
        panel_a,
        gradient[: panel_a.height],
        mean_threshold=248.0,
        chroma_threshold=12.0,
    )
    bj_gradient = bj
    bj_mask_count = 0

    canvas = Image.fromarray(gradient, mode="RGBA")
    canvas.alpha_composite(panel_a_gradient, (0, 0))
    canvas.alpha_composite(bj_gradient, (0, panel_a.height))

    stem = args.output / args.stem
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    metadata_path = args.output / f"{args.stem}_metadata.json"
    audit_path = args.output / f"{args.stem}_audit.txt"
    registry_path = args.output / f"{args.stem}_panel_registry.csv"

    canvas.save(png, dpi=(args.dpi, args.dpi))
    canvas.convert("RGB").save(tiff, compression="tiff_lzw", dpi=(args.dpi, args.dpi))
    save_pdf_svg_from_canvas(canvas, pdf, svg, args.dpi)

    registry_rows = [
        {
            "panel": "a",
            "content": "approved cropped method-entry panel",
            "local_source": "",
            "remote_source": str(args.panel_a_pdf),
            "selection": "user-approved current PDF; vertical blank margin trimmed only",
            "source_script": str(args.panel_a_script),
        },
        {
            "panel": "b-j",
            "content": "approved current b-j composite",
            "local_source": args.local_bj_mirror,
            "remote_source": str(args.bj_png),
            "selection": "user-specified coordinated-mist-soft b-j source",
            "source_script": "source scripts recorded in b-j metadata",
        },
    ]
    write_csv(registry_path, registry_rows)

    bj_metadata = None
    if args.bj_metadata.exists():
        bj_metadata = json.loads(args.bj_metadata.read_text(encoding="utf-8"))

    outputs = {".png": png, ".tiff": tiff, ".pdf": pdf, ".svg": svg}
    metadata = {
        "figure": args.stem,
        "composition": "cropped approved panel a plus approved b-j composite on one unified mist-continuum background",
        "backend": "Python compositor; Poppler PDF rasterization and matplotlib PDF/SVG container export",
        "canvas": {
            "width_px": width,
            "height_px": height,
            "dpi": args.dpi,
            "panel_a_height_px": panel_a.height,
            "bj_height_px": bj.height,
            "vertical_gap_px": 0,
            "widths_matched_without_scaling": True,
            "panel_a_width_adjustment": width_adjustment,
        },
        "panel_a_crop": crop_metadata,
        "panel_a_expansion_alignment": panel_a_expansion_alignment,
        "panel_a_source_layout_change": (
            "panel a was redrawn from its original matplotlib script; compositor did not scale or warp it"
            if panel_a_expansion_alignment is None
            else "legacy compositor raster expansion was explicitly requested"
        ),
        "gradient": {
            "name": "mist-continuum",
            "stops": [{"position": float(p), "rgb": list(rgb)} for p, rgb in GRADIENT_STOPS],
            "description": "very-light cool blue-white to neutral mist to warm ivory mist to sage mist",
            "scope": "panel-a background pixels replaced by the gradient; registered b-j composite kept pixel-identical",
            "scientific_pixels_recolored": "only edge-connected low-chroma background candidates",
            "bj_scientific_source_preserved": True,
        },
        "background_masks": {
            "panel_a_mean_threshold": 248.0,
            "panel_a_chroma_threshold": 12.0,
            "bj_mean_threshold": None,
            "bj_chroma_threshold": None,
            "panel_a_pixels_recolored": panel_a_mask_count,
            "bj_pixels_recolored": bj_mask_count,
        },
        "sources": {
            "panel_a_pdf_remote": str(args.panel_a_pdf),
            "panel_a_script_remote": str(args.panel_a_script),
            "bj_png_remote_mirror": str(args.bj_png),
            "bj_png_local_user_source": args.local_bj_mirror,
            "bj_metadata_remote_mirror": str(args.bj_metadata) if args.bj_metadata.exists() else None,
        },
        "source_dimensions_px": {
            "panel_a_raw_rendered": list(panel_a_raw.size),
            "panel_a_width_matched": list(panel_a_width_matched.size),
            "panel_a_cropped": list(panel_a.size),
            "bj": list(bj.size),
        },
        "source_sha256": {
            "panel_a_pdf": sha256_file(args.panel_a_pdf),
            "panel_a_script": sha256_file(args.panel_a_script) if args.panel_a_script.exists() else None,
            "bj_png": sha256_file(args.bj_png),
            "bj_metadata": sha256_file(args.bj_metadata) if args.bj_metadata.exists() else None,
        },
        "bj_source_metadata_summary": {
            "figure": bj_metadata.get("figure") if isinstance(bj_metadata, dict) else None,
            "canvas": bj_metadata.get("canvas") if isinstance(bj_metadata, dict) else None,
            "sources": bj_metadata.get("sources") if isinstance(bj_metadata, dict) else None,
        },
        "panel_registry": str(registry_path),
        "outputs": {ext: str(path) for ext, path in outputs.items()},
        "output_sizes_bytes": {ext: file_size(path) for ext, path in outputs.items()},
        "output_sha256": {ext: sha256_file(path) for ext, path in outputs.items()},
    }
    metadata["output_size_exactly_matches_metadata"] = canvas.size == (
        metadata["canvas"]["width_px"],
        metadata["canvas"]["height_px"],
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        f"figure={args.stem}",
        f"dpi={args.dpi}",
        f"panel_a_pdf={args.panel_a_pdf}",
        f"panel_a_raw_rendered_size={panel_a_raw.width}x{panel_a_raw.height}",
        f"panel_a_cropped_size={panel_a.width}x{panel_a.height}",
        f"panel_a_crop={crop_metadata}",
        f"panel_a_expansion_alignment={panel_a_expansion_alignment}",
        f"panel_a_width_adjustment={width_adjustment}",
        f"bj_png={args.bj_png}",
        f"bj_size={bj.width}x{bj.height}",
        f"panel_a_background_pixels_recolored={panel_a_mask_count}",
        f"bj_background_pixels_recolored={bj_mask_count}",
        f"final_size={width}x{height}",
        "vertical_gap_px=0",
        "gradient=mist-continuum",
        f"panel_registry={registry_path}",
        f"metadata={metadata_path}",
        f"output_size_exactly_matches_metadata={metadata['output_size_exactly_matches_metadata']}",
    ]
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "outputs": metadata["outputs"],
                "metadata": str(metadata_path),
                "audit": str(audit_path),
                "panel_registry": str(registry_path),
                "final_size": [width, height],
                "panel_a_crop": crop_metadata,
                "background_pixels_recolored": {
                    "panel_a": panel_a_mask_count,
                    "bj": bj_mask_count,
                },
                "size_exact": metadata["output_size_exactly_matches_metadata"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
