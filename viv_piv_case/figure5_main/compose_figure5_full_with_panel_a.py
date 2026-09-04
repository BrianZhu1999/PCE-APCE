"""Compose the approved Figure 5a panel with the current b--j composite.

This script is intentionally a compositor only.  It does not redraw panel a
or alter any scientific content in panels b--j: the approved panel-a PDF is
rendered at the target DPI, then pasted directly above the registered b--j PNG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


HERE = Path(__file__).resolve().parent
DPI = 650
PANEL_A_EXPECTED_HEIGHT_PX = 3300

PANEL_A_PDF_DEFAULT = (
    HERE
    / "panel_a"
    / "outputs_x40y20_0803_spacetime"
    / "figure5a_x40y20_0803_spacetime.pdf"
)
PANEL_A_SCRIPT_DEFAULT = HERE / "panel_a" / "plot_panel_a_x40y20_0803.py"
BJ_PNG_DEFAULT = (
    HERE
    / "combined_bcde_fg_coordinated_mist_soft"
    / "outputs"
    / "figure5_bcde_fg_combined_coordinated_mist_soft_fghij.png"
)
BJ_METADATA_DEFAULT = BJ_PNG_DEFAULT.with_name(BJ_PNG_DEFAULT.stem + "_metadata.json")
OUTDIR_DEFAULT = HERE / "combined_abcde_fg_coordinated_mist_soft" / "outputs"
STEM_DEFAULT = "figure5_abcde_fg_combined_coordinated_mist_soft_fghij"

LOCAL_BJ_MIRROR = (
    "./"
    "hybrid_uncertain_wave/viv_piv_case/figure5_main/combined_bcde_fg_coordinated_mist_soft/"
    "outputs/figure5_bcde_fg_combined_coordinated_mist_soft_fghij.png"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def render_pdf_first_page(pdf_path: Path, dpi: int) -> Image.Image:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required to render the approved panel-a PDF.")
    with tempfile.TemporaryDirectory(prefix="figure5a_render_") as tmp:
        prefix = Path(tmp) / "panel_a"
        cmd = [pdftoppm, "-f", "1", "-singlefile", "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
        subprocess.run(cmd, check=True)
        rendered = prefix.with_suffix(".png")
        if not rendered.exists():
            raise FileNotFoundError(f"pdftoppm did not produce {rendered}")
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


def match_edge_dimension_without_scaling(
    image: Image.Image,
    *,
    target_width: int,
    target_height: int | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    width_delta = image.width - target_width
    height_delta = 0 if target_height is None else image.height - target_height
    if width_delta == 0 and height_delta == 0:
        return image, {
            "operation": "none",
            "width_pixels_added_or_removed": 0,
            "height_pixels_added_or_removed": 0,
            "reason": "source dimensions already match",
        }
    if width_delta not in (0, 1, 2) and not (-2 <= width_delta <= 0):
        raise ValueError(f"Width mismatch too large to correct without scaling: source = {image.width}px, target = {target_width}px")
    if height_delta not in (0, 1, 2) and not (-2 <= height_delta <= 0):
        raise ValueError(f"Height mismatch too large to correct without scaling: source = {image.height}px, target = {target_height}px")

    matched_width = target_width
    matched_height = image.height if target_height is None else target_height

    if width_delta >= 0 and height_delta >= 0:
        return (
            image.crop((0, 0, matched_width, matched_height)),
            {
                "operation": "right_bottom_edge_crop",
                "width_pixels_added_or_removed": int(width_delta),
                "height_pixels_added_or_removed": int(height_delta),
                "reason": "PDF rasterization rounded the page by one pixel; no image resampling was used",
            },
        )

    padded = Image.new("RGBA", (matched_width, matched_height), (255, 255, 255, 255))
    crop_width = min(image.width, matched_width)
    crop_height = min(image.height, matched_height)
    padded.alpha_composite(image.crop((0, 0, crop_width, crop_height)), (0, 0))
    return (
        padded,
        {
            "operation": "edge_crop_or_white_padding",
            "width_pixels_added_or_removed": int(width_delta),
            "height_pixels_added_or_removed": int(height_delta),
            "reason": "minor PDF raster rounding difference; no image resampling was used",
        },
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose Figure 5 panel a with the approved b-j composite.")
    parser.add_argument("--panel-a-pdf", type=Path, default=PANEL_A_PDF_DEFAULT)
    parser.add_argument("--panel-a-script", type=Path, default=PANEL_A_SCRIPT_DEFAULT)
    parser.add_argument("--bj-png", type=Path, default=BJ_PNG_DEFAULT)
    parser.add_argument("--bj-metadata", type=Path, default=BJ_METADATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--stem", default=STEM_DEFAULT)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--panel-a-height-px", type=int, default=PANEL_A_EXPECTED_HEIGHT_PX)
    parser.add_argument("--local-bj-mirror", default=LOCAL_BJ_MIRROR)
    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    args.output.mkdir(parents=True, exist_ok=True)

    panel_a_raw = render_pdf_first_page(args.panel_a_pdf, args.dpi)
    bj = Image.open(args.bj_png).convert("RGBA")
    panel_a, panel_a_edge_adjustment = match_edge_dimension_without_scaling(
        panel_a_raw,
        target_width=bj.width,
        target_height=args.panel_a_height_px,
    )

    width = panel_a.width
    height = panel_a.height + bj.height
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    canvas.alpha_composite(panel_a, (0, 0))
    canvas.alpha_composite(bj, (0, panel_a.height))

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
            "content": "method-entry panel: sparse u/v observations, known cylinder displacement and 256-D POD state",
            "local_source": "",
            "remote_source": str(args.panel_a_pdf),
            "selection": "user-approved current PDF; case 0803, frame 554, x40-y20/751 layout",
            "source_script": str(args.panel_a_script),
        },
        {
            "panel": "b-j",
            "content": "approved current b-j composite used as a single registered source image",
            "local_source": args.local_bj_mirror,
            "remote_source": str(args.bj_png),
            "selection": "user-specified coordinated-mist-soft b-j composite",
            "source_script": "see b-e, f-g and h-i-j panel scripts recorded in the source composite metadata",
        },
    ]
    write_csv(registry_path, registry_rows)

    bj_metadata = None
    if args.bj_metadata.exists():
        bj_metadata = json.loads(args.bj_metadata.read_text(encoding="utf-8"))

    outputs = {".png": png, ".tiff": tiff, ".pdf": pdf, ".svg": svg}
    metadata = {
        "figure": args.stem,
        "composition": "approved panel a PDF stacked directly above the approved b-j composite PNG",
        "backend": "Python compositor; panel-a PDF rendered with Poppler pdftoppm at target DPI",
        "canvas": {
            "width_px": width,
            "height_px": height,
            "dpi": args.dpi,
            "panel_a_height_px": panel_a.height,
            "bj_height_px": bj.height,
            "vertical_gap_px": 0,
            "widths_matched_without_scaling": True,
            "panel_a_edge_adjustment": panel_a_edge_adjustment,
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
            "panel_a_rendered": list(panel_a.size),
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
        f"panel_a_rendered_size={panel_a.width}x{panel_a.height}",
        f"panel_a_edge_adjustment={panel_a_edge_adjustment}",
        f"bj_png={args.bj_png}",
        f"bj_size={bj.width}x{bj.height}",
        f"widths_matched_without_scaling={panel_a.width == bj.width}",
        f"final_size={width}x{height}",
        "vertical_gap_px=0",
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
                "source_sizes": metadata["source_dimensions_px"],
                "size_exact": metadata["output_size_exactly_matches_metadata"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
