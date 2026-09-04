#!/usr/bin/env python3
"""Run structural and export QA for the Baoding five-panel figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image


STEM = "supplementary_data_figure2_baoding_single_dual_field"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgba_is_white(rgba: list[float], tolerance: float = 0.02) -> bool:
    return len(rgba) >= 3 and all(abs(float(value) - 1.0) <= tolerance for value in rgba[:3])


def rgba_is_dark(rgba: list[float], tolerance: float = 0.20) -> bool:
    return len(rgba) >= 3 and all(float(value) <= tolerance for value in rgba[:3])


def bbox_inside(inner: list[float], outer: list[float], tolerance: float = 1e-6) -> bool:
    return (
        len(inner) == 4 and len(outer) == 4
        and inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument(
        "--visual-review-confirmed",
        action="store_true",
        help="Mark the separately performed PNG visual inspection as complete.",
    )
    args = parser.parse_args()

    root = args.figure_root
    paths = {suffix: root / f"{STEM}.{suffix}" for suffix in ("png", "pdf", "svg", "tiff")}
    registry_path = root / f"{STEM}_registry.json"
    source_path = root / f"{STEM}_source.csv"
    panel_path = root / f"{STEM}_panel_registry.csv"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source_rows = read_csv(source_path)
    panel_rows = read_csv(panel_path)

    checks: dict[str, bool] = {}
    checks["all_exports_nonempty"] = all(path.is_file() and path.stat().st_size > 10_000 for path in paths.values())
    checks["five_registered_panels"] = [row["panel"] for row in panel_rows] == list("abcde")
    checks["panel_d_is_reliability_diagnostic"] = registry["figure_contract"]["panel_map"].get("d") == "calibration-only node-target DOA reliability"
    checks["confidential_closeup_not_mirrored"] = not (root / "source_photo_2017_helicopter.jpeg").exists()
    checks["quantitative_cells_square"] = all(
        abs(float(box["physical_width_to_height"]) - 1.0) < 1e-6
        for panel, box in registry["layout_qa"]["panel_boxes"].items() if panel in "abcd"
    )
    checks["photo_frames_absent"] = registry["layout_qa"]["photo_frames"] == "none"
    checks["panel_e_photo_uncropped_without_overlay"] = all(
        registry["image_integrity"][panel].get("crop") == "none"
        and registry["image_integrity"][panel].get("overlay") == "none"
        for panel in ("panel_e",)
    )
    checks["panel_d_reliability_source_recorded"] = (
        registry["image_integrity"]["panel_d"].get("type") == "quantitative diagnostic"
        and bool(registry["image_integrity"]["panel_d"].get("not_an_image"))
        and any(
            token in registry["figure_contract"]["reviewer_risks"][-1]
            for token in ("dB SNR", "dB-SNR")
        )
    )
    checks["e_photo_table_same_width"] = bool(registry["layout_qa"].get("photo_table_same_x_bounds"))
    checks["e_right_edge_matches_c"] = bool(registry["layout_qa"].get("e_right_aligned_with_c"))
    palettes = registry.get("panel_palettes", {})
    checks["c_d_e_palettes_are_distinct"] = (
        set(palettes.get("c", {}).values()).isdisjoint(set(palettes.get("d", {}).values()))
        and set(palettes.get("c", {}).values()).isdisjoint(set(palettes.get("e", {}).values()))
        and set(palettes.get("d", {}).values()).isdisjoint(set(palettes.get("e", {}).values()))
    )
    checks["d_palette_high_separation"] = palettes.get("d") == {
        "target1": "#8E5AA8", "target2": "#D49A28"
    }
    checks["d_legend_single_row"] = (
        registry.get("reliability_semantics", {}).get("legend_columns") == 2
        and registry.get("reliability_semantics", {}).get("legend_layout") == "single row"
    )
    panel_c_semantics = registry["uncertainty_semantics"]["panel_c"]
    checks["panel_c_uses_normalized_progress"] = "normalized progress" in panel_c_semantics
    checks["canvas_scale_vs_v9 == 0.9"] = abs(float(registry["typography"]["canvas_scale_vs_v9"]) - 0.9) < 1e-9
    checks["canvas_is_90pct_of_v9"] = all(
        abs(float(value) - expected) < 1e-6
        for value, expected in zip(registry["layout_qa"]["canvas_inches"], (18.36, 12.24), strict=True)
    )
    shared_limits = registry["track_axes"]["ab_shared_limits"]
    a_limits = registry["track_axes"].get("a_limits", {})
    b_limits = registry["track_axes"].get("b_limits", {})
    checks["ab_axes_share_limits"] = (
        len(shared_limits.get("x", [])) == 2
        and len(shared_limits.get("y", [])) == 2
        and a_limits.get("x") == b_limits.get("x") == shared_limits.get("x")
        and a_limits.get("y") == b_limits.get("y") == shared_limits.get("y")
    )
    checks["endpoint_labels_removed"] = not bool(registry["track_axes"]["endpoint_labels"])
    checks["endpoint_markers_in_legend"] = bool(registry["track_axes"]["endpoint_markers_in_legend"])
    checks["ab_limits_frozen_to_v38"] = (
        shared_limits.get("x") == a_limits.get("x") == b_limits.get("x")
        == [-762.802347903294, 837.197652096706]
        and shared_limits.get("y") == a_limits.get("y") == b_limits.get("y")
        == [-900.0, 700.0]
    )
    checks["ab_equal_fixed_spans"] = (
        abs(float(shared_limits["x"][1]) - float(shared_limits["x"][0]) - 1600.0) < 1e-9
        and abs(float(shared_limits["y"][1]) - float(shared_limits["y"][0]) - 1600.0) < 1e-9
    )
    checks["node_labels_present_in_b"] = bool(registry["track_axes"]["node_labels_in_b"])
    checks["single_window_67_frames"] = (
        int(registry["single_window"].get("frames", registry["single_window"].get("length_frames", -1))) == 67
        and int(registry["single_window"]["start_time_s"]) == 46254
        and int(registry["single_window"]["end_time_s"]) == 46320
    )
    dual_window = registry["dual_window"]
    dual_length = int(dual_window["length_frames"])
    dual_start = int(dual_window["start_time_s"])
    dual_end = int(dual_window["end_time_s"])
    checks["dual_window_60_stable_segment_frames"] = (
        dual_length == 60
        and dual_start == 46593
        and dual_end == 46652
        and dual_end - dual_start == dual_length - 1
        and bool(dual_window.get("stable_segment_admitted"))
        and bool(dual_window.get("admitted_acoustic_selection"))
        and not bool(dual_window.get("complete_circle_geometry_admitted"))
    )
    checks["dual_admission_all_passed"] = all(
        bool(dual_window[key])
        for key in ("admitted_identity", "admitted_jump", "admitted_error", "admitted_uncertainty", "admitted_covariance")
    )
    reliability_profiles = registry.get("dual_frontend_reliability_profiles", {})
    reliability_selection = registry.get("dual_reliability_profile_selection") or {}
    checks["dual_target_specific_reliability_profile_frozen"] = (
        reliability_profiles == {"1": "acoustic_reliability", "2": "acoustic_reliability"}
        and reliability_selection.get("selection_status") == "frozen before 60-frame evaluation"
        and reliability_selection.get("calibration_interval_s") == [46540, 46560]
        and reliability_selection.get("evaluation_interval_s") == [46593, 46652]
        and reliability_selection.get("selected_profile_by_target") == {"1": "acoustic_reliability", "2": "acoustic_reliability"}
    )
    checks["source_row_count"] = len(source_rows) == 67 + 2 * dual_length

    metrics = registry["metrics"]
    # v32 intentionally replaces the v30 single-source showcase with the
    # isolated A6 frontend + established single-source backend.  The source
    # registry, rather than a stale v30 literal, is the numerical contract.
    checks["scientific_values_match_authoritative_contract"] = False
    authoritative_metrics: dict[str, float] = {}
    authoritative_error = ""
    try:
        source_paths = registry["sources"]
        authoritative_single = read_csv(Path(source_paths["single_source_csv"]))
        authoritative_dual = read_csv(Path(source_paths["dual_source_csv"]))
        single_errors = np.asarray([float(row["apce_position_error_of_median_m"]) for row in authoritative_single])
        authoritative_metrics["single_rmse_m"] = float(np.sqrt(np.mean(np.square(single_errors))))
        authoritative_metrics["single_median_error_m"] = float(np.median(single_errors))
        authoritative_metrics["single_p90_error_m"] = float(np.percentile(single_errors, 90.0))
        checks["authoritative_single_window"] = (
            len(authoritative_single) == 67
            and int(float(authoritative_single[0]["time_s"])) == 46254
            and int(float(authoritative_single[-1]["time_s"])) == 46320
        )
        checks["authoritative_dual_window"] = (
            len(authoritative_dual) == dual_length
            and int(float(authoritative_dual[0]["time_s"])) == dual_start
            and int(float(authoritative_dual[-1]["time_s"])) == dual_end
            and np.allclose(np.diff([float(row["elapsed_s"]) for row in authoritative_dual]), 1.0)
        )
        for target in (1, 2):
            errors = np.asarray([float(row[f"target{target}_apce_error_m"]) for row in authoritative_dual])
            authoritative_metrics[f"target{target}_rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
            authoritative_metrics[f"target{target}_median_error_m"] = float(np.median(errors))
            authoritative_metrics[f"target{target}_p90_error_m"] = float(np.percentile(errors, 90.0))
        derived_single = source_rows[: len(authoritative_single)]
        derived_dual = source_rows[len(authoritative_single) :]
        derived_match = len(derived_single) == len(authoritative_single) and len(derived_dual) == 2 * len(authoritative_dual)
        if derived_match:
            for derived, raw in zip(derived_single, authoritative_single, strict=True):
                derived_match &= abs(float(derived["time_s"]) - float(raw["time_s"])) < 1e-9
                derived_match &= abs(float(derived["position_error_m"]) - float(raw["apce_position_error_of_median_m"])) < 1e-9
            for index, raw in enumerate(authoritative_dual):
                for target in (1, 2):
                    derived = derived_dual[2 * index + target - 1]
                    derived_match &= int(derived["target"]) == target
                    derived_match &= abs(float(derived["time_s"]) - float(raw["time_s"])) < 1e-9
                    derived_match &= abs(float(derived["position_error_m"]) - float(raw[f"target{target}_apce_error_m"])) < 1e-9
        checks["derived_source_matches_authoritative"] = bool(derived_match)
        checks["authoritative_source_recomputed"] = (
            checks["authoritative_single_window"]
            and checks["authoritative_dual_window"]
            and abs(authoritative_metrics["single_rmse_m"] - float(metrics["single"]["rmse_m"])) < 1e-9
            and abs(authoritative_metrics["single_median_error_m"] - float(metrics["single"]["median_error_m"])) < 1e-9
            and abs(authoritative_metrics["single_p90_error_m"] - float(metrics["single"]["p90_error_m"])) < 1e-9
            and all(
                abs(authoritative_metrics[f"target{target}_rmse_m"] - float(metrics[f"dual_target{target}"]["rmse_m"])) < 1e-9
                for target in (1, 2)
            )
        )
        checks["scientific_values_match_authoritative_contract"] = checks["authoritative_source_recomputed"]
    except (KeyError, OSError, ValueError, IndexError) as exc:
        checks["authoritative_single_window"] = False
        checks["authoritative_dual_window"] = False
        checks["derived_source_matches_authoritative"] = False
        checks["authoritative_source_recomputed"] = False
        authoritative_error = f"{type(exc).__name__}: {exc}"

    with Image.open(paths["png"]) as image:
        png_size = list(image.size)
        rgb = np.asarray(image.convert("RGB"))
    panel_variances = []
    for row_index in range(2):
        for column_index in range(3):
            y0, y1 = row_index * rgb.shape[0] // 2, (row_index + 1) * rgb.shape[0] // 2
            x0, x1 = column_index * rgb.shape[1] // 3, (column_index + 1) * rgb.shape[1] // 3
            panel_variances.append(float(np.var(rgb[y0:y1, x0:x1])))
    checks["png_nonblank_all_grid_cells"] = all(value > 25.0 for value in panel_variances)

    with Image.open(paths["tiff"]) as image:
        tiff_size = list(image.size)
        tiff_dpi = [float(value) for value in image.info.get("dpi", (0.0, 0.0))]
    checks["tiff_600dpi"] = len(tiff_dpi) >= 2 and all(abs(value - 600.0) < 0.5 for value in tiff_dpi[:2])
    checks["export_dimensions_match_canvas"] = (
        png_size == [11934, 7956]
        and tiff_size == [11016, 7344]
    )

    svg_raw = paths["svg"].read_text(encoding="utf-8")
    svg_root = ElementTree.fromstring(svg_raw)
    svg_nodes = list(svg_root.iter())
    text_nodes = [node for node in svg_nodes if node.tag.endswith("text")]
    svg_text = ["".join(node.itertext()).strip() for node in text_nodes]
    checks["svg_editable_text"] = len(svg_text) >= 40
    checks["no_one_point_zero_tick"] = "1.0" not in svg_text
    checks["no_gray_explanatory_sentence"] = not any(
        len("".join(node.itertext()).strip()) > 120
        and any(token in node.attrib.get("style", "").lower() for token in ("#777", "#888", "#999", "gray", "grey"))
        for node in text_nodes
    )
    checks["all_panel_letters_in_svg"] = all(letter in svg_text for letter in "abcde") and "f" not in [text for text in svg_text if len(text) == 1]
    checks["a_b_titles_without_seconds"] = (
        "Single-source tracking" in svg_text
        and "Dual-source tracking" in svg_text
        and "Single-source tracking (67 s)" not in svg_text
        and "Dual-source tracking (15 s)" not in svg_text
    )
    checks["confidential_closeup_title_absent"] = (
        "Helicopter close-up" not in svg_text
        and "Field array and tracked helicopter" not in svg_text
    )
    legend_labels = ("T1 GPS", "T1 APCE", "T2 GPS", "T2 APCE", "Marginal width", "Array node", "Start", "End")
    checks["b_legend_contains_all_semantics"] = all(label in svg_text for label in legend_labels)
    checks["c_legend_contains_all_semantics"] = all(label in svg_text for label in ("Single (67 s)", f"Dual T1 ({dual_length} s)", f"Dual T2 ({dual_length} s)"))
    checks["endpoint_text_only_in_legends"] = (
        "Start" in svg_text and "End" in svg_text
        and "T1 start" not in svg_text and "T1 end" not in svg_text
        and "T2 start" not in svg_text and "T2 end" not in svg_text
    )

    # Typography and legend checks use both rendered SVG text/style and the
    # artist geometry captured by the figure source.
    typography = registry["typography"]
    checks["panel_label_pt == 26"] = int(typography["panel_label_pt"]) == 26 and sum(
        "font-size: 26px" in node.attrib.get("style", "") and "font-weight: 700" in node.attrib.get("style", "")
        for node in text_nodes if "".join(node.itertext()).strip() in set("abcde")
    ) >= 5
    title_texts = ("Single-source tracking", "Dual-source tracking", "Tracking error and uncertainty", "Node-target DOA reliability", "Array geometry and field deployment")
    checks["panel_title_pt == 17"] = int(typography["panel_title_pt"]) == 17 and all(
        any("font-size: 17px" in node.attrib.get("style", "") and "font-weight: 700" not in node.attrib.get("style", "") for node in text_nodes if "".join(node.itertext()).strip() == title)
        for title in title_texts
    )
    checks["panel_labels_moved_up"] = abs(float(registry["header_layout"]["label_y_offset_from_axes_top_fraction"]) - 0.018) < 1e-12
    checks["panel_titles_moved_down"] = abs(float(registry["header_layout"]["title_y_offset_from_axes_top_fraction"]) - 0.012) < 1e-12
    checks["d_legend_borderpad_restored"] = abs(float(registry["panel_d_legend_layout"]["borderpad_font_fraction"]) - 0.34) < 1e-12
    checks["d_legend_top_aligned"] = abs(float(registry["panel_d_legend_layout"]["anchor_axes_fraction"][1]) - 0.985) < 1e-12
    checks["d_node_rows_spread_1.15x"] = (
        abs(float(registry["panel_d_legend_layout"]["node_row_spacing"]) - 1.15) < 1e-12
        and abs(float(registry["reliability_semantics"]["node_row_spacing"]) - 1.15) < 1e-12
    )
    checks["d_legend_layout_preserves_dimensions"] = bool(registry["panel_d_legend_layout"]["canvas_and_axes_dimensions_unchanged"])
    checks["axis_label_pt == 15"] = int(typography["axis_label_pt"]) == 15
    checks["tick_pt == 13"] = int(typography["tick_pt"]) == 13
    checks["array_label_pt == 11"] = int(typography["array_label_pt"]) == 11
    legend_specs = registry.get("legend_specs", registry.get("layout_qa", {}).get("legend_records", {}))
    def frame_ok(name: str) -> bool:
        spec = legend_specs.get(name, {})
        return (
            bool(spec.get("frameon"))
            and rgba_is_white(spec.get("facecolor_rgba", []))
            and rgba_is_dark(spec.get("edgecolor_rgba", []))
            and 0.90 <= float(spec.get("framealpha", 0.0)) <= 0.96
        )
    def legend_group_for(label: str):
        for candidate in svg_nodes:
            if not candidate.tag.endswith("g") or not str(candidate.attrib.get("id", "")).startswith("legend"):
                continue
            if label in ["".join(node.itertext()).strip() for node in candidate.iter() if node.tag.endswith("text")]:
                return candidate
        return None
    def svg_frame_ok(label: str) -> bool:
        group = legend_group_for(label)
        if group is None:
            return False
        styles = [node.attrib.get("style", "").lower() for node in group.iter() if node.tag.endswith("path")]
        return any(
            "fill: #ffffff" in style
            and "opacity: 0.9" in style
            and "stroke: #222222" in style
            for style in styles
        )
    white_frame_styles = len(re.findall(r"fill:\s*#ffffff;\s*opacity:\s*0\.9[23456]", svg_raw.lower()))
    dark_frame_styles = len(re.findall(r"stroke:\s*#222222", svg_raw.lower()))
    checks["svg_legend_frames_visible"] = white_frame_styles >= 3 and dark_frame_styles >= 3
    checks["a_legend_box"] = frame_ok("a") and svg_frame_ok("GPS")
    checks["b_legend_box"] = frame_ok("b") and svg_frame_ok("T1 GPS")
    checks["a_legend_lower_left"] = (
        legend_specs.get("a", {}).get("loc") == "lower left"
        and bbox_inside([float(value) for value in legend_specs.get("a", {}).get("bbox_axes_fraction", [])], [0.0, 0.0, 1.0, 1.0], tolerance=2e-3)
        and float(legend_specs.get("a", {}).get("bbox_axes_fraction", [1.0, 1.0, 0.0, 0.0])[0]) < 0.5
        and float(legend_specs.get("a", {}).get("bbox_axes_fraction", [1.0, 1.0, 0.0, 0.0])[1]) < 0.5
    )
    checks["b_legend_inside_axes"] = (
        legend_specs.get("b", {}).get("loc") == "lower left"
        and bbox_inside([float(value) for value in legend_specs.get("b", {}).get("bbox_axes_fraction", [])], [0.0, 0.0, 1.0, 1.0], tolerance=2e-3)
    )
    checks["a_legend_visual_order"] = legend_specs.get("a", {}).get("labels") == ["GPS", "Marginal width", "APCE", "Array node", "Start", "End"]
    checks["b_legend_visual_order"] = legend_specs.get("b", {}).get("labels") == ["T1 GPS", "T2 GPS", "Marginal width", "T1 APCE", "T2 APCE", "Array node", "Start", "End"]
    panel_c_box = registry["layout_qa"]["panel_boxes"]["c"]
    panel_c_bounds = [float(panel_c_box["x0_fraction"]), float(panel_c_box["y0_fraction"]), float(panel_c_box["x1_fraction"]), float(panel_c_box["y1_fraction"])]
    checks["c_legend_inside_panel"] = (
        frame_ok("c")
        and legend_specs.get("c", {}).get("loc") == "upper left"
        and bbox_inside([float(value) for value in legend_specs.get("c", {}).get("bbox_fig_fraction", [])], panel_c_bounds, tolerance=2e-3)
        and all(0.0 <= float(value) <= 1.0 for value in legend_specs.get("c", {}).get("bbox_axes_fraction", [2.0, 2.0, -1.0, -1.0]))
    )

    array_spec = registry["array"]
    microphone_labels = [f"M{index}" for index in range(1, 20)]
    microphone_label_nodes = {
        label: [node for node in text_nodes if "".join(node.itertext()).strip() == label]
        for label in microphone_labels
    }
    checks["e_all_19_microphone_labels"] = (
        array_spec.get("microphone_labels") == microphone_labels
        and all(label in svg_text for label in microphone_labels)
        and all(
            any("font-size: 11px" in node.attrib.get("style", "") for node in nodes)
            for nodes in microphone_label_nodes.values()
        )
    )
    source_script_text = Path(registry.get("script", "")).read_text(encoding="utf-8") if registry.get("script") and Path(registry["script"]).is_file() else ""
    checks["e_marker_size_doubled"] = (
        int(array_spec.get("marker_size_v12", -1)) == 58
        and int(array_spec.get("marker_size", -1)) == 116
        and abs(float(array_spec.get("marker_size_ratio_vs_v12", 0.0)) - 2.0) < 1e-9
        and "ARRAY_MARKER_SIZE = 116" in source_script_text
        and "s=ARRAY_MARKER_SIZE" in source_script_text
    )
    label_layout = array_spec.get("label_layout", {})
    checks["e_rendered_label_boxes_do_not_overlap"] = (
        label_layout.get("all_labels_nonoverlapping") is True
        and label_layout.get("label_label_overlaps") == []
        and label_layout.get("label_marker_overlaps") == []
        and label_layout.get("labels_outside_axes") == []
        and len(label_layout.get("selected_offsets_pt", {})) == 19
        and "rendered-bbox collision audit" in label_layout.get("method", "")
    )

    image_integrity = registry["image_integrity"]
    try:
        checks["photo_mirror_hashes"] = (
            sha256(Path(image_integrity["panel_e"]["mirrored_file"])) == image_integrity["panel_e"]["raw_sha256"]
        )
    except (OSError, KeyError):
        checks["photo_mirror_hashes"] = False
    checks["panel_d_source_hash_matches"] = (
        sha256(Path(image_integrity["panel_d"]["source_file"])) == image_integrity["panel_d"]["source_sha256"]
    )

    pdf_info = subprocess.run(
        ["pdfinfo", str(paths["pdf"])], check=True, capture_output=True, text=True
    ).stdout
    checks["pdf_single_page"] = bool(re.search(r"^Pages:\s+1\s*$", pdf_info, flags=re.MULTILINE))
    extracted = subprocess.run(
        ["pdftotext", str(paths["pdf"]), "-"], check=True, capture_output=True, text=True
    ).stdout
    checks["pdf_text_extractable"] = all(title in extracted for title in ("Single-source tracking", "Dual-source tracking", "Node-target DOA reliability"))
    checks["confidential_closeup_absent_from_pdf"] = "Helicopter close-up" not in extracted
    checks["manual_visual_review_confirmed"] = bool(args.visual_review_confirmed)

    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "details": {
            "png_size_px": png_size,
            "tiff_size_px": tiff_size,
            "tiff_dpi": tiff_dpi,
            "grid_cell_pixel_variances": panel_variances,
            "svg_text_nodes": len(svg_text),
            "legend_bboxes_fig_fraction": {name: spec.get("bbox_fig_fraction") for name, spec in legend_specs.items()},
            "authoritative_metrics": authoritative_metrics,
            "authoritative_error": authoritative_error,
            "source_rows": len(source_rows),
            "historical_v30_reference": {
                "single_rmse_m": 39.49474662444076,
                "single_median_error_m": 35.703512310827286,
                "single_p90_error_m": 54.551986097342294,
                "interpretation": "reference only; v32 deliberately uses the isolated A6 single-source backend contract",
            },
        },
        "visual_review": {
            "performed_on_mirrored_png": bool(args.visual_review_confirmed),
            "five_panels_visible": checks["png_nonblank_all_grid_cells"],
            "legends_do_not_occlude_trajectory_lines": bool(args.visual_review_confirmed),
            "single_node_photo_clear": False,
            "panel_d_reliability_visible": bool(args.visual_review_confirmed),
            "confidential_closeup_absent": bool(args.visual_review_confirmed and checks["confidential_closeup_not_mirrored"]),
            "photographs_unframed": checks["photo_frames_absent"],
            "gray_explanatory_sentence_present": not checks["no_gray_explanatory_sentence"],
        },
    }
    output = root / f"{STEM}_qa.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
