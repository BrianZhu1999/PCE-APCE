from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MM_PER_INCH = 25.4


@dataclass(frozen=True)
class BoxMM:
    """A rectangle in millimetres, using matplotlib's bottom-left origin."""

    x: float
    y: float
    w: float
    h: float

    def scaled(self, sx: float, sy: float) -> "BoxMM":
        return BoxMM(self.x * sx, self.y * sy, self.w * sx, self.h * sy)

    def inches(self) -> tuple[float, float, float, float]:
        return (self.x / MM_PER_INCH, self.y / MM_PER_INCH, self.w / MM_PER_INCH, self.h / MM_PER_INCH)

    def normalized(self, fig_w_mm: float, fig_h_mm: float) -> tuple[float, float, float, float]:
        return (self.x / fig_w_mm, self.y / fig_h_mm, self.w / fig_w_mm, self.h / fig_h_mm)


@dataclass(frozen=True)
class Figure2Layout:
    """Coupled Figure 2 layout.

    The default 183 mm x 145 mm canvas is a two-column Nature-style composite.
    All panel boxes are defined once in a baseline mm coordinate system and
    then scaled when width or height changes.
    """

    width_mm: float = 183.0
    height_mm: float = 145.0

    base_width_mm: float = 183.0
    base_height_mm: float = 145.0

    def scale_x(self) -> float:
        return self.width_mm / self.base_width_mm

    def scale_y(self) -> float:
        return self.height_mm / self.base_height_mm

    def figsize_in(self) -> tuple[float, float]:
        return (self.width_mm / MM_PER_INCH, self.height_mm / MM_PER_INCH)

    def font_scale(self) -> float:
        return min(self.scale_x(), self.scale_y())

    def panel_boxes_mm(self) -> dict[str, BoxMM]:
        sx, sy = self.scale_x(), self.scale_y()
        base = {
            "a": BoxMM(4.0, 74.0, 50.0, 66.0),
            "c": BoxMM(59.0, 74.0, 72.0, 66.0),
            "d": BoxMM(136.0, 74.0, 43.0, 66.0),
            "e": BoxMM(4.0, 42.0, 55.0, 24.0),
            "f": BoxMM(64.0, 42.0, 55.0, 24.0),
            "g": BoxMM(124.0, 42.0, 55.0, 24.0),
            "h": BoxMM(4.0, 6.0, 53.0, 28.0),
            "i": BoxMM(62.0, 6.0, 49.0, 28.0),
            "j": BoxMM(116.0, 6.0, 29.0, 28.0),
            "k": BoxMM(150.0, 6.0, 29.0, 28.0),
        }
        return {key: box.scaled(sx, sy) for key, box in base.items()}

    def panel_boxes_norm(self) -> dict[str, tuple[float, float, float, float]]:
        return {key: box.normalized(self.width_mm, self.height_mm) for key, box in self.panel_boxes_mm().items()}

    def style(self) -> dict[str, float | str | list[str]]:
        fs = self.font_scale()
        return {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.0 * fs,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "axes.linewidth": 0.50 * fs,
            "xtick.major.width": 0.45 * fs,
            "ytick.major.width": 0.45 * fs,
            "xtick.major.size": 1.6 * fs,
            "ytick.major.size": 1.6 * fs,
            "legend.frameon": False,
            "mathtext.default": "it",
        }

    def text_sizes_pt(self) -> dict[str, float]:
        fs = self.font_scale()
        return {
            "panel_letter": 8.0 * fs,
            "panel_title": 7.0 * fs,
            "subpanel_title": 5.6 * fs,
            "axis_label": 5.6 * fs,
            "tick_label": 4.8 * fs,
            "legend": 5.8 * fs,
            "equation": 6.0 * fs,
            "micro_annotation": 4.8 * fs,
        }

    def stroke_widths_pt(self) -> dict[str, float]:
        fs = self.font_scale()
        return {
            "group_rule": 0.45 * fs,
            "axis_spine": 0.50 * fs,
            "thin_line": 0.55 * fs,
            "data_line": 0.75 * fs,
            "truth_line": 0.90 * fs,
            "apce_line": 0.95 * fs,
            "bar_edge": 0.35 * fs,
            "errorbar": 0.45 * fs,
            "highlight_box": 0.65 * fs,
        }


def subdivide_box(
    box: BoxMM,
    rows: int,
    cols: int,
    pad_x_mm: float = 3.0,
    pad_y_mm: float = 3.0,
    gap_x_mm: float = 2.0,
    gap_y_mm: float = 2.0,
) -> list[BoxMM]:
    """Subdivide a panel box into row-major child boxes in mm."""

    inner_x = box.x + pad_x_mm
    inner_y = box.y + pad_y_mm
    inner_w = box.w - 2 * pad_x_mm
    inner_h = box.h - 2 * pad_y_mm
    cell_w = (inner_w - gap_x_mm * (cols - 1)) / cols
    cell_h = (inner_h - gap_y_mm * (rows - 1)) / rows
    cells: list[BoxMM] = []
    for r in range(rows):
        for c in range(cols):
            x = inner_x + c * (cell_w + gap_x_mm)
            y = inner_y + (rows - 1 - r) * (cell_h + gap_y_mm)
            cells.append(BoxMM(x, y, cell_w, cell_h))
    return cells


def child_layout_mm(layout: Figure2Layout) -> dict[str, list[BoxMM]]:
    """Default child-slot subdivision for the Figure 2 master frame."""

    boxes = layout.panel_boxes_mm()
    return {
        "a": subdivide_box(boxes["a"], 3, 2, pad_x_mm=3.5, pad_y_mm=5.5, gap_x_mm=2.5, gap_y_mm=4.0),
        "c": subdivide_box(boxes["c"], 3, 3, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.0, gap_y_mm=2.0),
        "d": subdivide_box(boxes["d"], 3, 2, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.0, gap_y_mm=2.0),
        "e": subdivide_box(boxes["e"], 1, 3, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.5, gap_y_mm=0.0),
        "f": subdivide_box(boxes["f"], 1, 3, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.5, gap_y_mm=0.0),
        "g": subdivide_box(boxes["g"], 1, 3, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.5, gap_y_mm=0.0),
        "h": subdivide_box(boxes["h"], 1, 3, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=2.5, gap_y_mm=0.0),
        "i": subdivide_box(boxes["i"], 1, 2, pad_x_mm=3.0, pad_y_mm=4.0, gap_x_mm=3.0, gap_y_mm=0.0),
        "j": subdivide_box(boxes["j"], 1, 1, pad_x_mm=3.0, pad_y_mm=4.0),
        "k": subdivide_box(boxes["k"], 1, 1, pad_x_mm=3.0, pad_y_mm=4.0),
    }


def format_panel_table(layout: Figure2Layout, letters: Iterable[str] | None = None) -> str:
    boxes = layout.panel_boxes_mm()
    letters = list(letters or boxes.keys())
    lines = [
        "| panel | x mm | y mm | w mm | h mm | w in | h in | normalized `(x,y,w,h)` |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in letters:
        box = boxes[key]
        norm = box.normalized(layout.width_mm, layout.height_mm)
        lines.append(
            f"| {key} | {box.x:.1f} | {box.y:.1f} | {box.w:.1f} | {box.h:.1f} | "
            f"{box.w / MM_PER_INCH:.2f} | {box.h / MM_PER_INCH:.2f} | "
            f"`({norm[0]:.4f}, {norm[1]:.4f}, {norm[2]:.4f}, {norm[3]:.4f})` |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    default_layout = Figure2Layout()
    print(f"Figure size: {default_layout.width_mm:.1f} mm x {default_layout.height_mm:.1f} mm")
    print(f"Figure size: {default_layout.figsize_in()[0]:.3f} in x {default_layout.figsize_in()[1]:.3f} in")
    print(format_panel_table(default_layout))
