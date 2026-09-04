import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BASE_DIR = Path(r"figures")
TRAJ_PATH = BASE_DIR / "figure3a_selected6_ode_trajectories_for_gpt_tidy.csv"
META_PATH = BASE_DIR / "figure3a_selected6_ode_metadata_for_gpt.csv"
OUT_DIR = BASE_DIR


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


BG = "#fbf8f1"
WALL = "#5a4642"

COLORS = {
    "Truth": "#b9d956",
    "PCE":   "#79bced",
    "APCE":  "#efb83d",
}

METHOD_ORDER = ["Truth", "PCE", "APCE"]
EQ_FONT_SIZE = 12.8


CASE_THEMES = {
    "chemical": {
        "bg": "#fbf7eb",
        "wall": "#5a4642",
        "box_top": "#fff7dc",
        "box_bottom": "#efe0b7",
    },
    "pk_infusion": {
        "bg": "#f6f9f8",
        "wall": "#475552",
        "box_top": "#effaf7",
        "box_bottom": "#d5e9e3",
    },
    "sir": {
        "bg": "#f7f6fb",
        "wall": "#50485b",
        "box_top": "#f6f1ff",
        "box_bottom": "#ddd4ec",
    },
    "pendulum": {
        "bg": "#fbf7f4",
        "wall": "#5b4a43",
        "box_top": "#fff2e8",
        "box_bottom": "#ead5c3",
    },
    "fhn": {
        "bg": "#f6f8fc",
        "wall": "#444f61",
        "box_top": "#edf4ff",
        "box_bottom": "#d2ddec",
    },
    "robertson": {
        "bg": "#f8faf3",
        "wall": "#4f5642",
        "box_top": "#f7fbdf",
        "box_bottom": "#dde8bd",
    },
}


CASE_CONFIGS = [
    {
        "case": "chemical",
        "title": "Chemical reaction",
        "output": "chemical_reaction_panel_template_v16",
        "variables": ("a", "b"),
        "mode": "two_state_time",
        "axis_labels": ("a", "b", "t"),
        "equations": [
            r"$\dot{a}=-2k(\alpha)a^2,\quad \dot{b}=k(\alpha)a^2$",
            r"$k(\alpha)=k_0+k_1\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
    {
        "case": "pk_infusion",
        "title": "PK infusion",
        "output": "pk_infusion_panel_template_v16",
        "variables": ("c",),
        "mode": "one_state_derivative_time",
        "axis_labels": ("c", r"$\dot c$", "t"),
        "equations": [
            r"$\dot{c}=q_0-k_ec+q_1c\Phi_{\mathrm{L}}^{-1}(\alpha)$",
            r"$+q_2\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
    {
        "case": "sir",
        "title": "SIR rumour",
        "output": "sir_rumour_panel_template_v16",
        "variables": ("s", "i", "r"),
        "mode": "three_state",
        "axis_labels": ("s", "i", "r"),
        "equations": [
            r"$\dot{i}=-is$",
            r"$\dot{s}=\beta is-\delta sr-\rho s-\eta sr\Phi_{\mathrm{L}}^{-1}(\alpha)$",
            r"$\dot{r}=\delta sr+\rho s+(1-\beta)is+\eta sr\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
    {
        "case": "pendulum",
        "title": "Forced pendulum",
        "output": "forced_pendulum_panel_template_v16",
        "variables": ("theta", "omega"),
        "mode": "two_state_time",
        "axis_labels": (r"$\theta$", r"$\omega$", "t"),
        "equations": [
            r"$\dot{\theta}=\omega$",
            r"$\dot{\omega}=-(g/l+s_\omega\Phi_{\mathrm{L}}^{-1}(\alpha))\sin\theta$",
            r"$-d\omega+A\cos\Omega t$",
        ],
    },
    {
        "case": "fhn",
        "title": "FitzHugh--Nagumo",
        "output": "fhn_panel_template_v16",
        "variables": ("v", "w"),
        "mode": "two_state_time",
        "axis_labels": ("v", "w", "t"),
        "equations": [
            r"$\dot{v}=v-v^3/3-w+I_0+s_I\Phi_{\mathrm{L}}^{-1}(\alpha)$",
            r"$\dot{w}=\epsilon(v+a-bw)$",
        ],
    },
    {
        "case": "robertson",
        "title": "Robertson kinetics",
        "output": "robertson_kinetics_panel_template_v16",
        "plot_slice": (105, 145),
        "variables": ("x", "y", "z"),
        "mode": "three_state",
        "axis_labels": ("x", "y", "z"),
        "equations": [
            r"$\dot{x}=-k_1x+k_3yz,\quad \dot{z}=k_2(\alpha)y^2$",
            r"$\dot{y}=k_1x-k_2(\alpha)y^2-k_3yz$",
            r"$k_2(\alpha)=3.0{\times}10^7+1.5{\times}10^7\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
]


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def lower_text(x):
    return str(x).strip().lower()


def case_rows(rows, case):
    return [r for r in rows if lower_text(r.get("case", "")) == lower_text(case)]


def load_case_data(traj_rows_all, config):
    rows = case_rows(traj_rows_all, config["case"])
    if not rows:
        raise RuntimeError(f"轨迹表中没有找到 case={config['case']} 的数据。")

    data = {method: {} for method in METHOD_ORDER}
    for method in METHOD_ORDER:
        for var in config["variables"]:
            rr = [
                r for r in rows
                if lower_text(r["method"]) == method.lower()
                and lower_text(r["variable"]) == lower_text(var)
            ]
            rr.sort(key=lambda x: int(float(x["step"])))
            if not rr:
                raise RuntimeError(f"缺少 {config['case']} 中 {method}-{var} 的轨迹数据。")
            data[method][var] = {
                "time": np.array([float(r["time"]) for r in rr]),
                "value": np.array([float(r["value"]) for r in rr]),
            }
    return data


def derivative(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 3:
        return np.zeros_like(y)
    return np.gradient(y, t)


def apply_plot_slice(config, x, y, z):
    """Apply a display-only continuous window when a long trajectory has a cleaner representative segment."""
    if "plot_slice" not in config:
        return x, y, z
    start, stop = config["plot_slice"]
    return x[start:stop], y[start:stop], z[start:stop]


def method_xyz(data, config, method):
    mode = config["mode"]
    variables = config["variables"]
    if mode == "two_state_time":
        x = data[method][variables[0]]["value"]
        y = data[method][variables[1]]["value"]
        z = data[method][variables[0]]["time"]
        return apply_plot_slice(config, x, y, z)
    if mode == "one_state_derivative_time":
        x = data[method][variables[0]]["value"]
        z = data[method][variables[0]]["time"]
        y = derivative(x, z)
        return apply_plot_slice(config, x, y, z)
    if mode == "three_state":
        x = data[method][variables[0]]["value"]
        y = data[method][variables[1]]["value"]
        z = data[method][variables[2]]["value"]
        return apply_plot_slice(config, x, y, z)
    raise ValueError(mode)


def finite_range(values, pad_frac=0.06, min_span=1e-6):
    arr = np.concatenate([np.asarray(v, dtype=float) for v in values])
    arr = arr[np.isfinite(arr)]
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(hi - lo, min_span)
    return lo - pad_frac * span, hi + pad_frac * span


def normalize_xyz(x, y, z, ranges):
    xmin, xmax, ymin, ymax, zmin, zmax = ranges
    return (
        (x - xmin) / max(xmax - xmin, 1e-12),
        (y - ymin) / max(ymax - ymin, 1e-12),
        (z - zmin) / max(zmax - zmin, 1e-12),
    )


def add_walls(ax, xmin, xmax, ymin, ymax, zmin, zmax, wall_color=WALL):
    floor = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
    ])
    left_wall = np.array([
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmin, ymin, zmax],
    ])
    back_wall = np.array([
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])
    for wall in [floor, left_wall, back_wall]:
        ax.add_collection3d(
            Poly3DCollection(
                [wall],
                facecolor=wall_color,
                edgecolor="none",
                alpha=0.98,
                zorder=1,
            )
        )


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def add_gradient_box_lines(fig, x, top, lines, width=0.66, fontsize=EQ_FONT_SIZE, box_top="#fff7dc", box_bottom="#efe0b7"):
    """Figure-coordinate gradient equation box with multiple aligned formula lines."""
    line_step = 0.048
    pad_top = 0.028
    pad_bottom = 0.030
    height = pad_top + pad_bottom + line_step * max(len(lines) - 1, 0)
    axg = fig.add_axes([x - width / 2, top - height, width, height], zorder=0)
    axg.set_axis_off()
    top_color = hex_to_rgb01(box_top)
    bottom_color = hex_to_rgb01(box_bottom)
    g = np.linspace(0, 1, 96)[:, None]
    rgb = (1 - g) * top_color + g * bottom_color
    img = np.repeat(rgb[:, None, :], 8, axis=1)
    axg.imshow(img, aspect="auto", origin="upper", extent=[0, 1, 0, 1])
    axg.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.05))
    first_y = 1.0 - pad_top / height
    ys = [first_y - i * (line_step / height) for i in range(len(lines))]
    for text, yy in zip(lines, ys):
        axg.text(
            0.50,
            yy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="Arial",
            color="black",
            transform=axg.transAxes,
            zorder=3,
        )


def draw_single_panel(config, traj_rows_all, letter=None):
    theme = CASE_THEMES.get(config["case"], CASE_THEMES["chemical"])
    data = load_case_data(traj_rows_all, config)

    xyz_by_method = {
        method: method_xyz(data, config, method)
        for method in METHOD_ORDER
    }
    xmin0, xmax0 = finite_range([xyz_by_method[m][0] for m in METHOD_ORDER], 0.07)
    ymin0, ymax0 = finite_range([xyz_by_method[m][1] for m in METHOD_ORDER], 0.07)
    zmin0, zmax0 = finite_range([xyz_by_method[m][2] for m in METHOD_ORDER], 0.07)
    original_ranges = (xmin0, xmax0, ymin0, ymax0, zmin0, zmax0)
    xyz_by_method = {
        method: normalize_xyz(*xyz_by_method[method], original_ranges)
        for method in METHOD_ORDER
    }
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    zmin, zmax = 0.0, 1.0

    fig = plt.figure(figsize=(4.35, 6.50), facecolor=BG)
    ax = fig.add_axes(
        [0.040, 0.425, 0.920, 0.475],
        projection="3d",
        computed_zorder=False,
    )
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.view_init(elev=15.0, azim=-68.0)
    try:
        ax.set_proj_type("persp", focal_length=1.15)
    except TypeError:
        ax.set_proj_type("persp")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((1.18, 1.00, 0.78))

    add_walls(ax, xmin, xmax, ymin, ymax, zmin, zmax, wall_color=theme["wall"])

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    origin = np.array([xmin, ymax, zmin])
    axis_end = {
        "x": np.array([0.92, 1.00, 0.00]),
        "y": np.array([0.00, 0.08, 0.00]),
        "z": np.array([0.00, 1.00, 0.90]),
    }
    axis_lw = 1.80
    head_scale = 0.1
    ax.quiver(origin[0], origin[1], origin[2],
              axis_end["x"][0] - origin[0], axis_end["x"][1] - origin[1], axis_end["x"][2] - origin[2],
              color="black", linewidth=axis_lw, arrow_length_ratio=head_scale, zorder=21)
    ax.quiver(origin[0], origin[1], origin[2],
              axis_end["y"][0] - origin[0], axis_end["y"][1] - origin[1], axis_end["y"][2] - origin[2],
              color="black", linewidth=axis_lw, arrow_length_ratio=head_scale, zorder=21)
    ax.quiver(origin[0], origin[1], origin[2],
              axis_end["z"][0] - origin[0], axis_end["z"][1] - origin[1], axis_end["z"][2] - origin[2],
              color="black", linewidth=axis_lw, arrow_length_ratio=head_scale, zorder=21)

    lx, ly, lz = config["axis_labels"]
    ax.text(xmax + 0.058 * dx, ymax - 0.045 * dy, zmin + 0.025 * dz, lx,
            fontsize=18, fontfamily="Arial", fontstyle="italic", color="black",
            ha="center", va="center", zorder=30)
    ax.text(xmin - 0.050 * dx, ymin - 0.085 * dy, zmin + 0.020 * dz, ly,
            fontsize=18, fontfamily="Arial", fontstyle="italic", color="black",
            ha="center", va="center", zorder=30)
    ax.text(xmin - 0.025 * dx, ymax + 0.025 * dy, zmax + 0.050 * dz, lz,
            fontsize=18, fontfamily="Arial", fontstyle="italic", color="black",
            ha="center", va="center", zorder=30)

    for method in METHOD_ORDER:
        x, y, z = xyz_by_method[method]
        c = COLORS[method]
        glow_lw = 4.6 if method != "APCE" else 2.8
        main_lw = 3.25 if method != "APCE" else 2.0
        z_order = {"Truth": 10, "PCE": 11, "APCE": 12}[method]
        ax.plot(x, y, z, color=c, lw=glow_lw, alpha=0.055, zorder=z_order)
        ax.plot(x, y, z, color=c, lw=main_lw, alpha=0.98, zorder=z_order + 1)
        xf, yf, zf = x[-1], y[-1], z[-1]
        for s, alpha in [(260, 0.05), (140, 0.11), (75, 0.24)]:
            ax.scatter([xf], [yf], [zf], s=s, c=c, alpha=alpha,
                       edgecolors="none", depthshade=False, zorder=24)
        ax.scatter([xf], [yf], [zf], s=24, c=c, edgecolors="white",
                   linewidths=0.5, depthshade=False, zorder=25)

    x0, y0, z0 = xyz_by_method["Truth"][0][0], xyz_by_method["Truth"][1][0], xyz_by_method["Truth"][2][0]
    ax.scatter([x0], [y0], [z0], s=24, c="white", edgecolors="black",
               linewidths=0.55, depthshade=False, zorder=25)

    if letter is not None:
        fig.text(0.15, 0.915, letter, fontsize=26, fontweight="bold",
                 fontfamily="Arial", ha="left", va="top", color="black")

    fig.text(0.50, 0.46, config["title"], ha="center", va="center",
             fontsize=18.5, fontfamily="Arial", color="black")

    eqs = config["equations"]
    formula_top = 0.405
    add_gradient_box_lines(
        fig,
        0.50,
        formula_top,
        eqs,
        width=0.66,
        box_top=theme["box_top"],
        box_bottom=theme["box_bottom"],
    )

    return fig


def save_single_panels(traj_rows_all):
    output_pngs = []
    for idx, config in enumerate(CASE_CONFIGS):
        letter = "a" if idx == 0 else None
        fig = draw_single_panel(config, traj_rows_all, letter=letter)
        base = OUT_DIR / config["output"]
        for ext in ("png", "pdf", "svg"):
            fig.savefig(base.with_suffix(f".{ext}"), facecolor=BG, bbox_inches=None)
        plt.close(fig)
        output_pngs.append(base.with_suffix(".png"))
    return output_pngs


def trim_image_xonly(img, bg_rgb=np.array([251, 248, 241]) / 255.0, tol=0.035):
    """Trim only horizontal near-background margins; preserve full height for exact top alignment."""
    arr = img[..., :3]
    mask = np.max(np.abs(arr - bg_rgb), axis=2) > tol
    if not np.any(mask):
        return img
    _, xs = np.where(mask)
    pad_x = 10
    x0 = max(0, xs.min() - pad_x)
    x1 = min(img.shape[1], xs.max() + pad_x + 1)
    return img[:, x0:x1, :]


def stitch_row(pngs):
    import matplotlib.image as mpimg

    images = [trim_image_xonly(mpimg.imread(p)) for p in pngs]
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_h = max(heights)
    gap_px = 18
    legend_h = 150
    total_w = sum(widths) + gap_px * (len(widths) - 1)
    canvas_h = max_h + legend_h

    # Keep the already approved single-panel design untouched; stitch the six
    # panel rasters with no extra decoration.
    fig_w = total_w / 600
    fig_h = canvas_h / 600
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600, facecolor=BG)
    x = 0
    for img, w, h in zip(images, widths, heights):
        y0 = (max_h - h) / canvas_h
        ax = fig.add_axes([x / total_w, y0, w / total_w, h / canvas_h])
        ax.imshow(img)
        ax.set_axis_off()
        x += w + gap_px

    legend_handles = [
        Line2D([0], [0], color=COLORS[m], lw=3.25 if m != "APCE" else 2.0, marker="o", markersize=7.4,
               markerfacecolor="white", markeredgecolor=COLORS[m],
               markeredgewidth=1.20, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.898),
        ncol=3,
        frameon=False,
        prop={"family": "Arial", "size": 18.0},
        handlelength=1.65,
        handletextpad=0.42,
        columnspacing=1.05,
    )
    row_base = OUT_DIR / "figure3a_selected6_ode_panels_row_template_v16"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(row_base.with_suffix(f".{ext}"), facecolor=BG, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return row_base


def main():
    traj_rows_all = read_csv_rows(TRAJ_PATH)
    read_csv_rows(META_PATH)  # keep the same input contract as the chemical template
    pngs = save_single_panels(traj_rows_all)
    row_base = stitch_row(pngs)
    print("Generated individual panels:")
    for p in pngs:
        print(" ", p)
    print("Generated row panel:")
    print(" ", row_base.with_suffix(".png"))
    print(" ", row_base.with_suffix(".pdf"))
    print(" ", row_base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
