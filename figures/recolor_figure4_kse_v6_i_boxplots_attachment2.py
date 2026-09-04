from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC_SVG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08.svg"
OUT_SVG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v7_i_palette_attachment2.svg"
OUT_HTML = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v7_i_palette_attachment2_render.html"
OUT_PNG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v7_i_palette_attachment2_chrome.png"

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")

TARGET_W = 11532
TARGET_H = 4125

COLOR_REPLACEMENTS = {
    "#9b59b6": "#fe9550",
    "#f39c12": "#c697e7",
    "#1f77b4": "#f37576",
    "#2ecc71": "#8eb8f2",
}


def main() -> None:
    text = SRC_SVG.read_text(encoding="utf-8")
    for old, new in COLOR_REPLACEMENTS.items():
        text = text.replace(old, new)
    OUT_SVG.write_text(text, encoding="utf-8")

    OUT_HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;padding:0;background:white;overflow:hidden;}"
        f"img{{display:block;width:{TARGET_W}px;height:{TARGET_H}px;}}</style></head>"
        f"<body><img src='{OUT_SVG.resolve().as_uri()}'></body></html>",
        encoding="utf-8",
    )
    if not CHROME.exists():
        raise FileNotFoundError(CHROME)
    subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={TARGET_W},{TARGET_H}",
            f"--screenshot={OUT_PNG}",
            OUT_HTML.resolve().as_uri(),
        ],
        check=True,
    )
    print(OUT_SVG.resolve())
    print(OUT_PNG.resolve())


if __name__ == "__main__":
    main()
