from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC_SVG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6.svg"
OUT_SVG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl.svg"
OUT_HTML = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_render.html"
OUT_PNG = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl.png"

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")

TARGET_W = 11532
TARGET_H = 4125


def main() -> None:
    text = SRC_SVG.read_text(encoding="utf-8")
    replacements = {
        ">a</text>": ">g</text>",
        ">b</text>": ">h</text>",
        ">c</text>": ">i</text>",
        ">d</text>": ">j</text>",
        ">e</text>": ">k</text>",
        ">f</text>": ">l</text>",
    }
    # Replace only the final six bold panel-label text nodes.
    for old, new in replacements.items():
        idx = text.rfind(old)
        if idx < 0:
            raise RuntimeError(f"Could not find panel label token {old!r}")
        text = text[:idx] + new + text[idx + len(old) :]
    # Move the two shared field colorbar labels ($u$) closer to the bars.
    # These are the h/k field colorbar labels in the final Figure 4 ordering.
    text = text.replace("translate(334.028 218.570434)", "translate(334.028 204.570434)")
    text = text.replace("translate(334.028 452.570434)", "translate(334.028 438.570434)")
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
