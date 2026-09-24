# Supplementary movies and interactive experiments

This folder accompanies the **PCE–APCE** method for dynamic assimilation under sparse observations. The collection is designed as a reader-facing companion to the paper: each movie shows a complete reconstruction or forecast sequence, while each interactive component exposes the observations, state estimates and uncertainty used in a representative case.

**Project repository:** <https://github.com/BrianZhu1999/PCE-APCE>

## Contents

- `index.html` — bilingual catalogue (English is the default; use the 中文 / English control to switch and the choice is remembered locally).
- `Supplementary_Movie_*.mp4` — eight numbered H.264 movies; Movie 7 includes isosurface and original slice versions.
- `*_interactive.html` — interactive views for sensor blackout, VIV–PIV, MeshRIR and acoustic-array tracking.
- `catalog.json` — machine-readable titles, routes, source notes and verification fields.
- `code/` — deterministic page builder and the local CSS/JavaScript sources.

The eight numbered movies accompany the corresponding Results sections. Movie 6 and the VIV–PIV interactive component use the paper-matched five-regime configuration: 751 spatial measurement locations and 1,502 scalar velocity observations.

### MeshRIR: isosurfaces and slices

The isosurface movie uses the collection's white background, Arial typography, yellow case label and three-column layout. Its title, time display and coordinate labels match the original MeshRIR movie.

Movie 7 and the [3D viewer](MeshRIR_3D_interactive.html) open in **Isosurfaces** mode. Switch to **Slices** to inspect the same recorded time. The movie switch retains playback position and pause state; the interactive switch also retains rotation and zoom. Separate video downloads are available on the Movie 7 card. The [original standalone slice viewer](MeshRIR_3D_slices_legacy.html) is preserved.

All 1,024 samples from 0 to 63.9375 ms are retained. The three panels show all 64 interior measurements, APCE reconstruction and measured reference on the 21 × 21 × 9 grid. Reconstruction also uses 128 boundary measurements. Isosurfaces use fixed pressure levels ±0.001, ±0.002 and ±0.004 in the data's arbitrary units; observed pressure and slice colours use a fixed ±0.02 scale with saturation tips. Surfaces are extracted from the original grid without smoothing pressure or geometry. Positive and negative surfaces can be toggled independently, and **Paper view** restores the camera used in the figure.

For offline use, keep `MeshRIR_3D_interactive.html` together with the **entire `meshrir_isosurface_data/` folder**. Surface files load by time segment; playback retains every recorded sample and may slow while loading on a slow device or connection. The archived slice viewer remains a single HTML file. Movie playback is 24 samples per second, with the same two-second title and one-second final hold in both versions.

## Movies and corresponding results

| Movie | Subject | Paper correspondence |
|---|---|---|
| 1 | Three Classical Uncertain Equations | Results 2.2; Fig. 2 |
| 2 | Five ODE Systems | Results 2.3; Fig. 3 |
| 3 | Lorenz–96 Dynamics | Results 2.4; Fig. 4 |
| 4 | Kuramoto–Sivashinsky Equation | Results 2.4; Fig. 4 |
| 5 | Kolmogorov Turbulent Flow | Results 2.4; Fig. 4 |
| 6 | VIV–PIV Experiment | Results 2.5; Fig. 5 |
| 7 | MeshRIR 3D Acoustic Field | Results 2.5; Extended Data Fig. 1 |
| 8 | Acoustic Source Localization and Trajectory Tracking | Results 2.5; Extended Data Fig. 2 and Supplementary Note 17 |

Movie 7 follows reconstruction over the observation window, 0–63.9375 ms. The 65-ms observation-free forecast illustrated in Fig. 1d is a separate view.

## Interactive visualizations

The sensor/blackout, VIV–PIV, MeshRIR and source-tracking viewers expose the corresponding observations and reconstructed states. The APCE inspector shows Lorenz–96 and Kuramoto–Sivashinsky candidate weights, normalized entropy, parameter estimates and one state coordinate. Its time control opens at the first valid posterior; unavailable initial entries are displayed as missing values.

For offline VIV–PIV viewing, keep `VIV_PIV_5regimes_interactive.html` with its `viv_piv_data/` folder. The viewer loads one regime at a time and shows loading progress. This preserves the original displayed fields and spectra while reducing the first-page transfer.

## Rebuild locally

From this directory, after placing the media files alongside `index.html`:

```bash
python code/build_collection_portal.py
```

The builder reads `catalog.json` (falling back to the legacy `collection_manifest.json` when needed) and writes a self-contained `index.html`. Scientific media are copied from the verified release records; the builder does not alter MP4 or component data.

## 中文说明

本目录是 PCE–APCE 动态同化方法的补充材料入口。视频展示完整的场重建和中断后预测过程，交互组件用于查看观测布局、状态估计和不确定性。页面默认英文，可用右上角按钮切换中文，选择会保存在浏览器中。

目录编号与论文正文中的引用一致。视频6与VIV–PIV交互组件均使用现稿五工况配置：751个空间观测点、1502个速度分量观测。

## Citation

Please cite the accompanying manuscript and the PCE–APCE repository when reusing these materials. See the repository's `CITATION.cff` for machine-readable metadata.
