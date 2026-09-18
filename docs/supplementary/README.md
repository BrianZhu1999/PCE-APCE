# Supplementary movies and interactive experiments

This folder accompanies the **PCE–APCE** method for dynamic assimilation under sparse observations. The collection is designed as a reader-facing companion to the paper: each movie shows a complete reconstruction or forecast sequence, while each interactive component exposes the observations, state estimates and uncertainty used in a representative case.

**Project repository:** <https://github.com/BrianZhu1999/PCE-APCE>

## Contents

- `index.html` — bilingual catalogue (English is the default; use the 中文 / English control to switch and the choice is remembered locally).
- `Supplementary_Movie_*.mp4` — eight numbered H.264 movies; Movie 7 includes isosurface and original slice versions.
- `*_interactive.html` — interactive views for sensor blackout, VIV–PIV, MeshRIR and acoustic-array tracking.
- `catalog.json` — machine-readable titles, routes, source notes and verification fields.
- `code/` — deterministic page builder and the local CSS/JavaScript sources.

The catalogue keeps the numbered sequence used in the manuscript. Movie 6 is the archived VIV–PIV movie supplied with the earlier five-regime presentation; the interactive VIV–PIV component is the current 751-observation view. These are labelled separately so the two records are not conflated.

### MeshRIR: isosurfaces and slices

Movie 7 and the [3D viewer](MeshRIR_3D_interactive.html) open in **Isosurfaces** mode. Switch to **Slices** to inspect the same recorded time. The movie switch retains playback position and pause state; the interactive switch also retains rotation and zoom. Separate video downloads are available on the Movie 7 card. The [original standalone slice viewer](MeshRIR_3D_slices_legacy.html) is preserved.

All 1,024 samples from 0 to 63.9375 ms are retained. The three panels show all 64 interior measurements, APCE reconstruction and measured reference on the 21 × 21 × 9 grid. Reconstruction also uses 128 boundary measurements. Isosurfaces use fixed pressure levels ±0.001, ±0.002 and ±0.004 in the data's arbitrary units; observed pressure and slice colours use a fixed ±0.02 scale with saturation tips. Surfaces are extracted from the original grid without smoothing pressure or geometry. Positive and negative surfaces can be toggled independently, and **Paper view** restores the camera used in the figure.

For offline use, keep `MeshRIR_3D_interactive.html` together with the **entire `meshrir_isosurface_data/` folder**. Surface files load by time segment; playback retains every recorded sample and may slow while loading on a slow device or connection. The archived slice viewer remains a single HTML file. Movie playback is 24 samples per second, with the same two-second title and one-second final hold in both versions.

## Rebuild locally

From this directory, after placing the media files alongside `index.html`:

```bash
python code/build_collection_portal.py
```

The builder reads `catalog.json` (falling back to the legacy `collection_manifest.json` when needed) and writes a self-contained `index.html`. Scientific media are copied from the verified release records; the builder does not alter MP4 or component data.

## 中文说明

本目录是 PCE–APCE 动态同化方法的补充材料入口。视频展示完整的场重建和中断后预测过程，交互组件用于查看观测布局、状态估计和不确定性。页面默认英文，可用右上角按钮切换中文，选择会保存在浏览器中。

目录中的编号与论文保持一致。第 6 部补充视频保留早期五工况 VIV–PIV 展示，VIV–PIV 交互组件对应当前 751 个观测点的版本，页面已分别标注。

## Citation

Please cite the accompanying manuscript and the PCE–APCE repository when reusing these materials. See the repository's `CITATION.cff` for machine-readable metadata.
