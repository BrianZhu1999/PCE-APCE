# Supplementary movies and interactive experiments

This folder accompanies the **PCE–APCE** method for dynamic assimilation under sparse observations. The collection is designed as a reader-facing companion to the paper: each movie shows a complete reconstruction or forecast sequence, while each interactive component exposes the observations, state estimates and uncertainty used in a representative case.

**Project repository:** <https://github.com/BrianZhu1999/PCE-APCE>

## Contents

- `index.html` — bilingual catalogue (English is the default; use the 中文 / English control to switch and the choice is remembered locally).
- `Supplementary_Movie_*.mp4` — eight H.264 supplementary videos.
- `*_interactive.html` — interactive views for sensor blackout, VIV–PIV, MeshRIR and acoustic-array tracking.
- `catalog.json` — machine-readable titles, routes, source notes and verification fields.
- `code/` — deterministic page builder and the local CSS/JavaScript sources.

The catalogue keeps the numbered sequence used in the manuscript. Movie 6 is the archived VIV–PIV movie supplied with the earlier five-regime presentation; the interactive VIV–PIV component is the current 751-observation view. These are labelled separately so the two records are not conflated.

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
