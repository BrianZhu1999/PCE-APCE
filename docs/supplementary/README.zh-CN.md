# 补充视频与交互查看器

本合集配套 PCE–APCE 论文，提供 8 部连续编号的视频和 8 个浏览器查看器。编号、首页和下载文件保持一致，便于论文引用。

## 视频

| 编号 | 标题 | 内容 |
| --- | --- | --- |
| 1 | **三个经典不确定方程** | 三个经典不确定系统中的重建与预测。 |
| 2 | **五个 ODE 系统** | 五个应用型常微分方程设置下的统一比较。 |
| 3 | **Lorenz–96 动力学** | 混沌基准中的稀疏观测重建与预测演化。 |
| 4 | **Kuramoto–Sivashinsky 方程** | 参考场、APCE 重建及观测中断后的预测。 |
| 5 | **Kolmogorov 湍流** | 速度场重建与周期谱涡量的中断后演化。 |
| 6 | **VIV–PIV 实验** | 五种尾流观测设置；视频使用 745 个有效空间 PIV 点（1,490 个标量观测），配套查看器使用正式的 751 点（1,502 个标量）x40y20 设置。 |
| 7 | **MeshRIR 三维声场** | 实测三维声场重建。 |
| 8 | **声源定位与轨迹跟踪** | 稀疏声学阵列下的单源、双源和三源轨迹跟踪。 |

所有视频均为 1920 × 1080、24 fps、H.264、yuv420p，并保留约 2 秒封面。可打开 [`index.html`](index.html) 播放。

## 查看器

### 案例查看器

- [`MeshRIR_3D_interactive.html`](MeshRIR_3D_interactive.html)：旋转查看实测三维声场。
- [`Baoding_Tracking_3D_interactive.html`](Baoding_Tracking_3D_interactive.html)：切换单源、双源和三源轨迹。
- [`Supplementary_Sensor_Blackout_Explorer.html`](Supplementary_Sensor_Blackout_Explorer.html)：查看 KSE、Lorenz–96 和 Kolmogorov 的稀疏观测与中断示例。
- [`VIV_PIV_5regimes_interactive.html`](VIV_PIV_5regimes_interactive.html)：查看正式自适应 x40y20 五工况 VIV–PIV 设置，使用 751 个有效空间点和论文色阶。

### 方法与诊断查看器

- [`Supplementary_APCE_Inspector.html`](Supplementary_APCE_Inspector.html)：查看候选权重、归一化熵和 APCE α 估计。
- [`Supplementary_FullRun_Calibration_Atlas.html`](Supplementary_FullRun_Calibration_Atlas.html)：按案例和样本集浏览 210 条开发记录。
- [`Supplementary_Runtime_Pareto.html`](Supplementary_Runtime_Pareto.html)：查看匹配样本集中的案例内运行时间中位数与预测 nRMSE。
- [`Supplementary_CrossCase_ForecastSource_Atlas.html`](Supplementary_CrossCase_ForecastSource_Atlas.html)：探索 18 个案例的描述性预测来源汇总，保留正负值。

查看器将显示数据和脚本嵌入 HTML，可直接本地打开。若要保留图谱页面的 CSV 下载按钮和来源链接，请下载完整发布压缩包。

## 完整性与来源

`catalog.json` 记录公开文件、SHA-256 校验值、源脚本记录和发布版本。详细生成记录保存在研究归档中。实验数据处理以及视频、查看器生成均使用 Super-Server 上的权威文件；公开页面只包含选定的显示数据和来源说明，不包含原始数据集。

引用时请同时引用软件发布版本和配套论文。上游实测数据请引用 VIV–PIV DOI [10.57745/HPA87O](https://doi.org/10.57745/HPA87O) 及 MeshRIR DOI [10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817)。
