# PCE–APCE

## 面向混合不确定动力学的成对累积预测证据

PCE 与 APCE 为动态同化提供统一的证据更新机制，面向候选模型、观测和潜在认知权重均存在不确定性的场景。候选动力学并行传播，通过与分析过程隔离的影子预测进行评分，再用累积预测证据组合。APCE 在此基础上加入熵感知的候选混合更新。

<p><a href="https://brianzhu1999.github.io/PCE-APCE/supplementary/"><strong>打开补充材料展示页</strong></a> · <a href="docs/supplementary/README.zh-CN.md">浏览发布文件</a> · <a href="docs/reproduction.md">复现实验</a> · <a href="CITATION.cff">引用软件</a></p>

仓库同时提供论文所用的基准实验设置，以及面向读者的 8 部补充视频和 8 个可交互 HTML 查看器。媒体目录说明每个案例的用途和评价设置，公开目录记录文件校验值与来源信息。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `pce_assimilation/` | PCE/APCE 证据更新、同化与集合分析 |
| `benchmarks/` | 经典系统、应用型 ODE 和高维动力学 |
| `viv_piv/` | 稀疏 VIV–PIV 尾流重建 |
| `acoustic_field_reconstruction/` | 实测三维声场重建 |
| `acoustic_array_tracking/` | 单源与双源声学阵列跟踪 |
| `docs/supplementary/` | 补充视频、交互查看器和公开媒体目录 |
| `tests/` | 方法与实验设置测试 |

补充合集用于解释代码和展示案例。全运行校准图谱包含 210 条开发记录，跨案例预测来源图谱包含 18 个案例的描述性汇总；这些页面保留记录值，用于探索，不能替代论文预先规定的评价。

## 安装

推荐 Python 3.11：

```bash
conda env create -f environment.yml
conda activate pce-apce
python -m pip install -e ".[dev]"
python -m pytest -q
```

需要 GPU 时，请安装适配目标机器的 PyTorch CUDA 版本；小型示例可在 CPU 上运行。

## 复现

每个基准提供模块入口，例如：

```bash
python -m benchmarks.classical_systems --case wave --method apce \
  --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu \
  --output results/applied_odes
```

完整命令、输入和案例设置见 [`docs/reproduction.md`](docs/reproduction.md)。实测数据案例的准备步骤见各目录内的 README。

## 数据与媒体

- VIV–PIV：[DOI 10.57745/HPA87O](https://doi.org/10.57745/HPA87O)
- MeshRIR：[DOI 10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817)
- 声学阵列跟踪：适配器读取经授权的本地测量数据副本，仓库不嵌入原始测量数据。

数据和输出路径通过命令行或 JSON 配置明确指定。生成结果写入用户指定目录；只有选入补充合集的媒体进入版本控制。查看 [`docs/supplementary/README.zh-CN.md`](docs/supplementary/README.zh-CN.md) 获取视频、查看器、完整性记录和本地打开说明。

## 引用

请同时引用配套论文和本软件发布版本。机器可读元数据见 [`CITATION.cff`](CITATION.cff)。

## 许可证

源代码采用 [MIT License](LICENSE)。补充媒体保留其来源和数据归属；本页不额外声明媒体许可证。
