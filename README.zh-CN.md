# PCE–APCE

**用于动力学推断的预测历史分离方法**

本仓库提供论文 *Observation feedback can conceal competing dynamics* 的算法实现与基准实验。PCE与APCE分别维护观测校正的状态分析历史和候选动力学的预测评分历史，并通过预测权重连接两者；APCE进一步自适应调整证据累积。

[补充视频与交互展示](https://brianzhu1999.github.io/PCE-APCE/supplementary/) · [实验复现说明](docs/reproduction.md) · [引用信息](CITATION.cff)

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `pce_assimilation/` | 集合分析与预测权重更新 |
| `benchmarks/` | 非线性系统与高维动力学 |
| `viv_piv/` | 实测圆柱尾流重建与预测 |
| `acoustic_field_reconstruction/` | 三维声场重建 |
| `acoustic_array_tracking/` | 单声源及双声源跟踪 |
| `three_source_tracking/` | 三声源声学定位与跟踪 |
| `reproducibility/` | 反馈实验和统计结果复算 |
| `docs/supplementary/` | 8部补充视频和5个交互查看器 |
| `tests/` | 方法与实验设置测试 |

## 安装

推荐Python 3.11。

```bash
conda env create -f environment.yml
conda activate pce-apce
python -m pip install -e ".[dev]"
python -m pytest -q
```

GPU运行请安装匹配CUDA环境的PyTorch版本。

## 运行示例

```bash
python -m benchmarks.classical_systems --case wave --method apce --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu --output results/applied_odes
```

完整运行命令见[复现说明](docs/reproduction.md)及各实验目录的README。

## 数据来源

- [VIV–PIV实测尾流](https://doi.org/10.57745/HPA87O)
- [MeshRIR声场测量](https://doi.org/10.5281/zenodo.5002817)
- 声源跟踪：实验配置指定的本地阵列测量输入

## 引用与许可

引用信息见[CITATION.cff](CITATION.cff)；源代码采用[MIT许可证](LICENSE)。
