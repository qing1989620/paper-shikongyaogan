# 数据集与 I10 小实验手册

本项目使用同一区域、不同时刻的 Landsat 精细影像与 MODIS 粗影像。当前主队列和 I10 小实验使用 LGC。CIA 用于后续跨区和域内对照。以下均为项目冻结的历史开发划分。

| 数据 | 训练块 | 验证块 | 历史测试块 | 当前用途 |
| --- | ---: | ---: | ---: | --- |
| LGC | 1,200 | 120 | 120 | 主队列使用训练及验证。小实验使用验证。 |
| CIA | 294 | 126 | 168 | 已可还原，等待独立协议下的评估。 |

## 文件结构和张量

- 切片路径是 work/data/patches_lgc/{train,val,test} 和 work/data/patches_cia/{train,val,test}。每个 NPZ 内有 f0、f1、f3、c0、c1、c3 六个数组，均为 256×256×6 的 float16。前两维为空间，末维为波段。
- f0、f3 为前后精细参考，c0、c1、c3 为前锚、目标日和后锚粗观测。f1 是目标日精细真值，仅用于监督和计算误差。推理输入来自另外五个数组和两个日期间隔。
- 原始整数反射率裁剪到 0–10,000，再缩放到 [−1,1]。评价还原至 [0,1]，RMSE 对每块的全部像元及六个波段计算后求块均值。粗观测已处在共同网格，仍保留原有的粗尺度观测含义。
- LGC 验证三元组是 2005-02-14、2005-03-02、2005-04-03。历史测试目标日是 2005-01-13，以往已用于评估。日期、原始坐标、字节数和 SHA-256 见 [切片清单](audit/dataset_manifest.csv) 与 [LGC 冻结清单](../work/configs/screening_2026/manifest_lgc_frozen.csv)。

## 数据来源和取得方式

两套数据由 Emelyanova、McVicar、Van Niel、Li 和 Van Dijk 发布于 CSIRO Data Access Portal。数据条目标注 CSIRO Data Licence。由每位使用者按官方条款自行获取原始影像并保留原出处。协作包保留文件级清单、重建脚本和本项目的模型权重。影像切片由使用者在本机重建。

- [LGC 官方集合](https://data.csiro.au/collection/csiro:5847) · [DOI](https://doi.org/10.4225/08/514B80FD805F4) · [数据条目](https://researchdata.edu.au/landsat-modis-lower-study-site/444987)
- [CIA 官方集合](https://data.csiro.au/collection/csiro:5846) · [DOI](https://doi.org/10.4225/08/514B7FD1C798C) · [数据条目](https://researchdata.edu.au/landsat-modis-coleambally-irrigation-area/444986)
- [CSIRO Data Licence 官方条款](https://data.csiro.au/dap/ws/v2/licences/1061)

下面的命令从 CSIRO 官方服务只取 LGC 验证所用的六个主影像文件。已有个人保存的官方文件时，可将其置于 work/data/raw/LGC/Landsat 和 work/data/raw/LGC/MODIS，文件夹层级见 [原始主文件清单](audit/raw_file_integrity.csv)。CIA 的两个传感器目录放在 work/data/raw_cia/CIA 下。

## 在 Windows 运行

从协作 ZIP 解压后的根目录运行，需有 Python 3.12、Git 和适配 PyTorch CUDA 12.1 的 NVIDIA 驱动。CPU 可以运行小样本检查。建议先确认 torch.cuda.is_available() 的返回值。

~~~powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& $py -m pip install -r work/configs/i10_probe_requirements.txt
& $py -B -X utf8 scripts/restore_upstreams.py --only STF
& $py -B -X utf8 scripts/fetch_frozen_data.py --dataset LGC --split val --download
& $py -B -X utf8 scripts/restore_frozen_data.py --dataset LGC --split val --generate
& $py -B -X utf8 scripts/i10_probe.py --patches 2 --repeats 1 --out probe_results/smoke
& $py -B -X utf8 scripts/i10_probe.py --patches 24 --repeats 1 --out probe_results/quick
~~~

没有 NVIDIA GPU 时可按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/) 使用相应的 CPU wheel，并给 probe 添加 --device cpu。若在已存在的仓库克隆中运行，检查点需从 [seed 42 完成快照](https://github.com/qing1989620/paper-shikongyaogan/releases/tag/seed42-complete-20261123) 取出两组 best.pth，并保留路径。协作 ZIP 已含这两组权重。

| 权重 | ZIP 内相对路径 |
| --- | --- |
| 基线 | work/runs/research_2026/replicate_base_seed42_v1/best.pth |
| I10 | work/runs/research_2026/replicate_i10_unmixing_seed42_v1/best.pth |

若需完整验证，运行 --patches 120 --repeats 3，并使用新的 --out 目录。脚本校验源码、冻结清单、入选切片和检查点的 SHA-256，并在终端按验证块打印进度。

恢复训练和验证两部分可使用以下命令。若已有个人保存的切片，省略 --generate 进行校验。CIA 使用 --dataset CIA，输入仍需来自官方集合。

~~~powershell
& $py -B -X utf8 scripts/fetch_frozen_data.py --dataset LGC --split train --split val --download
& $py -B -X utf8 scripts/restore_frozen_data.py --dataset LGC --split train --split val --generate
~~~

若遇到切片字节哈希与冻结值不一致，核对原始文件清单和 NumPy 版本，再在干净的目录重建。工具保留已存在的切片。

## 交回结果

交回 probe_report.json、paired_scores.csv 和 conditions.csv，附设备与入选块数量。完整 I10 是配对差值的参考，正的 rmse_minus_i10_full 表示该模式的误差较高。组间比较和后续动作见 [实验计划](I10_EXPERIMENT_PLAN.md)。
