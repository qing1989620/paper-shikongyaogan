# 双锚成分比例传递用于单步遥感时空融合

2026 年 11 月 25 日中英文阶段定稿。采用匿名 A4 双栏版式，两篇各五页，任务、数字、公式和图表在两种语言中对应。

| 内容 | 文件 |
| --- | --- |
| 中文正文 | [PDF](output/pdf/STF_stage_final_zh.pdf) · [LaTeX 入口](source/manuscript_zh.tex) |
| 英文正文 | [PDF](output/pdf/STF_stage_final_en.pdf) · [LaTeX 入口](source/manuscript_en.tex) |
| 可编辑机制图 | [中文版](source/figures/mechanism_zh.pptx) · [英文版](source/figures/mechanism_en.pptx) |
| 完整结果 | [45 组分数](evidence/scores.csv) · [逐轮曲线](evidence/curves.csv) · [逐块配对](evidence/patch_pairs.csv) |
| I10 诊断 | [场景内区间](evidence/pair_intervals.csv) · [粗观测分层](evidence/coarse_strata.csv) · [通道熵](evidence/i10_entropy.csv) · [固定权重干预](evidence/i10_probe.json) |
| 实验与稿件归档 | [GitHub Release](https://github.com/qing1989620/paper-shikongyaogan/releases/tag/stf-stage-20261125) |

## 本次结果

在同一个 LGC 开发日期上，I10 的最佳检查点 RMSE 在两个种子分别降低 3.38% 和 1.23%。共同第 40 轮分别降低 1.87% 和 3.29%。I10 是唯一同时满足四个比较方向均改善的变体。另有 I01 在两种子最佳检查点均改善，I03、I12、I14 在两种子共同第 40 轮均改善。共同第 40 轮最低值来自 I14。

I10 在两次训练中都有接近单通道的目标权重。两块干预样本的粗偏差置零或置乱结果相同。空间区间仅估计当前开发场景内的变异。强基线共同协议比较、机制重训和独立日期确认进入 [后续实验计划](https://github.com/qing1989620/paper-shikongyaogan/blob/main/docs/I10_EXPERIMENT_PLAN.md)。

## 术语和数字

| 中文 | 英文与符号 |
| --- | --- |
| 双锚成分比例传递 | Dual-anchor component proportion transfer，I10 |
| 目标日粗偏差 | Target-date coarse departure，$D_C$ |
| 最佳检查点 | Best checkpoint，按组内最低验证 RMSE 选择 |
| 共同第 40 轮 | Common epoch-40 endpoint，取逐轮日志中的第 40 轮 |
| 配对 RMSE 降幅 | Paired RMSE reduction，基线减变体后除以同种子基线 |

## 复算和复编

安装有 XeLaTeX、BibTeX 和 latexmk 的 Windows 环境可直接运行 `.\build.ps1`。中英文共用 `values.tex` 和两张结果表的行数据。中文使用宋体、楷体和微软雅黑，英文使用 Times New Roman。机制图保留原可编辑形状。统计图由随包脚本从证据 CSV 生成。

从 Release 下载并解压实验记录 ZIP 后，可用 Python 3.12、NumPy 和 Matplotlib 重算并编译。`$recordsRoot` 指解压后含有 `work` 的目录。

~~~powershell
python -B -X utf8 .\scripts\freeze_evidence.py --records-root $recordsRoot
python -B -X utf8 .\scripts\plot_results.py
.\build.ps1
~~~

粗观测偏离量来自同一冻结开发网格，其原始数值保存在 `evidence/coarse_deviation.csv`。主表的最佳值取各组冻结摘要，共同终点取逐轮日志。空间配对从导出的逐块 RMSE 重算，适用范围为同一开发场景。
