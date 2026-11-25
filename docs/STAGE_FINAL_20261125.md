# 2026-11-25 阶段定稿与实验归档

两个训练种子各完成 15 组 40 轮复验，共 30 组。另有 15 组六轮初筛。队列在 2026 年 11 月 25 日 19:15:14（北京时间）写入第二个种子的暂停状态。[暂停记录](../work/runs/research_2026/PAUSE_STATUS.json)

## 稿件

- [中英文 PDF、LaTeX 源码与可编辑机制图](../work/paper/research_2026/stage_final_20261125/README.md)
- [45 组结果表](../work/paper/research_2026/stage_final_20261125/evidence/scores.csv)
- [当前强基线与机制验证计划](I10_EXPERIMENT_PLAN.md)

## 实验资产

| Release 资产 | 内容 |
| --- | --- |
| `STF_stage_final_20261125.zip` | 中英文稿、可编辑机制图、编译和复算脚本及派生证据 |
| `STF_experiment_records_20261125.zip` | 45 组配置、源码快照、逐轮日志、逐块验证、冻结划分、运行状态与 I10 两块干预 |
| `STF_checkpoints_screen42_20261125.zip` | 15 组初筛的最佳和续训权重，以及初筛共用起点 |
| `STF_checkpoints_seed42_20261125.zip` | 种子 42 的 15 组最佳和续训权重 |
| `STF_checkpoints_seed123_20261125.zip` | 种子 123 的 15 组最佳和续训权重 |
| `SHA256SUMS.txt` | 发布资产的 SHA-256 |

权重包保留每组的原始 `best.pth` 与 `resume.pth`。每个实验 ZIP 中的 `ARCHIVE_MANIFEST.json` 列出相对路径、大小和 SHA-256。下载后可以先核对发布资产哈希，再将需要的包解压到独立恢复目录。原始影像的获取方法见 [数据集手册](DATASET_MANUAL.md)。

下载入口见 [GitHub Release](https://github.com/qing1989620/paper-shikongyaogan/releases/tag/stf-stage-20261125)。

## 本阶段支持的结论

| 训练种子 | 基线最佳 RMSE | I10 最佳 RMSE | 最佳降幅 | 基线第 40 轮 | I10 第 40 轮 | 第 40 轮降幅 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.025080 | 0.024233 | 3.38% | 0.025618 | 0.025140 | 1.87% |
| 123 | 0.024722 | 0.024417 | 1.23% | 0.027345 | 0.026445 | 3.29% |

这是两个训练种子在同一 LGC 开发日期上的内部配对结果。稿件保留了 I14 的共同终点表现、当前时间修正干预及场景内不确定性。强外部方法原版对照和独立日期验证按后续计划执行。
