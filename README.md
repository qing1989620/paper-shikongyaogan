# 遥感时空融合实验

本仓库保存 2026 年遥感时空融合研究的项目代码、冻结实验协议、阶段论文和进度备份。实验在相同 LGC 开发划分上对比基线与 14 个变体，按训练种子配对，并同时记录最佳验证检查点和第 40 轮结果。

| 内容 | 入口 |
|---|---|
| 训练、评测和数据准备 | [work/tools](work/tools/) |
| 队列、暂停和复跑 | [scripts](scripts/) |
| 创新机制和比较口径 | [创新计划](docs/INNOVATION_PROGRAM_2026.md)、[基准协议](docs/BENCHMARK_PROTOCOL.md) |
| 2026-11-22 阶段稿 | [中英文稿与复算源码](work/paper/research_2026/stage_draft_20261122/README.md) |
| 已完成组的阶段进度 | [Seed 42 结果表](docs/SEED42_PROGRESS.md) |
| 当前实验归档 | [备份状态](work/runs/research_2026/BACKUP_STATE.json)、[检查点下载](https://github.com/qing1989620/paper-shikongyaogan/releases/tag/backup-20261123-seed42) |
| 在另一台 Windows 电脑恢复 | [恢复说明](docs/RESTORE_BACKUP.md) |

## 备份进度

2026 年 11 月 23 日的首次快照已保存 seed 42 的 8 组完整结果、对应最佳权重，以及进行中的 I04 第 9 轮续训检查点。结果基于开发验证集，快照内的进度以抓取时刻为准。本机队列以 seed 42 的 15 组全部完成为暂停边界。

## 数据来源

LGC 和 CIA 由 CSIRO Data Access Portal 提供。仓库保存获取入口、主文件哈希、逐切片冻结清单和无覆盖恢复工具。依原数据许可，影像及保留其像素值的切片通过官方来源或授权的本地副本恢复。 [LGC 原始集合](https://doi.org/10.4225/08/514B80FD805F4) · [CIA 原始集合](https://doi.org/10.4225/08/514B7FD1C798C) · [CSIRO Data Licence](https://research.csiro.au/dap/licences/csiro-data-licence/)
