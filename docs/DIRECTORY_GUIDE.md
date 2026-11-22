# 目录与维护约定

整理日期：2026-11-19。此次操作移动归档，没有删除文件，没有去重删除 checkpoint，没有覆盖旧论文或实验 CSV。

## 目录职责

| 目录 | 内容与使用方式 |
|---|---|
| 根目录 README.md | 日常唯一入口，当前状态与下一项工作 |
| docs/ | 现行计划、协议、审计结论、基线状态、实验队列 |
| docs/audit/ | 自动生成的日期/坐标、大小、结果、权重、SHA、环境与迁移清单 |
| docs/history/2026-11-17_18/ | 原 `work/*.md` 历史方案、旧状态表和训练监控，不与现行计划混淆 |
| work/data/ | 当前原始数据与 patch；CIA 原始数据不完整，见该目录 README |
| work/repos/ | 外部方法代码；STF/STFMamba 保持原有 commit 和本地修改；新增 SA-STF 稀疏克隆 |
| work/tools/ | 训练、推理、指标、聚合工具；详见目录内 README |
| work/runs/ | 权重及训练 CSV。旧目录保留；当前十四方向开发实验在 `research_2026/<run_id>/`，正式 benchmark_v1 另存 |
| work/results/ | 历史 CSV/缓存保留；新开发实况在 `research_2026/`，正式 benchmark_v1 另存 |
| work/logs/legacy/ | 原 `work/log_train_*.txt`、`log_arch_*.txt` |
| work/paper/ | 原英文/中文 TeX、PDF、图片与模板，状态为历史草稿 |
| work/configs/ | 新实验显式配置；当前 benchmark v1 为 DRAFT |
| work/envs/research_v1/ | 本轮轻量研究环境，继承系统 Torch，独立固定数值包；不打进论文代码交付包 |
| work/checkpoints/external/ | 新获取的作者权重及来源，和项目自训权重分开 |
| references/papers/ | DeepFDP、GSINet、SA-STF 文献原件 |
| references/literature/ | 原文献深读 XLSX，不修改工作表 |
| references/extracted/ | 已有文献提取文本 |
| archive/handover/ | 原 HANDOVER 与聊天记录，内容原样保存 |
| archive/reports/2026-09/ | 9 月各阶段报告、PDF，历史数字不作新基准 |
| archive/packages/ | 原两个 ZIP 交付包 |
| archive/snapshots/2026-11-18_deliverables/ | 原 `deliverables/` 整体移动，含所有备份权重和 tar；不当成当前可编辑主目录 |
| archive/legacy_tools/ | 旧自动清理脚本原件，仅供复核 |
| scripts/ | 新审计、数值核验、一次性归档脚本 |

## 为什么保留 work 的核心结构

训练、评测和绘图脚本依赖 `work/data`、`work/repos`、`work/runs`、`work/results`、`work/paper`。这些路径保留以避免整理导致运行失效；散落在根目录及 work 顶层的报告、日志和重复交付入口才移动到各自类别。旧历史文档里的路径保留原貌，查不到时用迁移清单定位。

旧 `tools/cleanup.py` 原件已归档，它默认会删除消融所需缓存。现入口改为只列出缓存占用，不删除；当前约 901 MiB 的 base val/test 缓存继续保留。

后续修复时另将旧 `prepare_train.py` 保存为 `archive/legacy_tools/prepare_train_legacy_2026-11-19.py`，新兼容入口调用数据准备 v2。这是初次 38 项迁移之后的独立修复，记录在现状审计；原 210 文件迁移账本保持历史原貌。

## 追溯或撤销一次移动

打开 `docs/audit/organization_moves.json`。每一项记录原路径 `source`、新路径 `destination`、文件大小与 SHA-256，并已标记 `done=true`、总校验 `verified=true`。本次 38 项移动、210 个文件前后哈希一致。

如需恢复某个文件，先确认原位置不存在新文件，核对 SHA，再把新位置的对象移回所列 source；不要直接覆盖。该账本是原始迁移记录，不随新文档更新。外部微信目录中的交接原件未动。

## 后续如何保持可管理

1. 新实验先登记 `EXPERIMENT_QUEUE.csv`，一次运行一个新 run_id；不复用旧 `runs/delsup` 写入新日志。
2. 小文件配置/指标/日志随运行存储；大权重只存工作副本及有版本的必要快照，避免每次复制全部目录。
3. `FINAL` 是流程状态而不是文件名承诺；当前旧 `FINAL_*` 仅是历史数据。正式输出带 protocol_id、metric_version 与源 run_id。
4. 读者面向入口只有根 README、现行计划和注册表；旧方案进 history，旧大交付进 archive。
5. 本次没有改写旧研究结论来掩盖问题；改进后的结论从新实验产生。

## 复核命令

在根目录运行 `python -X utf8 scripts/audit_workspace.py` 生成当前清单；`python -X utf8 scripts/verify_metrics.py` 验证新指标和缓存样本。审计脚本的原图检查是每组三个代表性 patch，不是所有 patch 的全像素认证。

`organize_workspace.py` 默认只显示迁移计划；已有本次账本后再次执行不会重复移动。现在无需再次整理。
