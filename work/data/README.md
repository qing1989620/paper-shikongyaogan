# 数据状态

2026-11-19 审计：

| 路径 | 当前状态 |
|---|---|
| patches_lgc/train | 1,200 patch，10 个时序组；日期已经从原始像元反查 |
| patches_lgc/val | 120 patch，一个目标日期：2005-03-02 |
| patches_lgc/test | 120 patch，一个目标日期：2005-01-13；历史上已经多次评测 |
| raw/LGC | 14 对原始主文件，字节数符合读取形状；每组三个代表性 patch 匹配通过 |
| patches_cia/val | 126 patch，3 个目标日期，保留 |
| patches_cia/test | 168 patch，4 个目标日期，保留 |
| patches_cia/train | 新生成 294 patch，7 组，训练 fine 不含 val/test 目标日；尚未据此完成 CIA 域内训练 |
| raw_cia/CIA | 34 个原始主文件已全部恢复，字节数及本地 SHA 已登记 |

清单：[dataset_manifest.csv](../../docs/audit/dataset_manifest.csv)；完整性：[raw_file_integrity.csv](../../docs/audit/raw_file_integrity.csv)。更新后共 2,028 patch、26 个时序组，每组三个代表性位置与原图逐数组匹配通过。日期是数据包配对目录的规范标签，不能据此声称已核查每个传感器的实际 UTC 获取时间。

旧 `tools/prepare_train.py` 只定义 5 组的问题已修复：现在该入口调用 `prepare_dataset_v2.py`，读取明确的 10 组配置；已对每组三个 patch 共 30 个真实位置匹配通过。默认只盘点，生成时要求新空目录，不覆盖本目录。旧脚本存于 archive/legacy_tools。

CIA 原始主文件恢复完成，初始零字节状态保存在 docs/audit/initial_20261119_0020/。旧 calib 最后一组三元组含验证目标日 2002-02-22，不能直接作为训练输入；v2 已过滤该组并生成 7 组、294 patch。已有下载分片保留。

所有现有 split 属于历史开发协议；正式 benchmark_v1 日期、mask 和整图拼接仍待冻结。
