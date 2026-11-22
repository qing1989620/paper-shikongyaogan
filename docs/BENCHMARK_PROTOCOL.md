# 统一基准协议

日期：2026-11-19。状态：**DRAFT，原则已确定，正式数据日期、掩膜和权重来源未全部核实，不应启动确认性排名。**

本协议区分历史开发数据 `legacy_v0` 和将冻结的正式 `benchmark_v1`。规则改变时增版本，不覆盖旧分数。

## 1. 任务与输入

`{F_prev, F_next, C_prev, C_target, C_next} -> F_target`：五个观测条件、六波段。`F_target` 只用于训练加噪/监督与评测，不传入推理函数。模型形状由观测影像决定，禁止借 GT 内容生成初始化或门控。

双锚模型可用相同五观测。单锚方法按原生输入标注；若做双向融合，融合规则只在验证集确定，单独命名。不给某方法额外真实目标日 fine、未来外部标签或未披露预训练数据。

## 2. 数据与划分

- 初始 `legacy_v0` 清单保存在 `audit/initial_20261119_0020/`。当前清单见 `audit/dataset_manifest.csv`：LGC train=10 组、val/test 各 1 组；CIA val/test=3/4 组，新增过滤日期冲突后的 train=7 组、294 patch，属于本次内部域内开发配置，不称作者官方划分。
- 原始 DN 裁剪 [0,10000] 后归一化至 [-1,1]，只是模型接口；评测显式恢复反射率 `(x+1)/2`。现有 float16 patch 已有量化，不把它当未量化原图。
- 数据布局 LGC BSQ；CIA fine BIL、coarse BSQ。CIA 原始文件已恢复；26 个时序组每组三个代表性位置匹配通过，不将抽样核验扩大为每个像素全部验证。
- 正式 v1 先对照作者数据划分，选择公开且可重建的日期协议并为所有重训方法共用。选择依据是可比性和独立性，不能挑对本文有利的日期。
- 每个 evaluation target 的 fine 日期不得出现在该模型训练输入/标签或额外预训练样本中。参考锚点重复使用按公开协议声明；空间同区域的时间插值与跨区域泛化分开表述。
- 外部权重训练日期不明：状态为 `train_overlap_unknown`，可探索运行，不进入干净排行榜。官方权重和当前自定义 split 冲突时，采用匹配的官方评测日期或在共同划分重训；不直接拼表。
- 历史 test 已多轮观察，不再声称完全封存。新确认性评估需预先固定方法并登记暴露历史；若新增独立数据，则在冻结后评估。
- v1 必须补：原始文件校验、日期表、band 顺序、有效区域与 NoData 掩膜、裁剪、波段/日期配准及分块坐标。所有方法同一掩膜，不按方法结果选择有效像素。

## 3. 两种空间评测

**patch 开发评测**：为快速定位问题，沿用固定 manifest，各方法相同 patch 集。图像 RMSE 与 patch 平均 RMSE 不是同一值。

**正式场景评测**：原始数据完整后，确定公共有效 ROI；可分块推理，但要冻结 padding、stride、边缘回填和重叠拼接，先重建一个场景再计算指标。SA-STF 作者推理采用 256 patch、stride=128 的中心区域拼接，当前旧代码是不重叠 patch，这个差别必须处理。

主表按目标日期先算场景指标，再等权平均日期；附录同时给逐日期与 patch 分布。RMSE 还保留全有效像素的 SSE/count，另报 pooled RMSE。不同日期面积不等时，不用 patch 数量无意决定权重。

## 4. 指标

新实现：`work/tools/benchmark_metrics.py`，版本 `reflectance_v1`。所有输入显式为反射率 HWC，预测不静默裁剪；若采用裁剪，所有方法同规则，另报越界比例。

| 指标 | 定义/单位 |
|---|---|
| RMSE | `sqrt(mean((prediction-GT)^2))`，反射率；主指标 |
| MAE | `mean(abs(prediction-GT))`，反射率 |
| PSNR | `-10*log10(MSE)`，data_range=1，dB；只做一次域转换 |
| SSIM | 11×11 Gaussian，σ=1.5，data_range=1，population covariance；各 band 等权；仅完全有效的窗口中心，去 5 pixel 边界 |
| SAM | 在反射率光谱向量算角度；主列 rad，附列 degree；零 GT 范数剔除并计数，GT 非零而预测零按 π/2 计 |
| ERGAS | `100*(h/l)*sqrt(mean_b(MSE_b / mean(GT_b)^2))`；h/l 必填，初始对照作者仓库 1/16，冻结前确认并在列名/元数据声明 |
| CC | 各 band Pearson 后等权平均；常数 band 返回未定义与计数，不用 0 假装有效 |

零均值 band 的 ERGAS、常数 band 的 CC 未定义时返回 NaN 和数量，由汇总器显式处理，不静默丢列。NoData 不简单用“所有零像素”猜测，须核实原始说明和统一掩膜。

旧 RMSE/MAE 一般可除以 2 恢复反射率；旧 SAM/SSIM 必须从预测重新算，不能只在聚合时改比例。旧 PSNR 导出重复加了约 6.0206 dB。新旧指标表不混用。

## 5. 训练、选模与推理随机性

- 机制比较训练种子：42、123、2026；各组配对初始化、数据顺序和噪声流，记录确定性设定与平台。
- 验证固定 3 个推理噪声 realization，分别计算后平均指标，不能先平均生成影像再打分。噪声 seed 由稳定的 dataset/group/row/col/repeat 哈希决定，不能是当前排序的 `42+i`。
- 选模使用与正式评测一致的反射率指标与日期聚合；不把 normalized global RMSE 与 mean patch RMSE 当同一验证值。
- 独立推理重复使用另一组已登记噪声，仅估计稳定性。增加噪声重复能减小随机波动，不能替代独立日期。
- 外部复现保留作者推荐采样设置；STFDiff 双锚本地参考配置是 10 步 η=0，1/2 步和 η=1 单列消融。不能仅压低外部方法 NFE 来做主精度比较。
- 新训练完整记录 seed、epoch/global_step、优化器、scheduler、验证样本/噪声、代码/数据 SHA、checkpoint SHA 和选择规则。

## 6. 统计与效率

- 配对比较使用完全相同日期、区域和 mask。多个日期足够时按日期重采样，并考虑日期内空间块相关性；同一目标日的不同锚组合也属于同一 cluster。
- 只有 1 个日期时只能描述该场景的空间不确定性，不能用 patch Wilcoxon 的小 p 值声称跨事件泛化。旧显著性仅留为探索记录。多对比时用 Holm 等预定校正。
- 训练种子差异与推理随机差异分开；同时给均值、标准差、逐日期值与配对差值置信区间。
- 计时在相同设备和输入尺寸进行，预热、CUDA synchronize，报中位数/P95、NFE、参数量、峰值显存；区分 model-only 和含 I/O/拼接的端到端。缓存命中的零耗时不用于速度表。

## 7. 输出合同

每次独立 `run_id` 至少包含：`config.json`、`environment.json`、`manifest_sha256`、源码 commit 与修改文件哈希、checkpoint SHA、`train.log`/`eval.log`、`per_patch.csv`、`per_date.csv`、`summary.json`、`predictions/` 或带来源哈希的可再生缓存。

结果表字段含 dataset、protocol_id、metric_version、method、run_id、train_seed、inference_repeat、date、patch_id、metric、value、mask_id。主表必须标记 training regime、输入协议、NFE、是否重训与外部预训练来源。

`work/configs/benchmark_v1.json` 中 `freeze_status=DRAFT` 的未决字段解决后才能登记 FROZEN；当前不会因写出协议文件就声称完成实验。
