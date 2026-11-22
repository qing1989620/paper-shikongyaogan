# 现状审计

日期：2026-11-19。交接文档作为待验证记录读取，其中的执行建议没有直接作为本次用户指令执行。

## 结论

现有工作保留了有用的模型、低频校正实验和失败探索，但还没有形成“机制创新在统一基准上超过外部强方法”的证据闭环。更优先的问题是比较口径与实验溯源存在错误。旧表不能直接升级为投稿主表；以下问题也不等于所有旧 RMSE 都无效。

### 同日后续修复记录

- 新研究环境已固定 NumPy 2.2.6 / SciPy 1.15.3，保留系统环境原状。
- 新基础模型路径及 η=0 十步采样与官方实现 max abs diff=0；旧 `official_10` 函数保留供追溯，不再作为新控制。
- `prepare_dataset_v2.py` 可重建明确的 LGC 10 组，30 个代表性训练位置与原图一致；`prepare_train.py` 已改为兼容入口，旧件归档。旧 5 组不一致的事实仍作为原审计记录保留。
- CIA 34 个主文件已恢复完成。原 calib 的末组含验证目标日 2002-02-22，已排除并生成 7 组、294 patch；更新后的 26 个时序组共 78 个代表性位置与原始数据全部匹配，当前训练 fine 与 heldout 目标日无交叉。初始缺失证据另存 audit/initial_20261119_0020/。
- SA-STF LGC 作者权重已下载、83,839,558 个参数值严格加载通过；50 步 CPU 小尺寸功能测试通过。采样函数 AST 与作者代码一致，只跳过推理用不到的训练 VGG 分配。补充材料 Table 3 已目视核验：其训练包含我们的 March02 val，而 January13 flood test 日期无交叉。共同 patch 评测已安排，但空间裁剪/训练配方不等同于原文，详见 [共同基准](COMMON_BENCHMARK_2026.md)。

## 本次确认的问题

| 编号 | 查到的事实 | 对结论的影响 | 处理 |
|---|---|---|---|
| A01 | `TCCR_STF/checkpoints/tccr_stfdiff_lgc_bsq_best.pth` 元数据为 epoch=25、seed=42；模型卡写的是本项目 LGC 训练。不是论文作者发布权重的证据 | “官方发布权重”标签不成立 | 改登记为“项目旧训 STFDiff”；不改原二进制名称 |
| A02 | 当前使用的是 5 个观测条件：F0/F3/C0/C1/C3；F1 是 GT。部分旧文档写“6 条件 + 1 GT” | 输入信息量描述错误 | 统一为五观测条件、双锚；数据包共有六张影像 |
| A03 | 本地官方双锚配置 `config/bi-stfdiff/.../LGC/inference~...py` 是 10 步、η=0；旧 `backbone_stfdiff.py::official_10` 是 η=1 | “严格官方 10 步”对照标签有误；步数与随机性混杂，旧提升比例须重验 | 正式比较补 η=0；η=1 留作采样消融 |
| A04 | LGC 训练目录实际 10 组三元组、1,200 patch；`prepare_train.py::TRIPLES` 只有 5 组、能生成 600 patch | 数据准备不可复现，增加训练数据的收益可能与结构收益混淆 | 已反查全部组的日期，准备阶段修复生成脚本；新实验冻结 manifest |
| A05 | `runs/delsup/best.pth` 为第 3 轮；`train_log.csv` 仅 4 轮；文本日志写 40 轮且最佳为第 27 轮。两份快照权重及 tar 内也是第 3 轮、同 SHA | 不能断言评测了文本日志中的最佳模型，更不能把失败归因坐实为“过拟合固定验证噪声” | 隔离此结果的因果解释；以后新 run_id 重跑，不覆盖旧记录 |
| A06 | 旧 `psnr()` 内已先转 [0,1]；`lock_numbers.conv()` 再加 6.0206 dB | 锁数流程的 PSNR 重复换算 | 已用 6 个真实缓存 patch 数值确认；新指标仅转换一次 |
| A07 | 旧 SAM 在 [-1,1] 光谱上算角度；转为反射率不仅是乘系数，还平移 +0.5，角度会变。旧 SSIM 同样在 [-1,1] 上用 data_range=1 的常量 | SAM 不能只把度换弧度后与文献比；SSIM 不能直接沿用 | 新模块在反射率域重算；旧 CSV 原样保留 |
| A08 | 旧 ERGAS 缺 h/l 因子；官方仓库配置明确使用 1/16 | 单位相差不是算法收益 | 新接口要求显式 ratio；正式表注明 ERGAS(1/16)，来源协议仍需确认 |
| A09 | LGC 旧 val/test 仅各 1 个目标日期；已有多轮 test 扫描。旧统计脚本按 patch 独立 bootstrap/Wilcoxon | 120 个 patch 不等于 120 个独立时间事件，旧 test 不能再称完全未见测试集 | 登记为已观察的开发证据；新结果按日期、空间块报告，另需锁定确认性评估 |
| A10 | CIA 原始 Landsat/MODIS 主文件 34/34 为零字节；已有 val=126、test=168 patch，未有 train patch | 可以做既有 patch 的探索性推理；不能声称原始数据可重建、CIA 域内训练已完成或已做全图评测 | 独立数据恢复任务，不删除现有 patch |
| A11 | 当前 CIA 自定义切分为 calib/val/test=8/3/4 triples；旧预注册表写 8/4/3。官方仓库日期组也与当前 CIA 自定义组不同 | 数据集同名仍不意味着与论文表可比 | 当前事实以 manifest 为准；正式 v1 划分另行冻结 |
| A12 | 全局 Python 中 NumPy=2.5.5，SciPy=1.13.1，导入出现不兼容警告；skimage/timm/OpenCV 未装 | 外部方法并非“下载权重即可跑”，旧环境数值依赖有风险 | 后续隔离环境；本次新增指标只依赖 NumPy，未修改全局环境 |
| A13 | 没找到任何外部方法已在本项目协议下完成的结果行 | 外部 SOTA 超越结论尚无依据 | E00/E01 优先建立真实外部对照 |

上述“官方配置”是本地固定 commit 下的双锚分支，不推广到所有 STFDiff 分支。单锚分支和双锚分支的 η 设置不能混用。

## 数据事实与边界

| 数据 | train | val | test | 当前可用范围 |
|---|---:|---:|---:|---|
| LGC | 1,200 patch，10 组、6 个目标日期 | 120 patch，目标 2005-03-02 | 120 patch，目标 2005-01-13 | 原始 14 对影像和现有 patch |
| CIA | 未生成 | 126 patch，3 个目标日期 | 168 patch，4 个目标日期 | 现有 patch；原始主文件失效 |

LGC 非重叠 patch 只覆盖左上 2560×3072，占原图 2720×3200 的约 90.35%；不能称为全图分数。CIA patch 对应中心 1792×1536，偏移 row=124、col=92。边界遗漏、中心裁剪和全图拼接需区分。

本次对 LGC 12 组每组 3 个 patch（首、中、末）逐数组匹配原始影像，全部一致；代表性检查没有发现当前 LGC 训练使用 val/test 目标日的 fine 影像。该结论不覆盖全部像素、外部 checkpoint 或上游训练历史。CIA 日期只由元数据声明，没有通过原始影像校验。

官方 `syy_setting.py` 把 LGC 2005-03-02 放在训练组，而本项目称其为 val；官方 CIA 训练组含本项目 test0 的目标日 2002-04-02。**因此不能把按该官方划分训练的权重直接放到当前自定义 val/test 上，当成无泄漏对照。** 外部 SA-STF 权重的具体训练日期仍待核实。

## 对旧结论的处理

- RLF 作为历史后处理对照保留；是否有新颖的算法贡献要对照文献，不能仅由“没有新增参数”或“1 NFE”推出。
- base 重训与旧训模型在 val 的差别属于训练差别；base 的旧 flood test RMSE 约 0.02160，旧训 fixed1 约 0.02053，说明 val 提升不等于泛化提升。
- `delsup` 不登记成“已证明有效”，也不登记成“已经确定失败且原因是固定噪声”。先排除权重错配，再做配对重训。
- a≈0.54/R²≈0.28 是特定数据、滤波与回归设定下的统计诊断，尚不能称为普适物理常数或不可逾越的信息上界。
- 旧 `FINAL_NUMBERS_LOCKED.csv` 为历史导出，仍存在上述指标和协议问题。没有新结果时，不制作新的“最终 SOTA 表”。

## 当前外部方法核验

2026-11-19 访问一手来源：

- [SA-STF 官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Semantic-Adaptive_Diffusion_for_Dynamic_Spatiotemporal_Fusion_CVPR_2026_paper.html) 与 [作者仓库](https://github.com/ZZZhangjinsong/SA-STF)：代码和三套权重链接可见。源码已稀疏克隆；权重尚未下载。正式日期划分未在当前 README/配置中完整给出。
- [STFDiff 作者仓库](https://github.com/huang-he99/STF)：上游有模型与数据处理说明；当前工作副本来自 [prowDIY/STF](https://github.com/prowDIY/STF)，固定 commit 为 `22a1c99c0bbd06e4ef931733dbcea9780b0ffd5b`。保留本地新增架构文件，未覆盖或拉取更新。
- [STFMamba 作者仓库](https://github.com/zhaomin0101/STFMamba)：已有本地代码；当前 README 没有确认可下载的预训练权重，依赖 Triton/状态空间算子，Windows 可运行性待验证。
- [DPDiff 作者仓库](https://github.com/paul623/DPDiff)：当前只有代码未来发布的说明，不能当作已可复跑的方法安排必跑期限。

## 证据入口

`audit/` 下：`dataset_manifest.csv`、`dataset_groups.json`、`raw_file_integrity.csv`、`preparation_reproducibility.json`、`split_overlap.json`、`checkpoint_inventory.csv`、`training_log_inventory.csv`、`result_inventory.csv`、`metric_verification.json`、`repositories.json`、`environment.json`。代码和权重包含 SHA-256；新实验还需要 run_id 与这些哈希绑定。
