# Pre-Registration — Reference-Pair Self-Calibrated Temporal Projection (RSP-STF)

版本：v1.0 ｜ 日期：2026-11-17 ｜ 状态：预注册（在看任何新结果前冻结）

> 本文件的作用：在跑新实验之前锁死协议、判据、指标和统计方法，防止 p-hacking / test 污染。
> 任何对本文件的修改必须追加变更记录并注明时间与理由。

---

## 1. 研究问题与假设

**背景**：STFDiff 官方双锚点协议（6 条件输入：fine_ref_prev/next + coarse_ref_prev/pred/next → fine_target）
下，第一步扩散输出 x₁ 存在系统性低频时间偏差。Stage3/4 已证明 Temporal Projection 有效，
但源域校准系数（12 个 per-band affine）具域依赖，LGC→CIA 零样本迁移不稳。

**核心方法 RSP**：对每个测试样本，用其自带的两对 reference pairs (F_prev,C_prev),(F_next,C_next)
以闭式最小二乘拟合 per-band 时间仿射 `LP(F_next−F_prev) ≈ a·LP(C_next−C_prev)+b`，
再用前后两锚双向估计预测日低频目标 T，对 x₁ 做单次投影 `x_out = x₁ − λ[LP(x₁)−T]`。
不访问预测日 fine GT、不更新 backbone、零新增神经网络参数、1 NFE。

**预注册假设**：
- H1（主）：在同一域内，RSP 的 RMSE 显著低于最强固定步数基线（fixed-1 / fixed-2）。
- H2（效率）：RSP 保持 1 NFE，推理延迟显著低于官方 10 步。
- H3（跨域）：逐样本自校准（RSP）在 CIA 上优于源域固定校准（Stage3 projection），
  且不使用预测日 GT。
- H4（机制）：增益来自低频时间残差修正；对高频子带几乎无改动（SAM 不恶化）。

**证伪条件（预先写死）**：
- 若 RSP 在 LGC val 上 RMSE 不优于 fixed-1，或 CI 跨 0，则 H1 证伪 → 论文主线放弃 RSP。
- 若 CIA 上 RSP 不优于 source-cal，则 H3 证伪 → 跨域主张删除，只保留单域。
- 若 SSIM/SAM/ERGAS 任一指标显著恶化（>2%），即使 RMSE 降也不得宣称整体更优。

---

## 2. 数据协议（冻结）

| 数据集 | 来源 | 规模 | 用途 | split 规则 |
|---|---|---|---|---|
| LGC (Lower Gwydir Catchment) | CSIRO csiro:5847 / Zenodo 10.5281/zenodo.15117928 | 14 对 L5 TM/MODIS 3200×2720@25m + 洪水掩膜 | 主域 | train=2004 年 9 对；val/flood=2005-01-13、03-02、04-03 等（与官方 syy_setting-9 一致）；flood=2004-12-28→2005-01-13 |
| CIA (Coleambally) | CSIRO csiro:5846 | 17 对 L7/MODIS 2040×1720 | 跨域验证 | date-level：calib=前 8 triples，val=中 4，test=后 3（冻结后不得改） |
| DX (大兴) | 百度网盘（需账号） | 中国北京城区 L8/MODIS | 可选第三域 | 拿不到则如实登记 BLOCKED，不臆造 |

**字节布局（已审计，冻结）**：
- LGC 全部 `.int` = BSQ (6,2720,3200) uint16 LE；读成 BIL 会产生错切图像（历史 D3-1 错误，禁止重犯）。
- CIA Landsat `.bil` = BIL；CIA MODIS `.int` = BSQ。
- DN 裁剪 [0,10000]，归一化 `DN/10000*2−1`。

**输入协议声明**：官方双锚点 6 输入，fine_target 只作训练 noising/loss 与评测 GT，推理期绝不进 TCCR。
论文必须如实写此协议，不得写成三输入 F0,C0,C1→F1。

---

## 3. 对比方法（主表）

同协议、同数据处理、同评测代码下复现/对比：
1. B0 Official STFDiff DDIM 10 步（η=1 官方配置）
2. B1 fixed-1（1 NFE，最强固定基线）
3. B2 fixed-2（2 NFE）
4. B3 Stage3 Projection-only（源域校准 a/b + λ）
5. B4 Stage3 TCCR full（路由）
6. **RSP（本文）**：逐样本自校准 + 双向低频目标估计 + 单次投影，1 NFE

协议不同不得进主表（GSINet/DeepFDP 缺 C0；SA-STF 五输入），只进 related work 协议对比表。
SwinSTFM 有官方 LGC 权重可作补充；STARFM 作定性佐证并注明口径差异。

---

## 4. 消融矩阵（在 val 上做全矩阵，test 只验 best+baseline）

| 因子 | 水平 |
|---|---|
| 校准方式 | source-cal（Stage3 固定 a/b） / per-sample OLS / per-sample scale / median / MAD / identity |
| 目标估计 | prev-only / bi-avg（前后锚双向）/ next-only |
| λ（投影步长） | 0（=fixed-1）/ 0.2 / 0.3 / 0.5 / 0.75 / 1.0 |
| 作用频段 | 仅低频（LP σ∈{1,2,3}）/ 全频（对照，预期有害） |
| 门控 | low-change gate q∈{0,0.25,0.5,0.75} |
| 是否与 TCCR 路由组合 | RSP-only / RSP+路由 |

---

## 5. 指标（全集，禁止只报 RMSE）

主指标：**RMSE↓**。次指标：MAE↓、PSNR↑、SSIM↑、SAM↓、ERGAS↓、CC↑。
效率：NFE、e2e latency(s/patch)、额外参数量、校准耗时(ms/sample)、峰值显存。
统计：配对 Wilcoxon（patch 级）+ date-level / spatial block bootstrap 95% CI（防 pseudo-replication）。
显著性水平 α=0.05；多重比较用 Holm 校正。种子 {42,123} 双 backbone checkpoint 复跑主变体。

**唯一数据源纪律**：所有论文数字来自一个 `FINAL_NUMBERS_LOCKED.csv`，由 raw run CSV 自动生成；
README/论文/报告只引用该文件。历史冲突数字（0.02002 vs 0.02032；CIA zero-shot −0.8% vs +2.5%）
一律不进主文，除非由 raw CSV 重算确认。

---

## 6. 成功判据（投稿门槛）

- **ICASSP/DLNN 级**：H1 成立（RMSE 显著优于 fixed-1，bootstrap CI 不跨 0，双 seed 一致）+
  完整指标无恶化 + 效率 ≥5× + 至少一个跨域正结果（CIA 或 LGC flood）。
- **CVPR 级**：上述全部 + 第二 backbone（DiffSTF 或 STFMamba）验证 + 第三数据集或严格三输入协议。
- 任一核心假设证伪 → 按 §1 证伪条件降级或转向，不硬凑。

---

## 7. 执行顺序（防止 test 污染）

1. 数据下载 + 字节布局审计（manifest + per-band mean 自检）
2. 复现 B0（官方 checkpoint / 重训至 plateau），锁定 val split
3. 在 val 上跑 §4 全消融矩阵，选出 best 变体（λ、校准方式、目标估计形式）
4. 冻结 best 超参与 calibration 协议
5. test 只跑一次：best + B0 + B1/B2，写入 FINAL_NUMBERS_LOCKED.csv
6. CIA 用冻结协议做域内校准 + test（不做反复调参）
7. 统计检验 + 效率基准 + 第二 seed 复跑
8. 生成 qualitative 三联图 + 论文图

---

## 8. 撞题边界（论文写作时遵守）

守住 RSP 七要素组合：双锚点协议 + 逐样本闭式自校准 + 不用预测日 fine GT +
不更新 backbone + 零新增网络参数 + 1 NFE + 对首步输出低频 temporal residual 投影。
- 不写 "first temporal consistency" / "first data consistency diffusion"（CCDF/SA-STF 已占）
- 不写 "test-time adaptation" 裸标签（TAO/跨区域 LST TTA 已占），用 "parameter-free test-time self-calibration"
- 不写 "learned temporal prior"（DPDiff TCP 已占），用 "fitted per-sample temporal affine"
- 与 SA-STF TFA 逐条对比：CNN 逐像素/训练期/深层特征/50步 vs 闭式逐样本/测试期/输出残差/1步
