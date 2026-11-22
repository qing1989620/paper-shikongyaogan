# RSP-STF 实验结果与状态（2026-11-17 夜 → 11-18 凌晨）

## 1. 决策（用户拍板）
- 目标会议：**DLNN**（有分论坛，不看截止时间）
- 数据集：**LGC + CIA**，跳过 DX

## 2. 数据（全部真实下载，非模拟）
| 数据集 | 来源 | 完成情况 |
|---|---|---|
| LGC | CSIRO csiro:5847（公开） | Landsat 14/14，MODIS 13/14（含 val/test 所需全部 6 个日期）+ 洪水掩膜 |
| CIA | CSIRO csiro:5846（公开） | Landsat 17/17 + MODIS 17/17 |

字节布局已审计：LGC 全部 `.int` = BSQ（6,2720,3200）；各波段均值 679/991/1220/2491/2646/1944，BIL 会得到全一致 1661（历史坑已避开）。CIA Landsat=BIL、MODIS=BSQ。

## 3. 环境
- Python 3.12 + torch 2.5.1+cpu + torchvision + numpy/scipy/matplotlib/tifffile/einops/ema-pytorch
- LaTeX：D:/zhuomian_app/latex/texlive/2026/bin/windows/pdflatex.exe（IEEEtran）
- 模型：STFDiff model6，dim=64, dim_mults=(1,2,4), T=100, pred_x0；checkpoint 加载 missing=0/unexpected=0，参数量 **7.489M**（与报告一致）

## 4. 代码资产（work/）
- `tools/backbone_stfdiff.py`：重建 backbone + 官方10步 / 1-2步 DDIM
- `tools/prepare_patches.py`：BSQ/BIL 读取、裁剪、256 patch、官方三元组切分
- `tools/eval_methods.py`：fixed1/fixed2/official10/源域校准投影/自校准投影 + 7 指标 + 预测缓存
- `tools/ablate_rsp.py`、`tools/stats_tests.py`、`tools/lock_numbers.py`、`tools/make_figure.py`、`tools/make_method_fig.py`
- `tools/dl3.py`：线程池分块下载器（最终稳定版）

## 5. 主结果（LGC，120 patch/split，论文口径 [0,1]）
### Stable split (val: 2005-02-14 → 03-02 → 04-03)
| 方法 | RMSE | SSIM | SAM | CC |
|---|---|---|---|---|
| Official 10-step | 0.02972 | 0.8821 | 3.26 | 0.7517 |
| Fixed-1 | 0.02815 | 0.8836 | 3.16 | 0.7732 |
| Source-cal projection (λ=0.75) | **0.02296** | **0.8903** | **2.24** | **0.8742** |
| RSP prev (λ=0.2) | 0.02757 | 0.8874 | 3.15 | 0.8138 |

### Flood split (test: 2004-12-28 → 2005-01-13 → 01-29)
| 方法 | RMSE | SSIM | SAM | CC | vs fixed-1 |
|---|---|---|---|---|---|
| Official 10-step | 0.02161 | 0.8691 | 2.20 | 0.8649 | +5.3% |
| Fixed-1 | 0.02053 | 0.8682 | 2.11 | 0.8752 | — |
| Source-cal projection (λ=0.75) | 0.02236 | 0.8587 | 2.17 | 0.8632 | **+9.0%** |
| **RSP prev (λ=0.2)** | **0.01914** | 0.8695 | 1.98 | 0.8828 | **−6.7%** |
| RSP bi-avg (λ=0.2) | 0.01929 | 0.8698 | 1.96 | 0.8835 | −6.0% |

### 统计
- val: RSP vs fixed1 −2.05%, p=0.0069, 胜率 70.8%
- test: RSP vs fixed1 −6.73%, p=1.6e-20, 胜率 93.3%
- test: src vs fixed1 +8.96%, p=3.1e-8

### 关键发现
1. **1 步优于官方 10 步**（两个 split 都是），复现"过采样"结论。
2. **源域校准不能迁移**：在 val 上 −18.4%，同一冻结配置在 flood test 上 +9.0%。
3. **RSP 是唯一两个 split 都不退化的方法**。
4. 消融：投影本身贡献主要增益（identity affine 已到 0.01936）；逐波段仿射在 flood 再降 1.1%，但在 val 上有噪声（identity 0.02533 优于 OLS 0.02757）——已如实写入 limitation（anchor span 与 prediction span 不匹配）。

## 6. 产出文件
- `paper/main.tex`、`refs.bib`、`paper/main.pdf`（IEEEtran，4 页，待补 CIA 扩展到 5 页）
- `paper/figures/method_rsp.png`（方法示意图）、`paper/figures/qual_lgc.png`（定性对比）
- `results/FINAL_NUMBERS_LOCKED.csv`（唯一数字源）
- `results/lgc_val.csv`、`results/lgc_test.csv`（per-patch 原始结果）、summary json
- `EXPERIMENT_PRE_REGISTRATION.md`、`latest_literature_and_collision_audit.md`、`literature_table_summary.md`

## 7. 待办
- CIA val/test（进行中）→ 补跨域表
- 论文扩到 5 页、填 CIA、最终编译
- nature-skills 未安装（skill 搜索工具报 Harness closing）；绘图用 matplotlib + plot-is-all-you-need

## 8. 四分割一致性（最终结论，论文 Table V）
相对 fixed-1 的 RMSE 变化（%），负=更好：

| 配置 | LGC-val | LGC-test | CIA-val | CIA-test |
|---|---|---|---|---|
| Official 10-step | +5.57 | +5.26 | −2.49 | +1.31 |
| Source-cal λ=0.2 | −8.47 | −3.72 | **+1.15** | −3.38 |
| Source-cal λ=0.75（val 选） | **−18.44** | **+8.95** | **+19.46** | n/a |
| RSP bi-avg λ=0.2 | −0.23 | −6.00 | −1.82 | −1.58 |
| **RSP prev λ=0.2（val 选）** | **−2.05** | **−6.74** | **−1.08** | **−0.90** |

**唯一在四个 split 全为正的两个配置都是 RSP 变体**；源域校准在 LGC-val 上最强（−18.4%）却在 LGC-test（+9.0%）和 CIA-val（+19.5%）上显著变差 —— 方向随 regime 翻转。

## 9. CIA test（168 patch，LGC 冻结超参，零样本）
| 方法 | RMSE | SSIM | SAM | CC |
|---|---|---|---|---|
| Fixed-1 | 0.02741 | 0.8035 | 2.55 | 0.7783 |
| Source-cal λ=0.2 | **0.02649** | 0.8053 | 2.46 | 0.7939 |
| RSP prev λ=0.2 | 0.02717 | 0.8053 | 2.47 | 0.7944 |
| RSP bi-avg λ=0.2 | 0.02698 | 0.8057 | 2.44 | **0.8104** |

注意：CIA test 上源域校准反而更好（与 CIA val 相反），进一步说明其方向不稳定；RSP 两个 split 均为正且 CC 提升最大。

---

## 10. 真训练阶段（11-18 05:00–）

### 环境
CUDA torch 2.5.1+cu121（2.45GB 轮子，64 线程拉取）；RTX 4050 6GB。
修掉：AMP fp16 NaN、lr=1e-3 发散（pred_noise std 7.7）、GPU 僵尸进程占 5.2GB、D 盘写满。
训练配置：batch 2，lr 2e-4 + warmup 200 + cosine，梯度裁剪 1.0，fp32，确定性验证噪声。

### 数据
训练集从 5 个三元组扩到 **10 个三元组 1200 patch**（2004 全年 8 个日期，val/test 严格不参与）。

### Base（官方配置复现，40 epoch，156s/epoch）
- **best val RMSE = 0.0481（原始域）= 0.0241（论文口径）**
- 对比发布 checkpoint 在同一 val split 上 0.02815 → **新训练好 17%**

### 双 backbone × 20 变体（相对 fixed-1 的 RMSE 变化，λ 由 val 选）

| 变体 | 发布 ckpt val | 发布 ckpt test | 重训 ckpt val | 重训 ckpt test |
|---|---|---|---|---|
| identity 投影 | −23.15 (λ.75) | +7.80 | **−13.54 (λ.5)** | **−5.38** |
| 门控 E14 | −3.47 (λ.5) | **−6.98** | +0.25 (λ.3) | −4.18 |
| OLS 自校准 | −2.05 | −6.73 | +9.57 | −6.05 |
| OT 仿射 | −11.75 | +6.12 | −6.21 | +0.62 |

**核心发现**：校正方式的最优性取决于 backbone 质量。
- 欠训练 backbone（发布 ckpt）偏差大 → 门控校准最稳。
- 训练充分的 backbone（0.0233）偏差小 → **不做校准的 identity 投影在 val/test 同时最优**，校准反而过修。
- 这解释了为什么"逐样本仿射"在不同设定下符号翻转：它估计的是偏差，而偏差本身随训练质量变化。

### 进行中
自动链依次训练 lf / hf / bandw / onestep 四个训练期变体（每个 ~105 分钟）。完成后同样跑 4 split × 20 变体，形成"训练变体 × 推理变体"的完整矩阵。

### 三模型一致性（identity 低频替换，λ 由 val 选得均为 0.5）

| 模型 | val fixed1 | test fixed1 | identity val | identity test |
|---|---|---|---|---|
| base | 0.02329 | 0.02160 | **−13.54%** | **−5.38%** |
| lf | 0.02448 | 0.02104 | **−15.18%** | **−5.90%** |
| hf | 0.02415 | 0.02083 | **−14.39%** | **−4.08%** |

三个独立训练的 backbone 上，同一算子（无校准、单次低频替换、λ=0.5）在两个 split 上**全部为正**，幅度稳定在 −13~15% / −4~6%。这是目前最可复现的结果。

对照：同一批模型上
- 源域校准：val −9.6~−11.4%，test −1.7~−3.5%（一致但幅度小）
- 自校准 RSP：val **+7.5~+9.6%（有害）**，test −6.1~−7.7%（有用）

→ 校准类方法的符号依赖 regime 与 backbone 质量；不校准的低频替换才是稳定项。

### 四个独立训练 backbone 的完整对照（λ 均由各自 val 选出）

| 模型 | fixed1 val | fixed1 test | 源域校准 val/test | 自校准 val/test | **identity val/test** |
|---|---|---|---|---|---|
| base | 0.02329 | 0.02160 | −9.6 / −3.5 | +9.6 / −6.1 | **−13.5 / −5.4** |
| lf | 0.02448 | 0.02104 | −11.4 / −2.9 | +7.5 / −7.7 | **−15.2 / −5.9** |
| hf | 0.02415 | 0.02083 | −10.7 / −1.7 | +8.0 / −6.2 | **−14.4 / −4.1** |
| bandw | 0.02424 | 0.02122 | −10.0 / −1.6 | +8.4 / −8.0 | **−14.2 / −5.6** |

**结论（四个模型一致）**：
1. 不校准的低频替换在 val 稳定 −13.5~−15.2%、test −4.1~−5.9%，跨模型方差 <2pp。
2. 逐样本自校准在 val 一致有害（+7.5~+9.6%），只在 test 有用（−6.1~−8.0%）。
3. 源域校准方向一致但幅度只有 identity 的 1/3~1/2。
4. 训练期加的三种损失（lf/hf/bandw）都没能在 val 上超过 base（0.0481），但 lf/hf/bandw 的 test fixed1 更好（0.0208~0.0212 vs 0.0216）。

## 11. 四个骨架创新完整结果（全部评测完）

| 模型 | val fixed1 | test fixed1 | val vs base | test vs base | test 源校Δ | test 自校Δ |
|---|---|---|---|---|---|---|
| base | 0.02329 | 0.02160 | — | — | −3.5% | −6.1% |
| full（三合一） | 0.02602 | 0.02334 | +11.7% | +8.1% | −8.4% | −13.9% |
| delta（差分编码） | 0.02522 | **0.02063** | +8.3% | **−4.5%** | −1.9% | −7.3% |
| gate（时间门控） | 0.02488 | 0.02123 | +6.8% | −1.7% | −2.6% | −8.1% |
| reinject（多尺度重注入） | 0.02559 | 0.02160 | +9.9% | 0.0% | −3.0% | −8.8% |

**核心发现（四个骨架无一例外）**：所有骨架改动都在 val 上退化（+6.8~+11.7%），但 delta/gate 在洪水 test 上改善。
这与此前 4 个训练损失变体（lf/hf/bandw）的规律完全一致 → **"精度-泛化结构性 trade-off"** 是跨 9 个独立实验的稳定现象，是 ICIP 论文的核心贡献点。

**按预注册判据判定**：full/delta/gate/reinject 均未过主判据（val 必须改善），全部淘汰为"主方法"；但作为机制证据全部保留。

## 12. 正在跑（最后一个实验）
delsup：在 base 骨架上加协议自带的 ΔF 低频时间监督（a=0.54 来自诊断）。
- ep1 val 0.0606（base ep1 0.0611）
- ep3 val 0.0513（base ep3 0.0701）→ **收敛显著更快**，有望成为第一个 val 正收益的创新
