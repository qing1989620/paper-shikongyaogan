# 从诊断出发的网络骨架创新（2026-11-18）

## 1. 诊断结论（问题定义）

```
GT fine 时间变化 vs coarse 时间变化:  a ≈ 0.54,  R² ≈ 0.28
网络预测变化 vs coarse:             a ≈ 0.47-0.57, R² ≈ 0.20-0.40
```

**两个事实**：
1. **a≈0.54 是物理事实**：粗分辨率像元混合导致时间变化幅度被平滑，fine 变化 ≈ 0.54 × coarse 变化。网络已学到这一点（a≈0.47-0.57），所以"网络没学好时间变化"是错的。
2. **R²≈0.28 是信息瓶颈**：coarse 只解释了 28% 的 fine 低频时间变化。剩下 72% 需要空间细节先验 + 两期 fine reference 推断。

## 2. 骨架的结构缺陷（从 forward 代码逐行分析）

现有 `BiPredNoiseNet`：
```python
x_fine   = conv(cat(F_prev, F_next))           # 两期 fine，无 temporal difference
x_coarse = conv(cat(C_prev, C_pred, C_next))   # 三期 coarse，无 temporal difference
x_clean  = conv(cat(x_fine, x_coarse))         # 1×1 压缩成一个向量
noise    = x_noisy - x_clean                   # 相减 → U-Net 只见到残差
```
三处硬伤：
- **(A) 输入层无显式时空分解**：三个 coarse 与两个 fine 只是 concat，1×1 conv 压扁。网络必须自己从 18 个通道里"发现" ΔC 和 ΔF，样本效率极低。
- **(B) 条件-噪声不对称**：t 只注入 noisy 分支，clean 分支与 t 无关。扩散早期（高噪声）应弱信 fine reference，后期（低噪声）应强信 —— 现有结构无法做到。
- **(C) 无跨尺度条件重注入**：条件只在入口出现一次，U-Net 深层只看 `noise=x_noisy-x_clean`，原始 F_prev/C_prev/C_pred 信息在深层衰减。

## 3. 三个骨架创新（全部从上述缺陷出发，可单独/组合训练）

### 创新 I：Decoupled Temporal Delta Encoder（解决 A）

**问题**：coarse 与 fine 的 temporal difference 是最干净的信号，却被埋在 concat 里。

**改法**：在入口增加一条显式时间差分通路：
```
ΔC_1 = C_pred - C_prev,   ΔC_2 = C_next - C_pred
ΔF   = F_next - F_prev
z_delta = Conv(cat(ΔC_1, ΔC_2, ΔF))       # 显式差分特征
x_clean = Conv(cat(x_fine, x_coarse, z_delta))
```
- 参数增量：~0.15M（一个 3×3 conv + 1×1 融合）
- 假设：给网络"算好的"差分，样本效率提升，尤其在小训练集上。

### 创新 II：Time-Conditioned Adaptive Condition Gate（解决 B）

**问题**：clean 条件与噪声级别 t 无关。

**改法**：用时间嵌入生成一个逐通道门控，控制条件强度：
```
t_emb = time_mlp(t)
g_t   = sigmoid(W_g · t_emb)              # (C,) 或 (C,1,1)
x_clean_eff = g_t * x_clean               # 高噪声时 g_t 小，低噪声时大
noise = x_noisy - x_clean_eff
```
- 参数增量：~0.03M
- 这是" άτις adaptive skip"在扩散 STF 的首次应用：让 backbone 自己学会"何时相信 reference"。
- **与推理期 identity 投影的关系**：identity 投影是"无条件信 coarse change"，而门控是"按噪声级别学信多少"——把后处理规则内化进网络。

### 创新 III：Multi-Scale Condition Re-injection（解决 C）

**问题**：条件在入口被压扁，深层丢失。

**改法**：在 U-Net 每个下采样尺度，把对应尺度的条件特征拼回 noisy 分支：
```
# 原: noise = x_noisy - x_clean; 各尺度只传 noise
# 新: 每个尺度额外注入 condition residual
for i, (dim_in, dim_out) in enumerate(in_out):
    cond_i = cond_downsample_i(x_clean)           # 与 noisy 同尺度的条件
    noise = noisy_res(torch.cat((noise, cond_i), dim=1), t)   # 或在 ResBlock 前加
```
- 参数增量：每尺度一个 1×1 conv，共 ~0.1M
- 假设：深层保留 coarse 空间结构，缓解 R²≈0.28 的信息瓶颈。

## 4. 训练策略创新（与骨架配套，不是加 loss）

### 策略 A：Delta-Supervised Pretext（用合法监督，不加 GT）
在训练时加一个辅助 head，从条件特征预测 ΔC 对应的 fine 变化方向，用 **F_next - F_prev 自身**做监督（双锚点协议自带）：
```
L_delta = || Head(x_clean) - LP(F_next - F_prev) ||_1
```
- 这不是"加 loss 凑数"，而是**强制 x_clean 编码真实时间变化**，直接针对 R²=0.28 的瓶颈。
- 与 Stage1 失败的频域先验不同：这里监督信号是协议自带的 fine reference，不是从 coarse 猜高频。

### 策略 B：Curriculum over noise levels
现有训练均匀采样 t。改为：早期 epoch 偏重高 t（学结构），后期偏重低 t（学细节）。
- 与创新 II 天然配合：门控在课程中学会从"弱信"到"强信"。

## 5. 实验矩阵（每个都跑 40 epoch，同数据同协议）

| 实验 | 骨架改动 | 训练策略 | 预期判定 |
|---|---|---|---|
| base | 无 | 标准 | 参照（已有 val 0.0233） |
| I | Δ encoder | 标准 | val 改善 >1% 则 keep |
| II | 时间门控 | 标准 | val 改善 + 深层可视化证实门控随 t 变化 |
| III | 多尺度重注入 | 标准 | val 改善 >1% |
| I+II | 组合 | 标准 | 看是否叠加 |
| I+III+A | 全改 + delta 监督 | 标准+pretext | 最强候选 |
| 最佳 + B | 最佳骨架 | 课程采样 | 最终 |

**成功判据**（预注册）：
- 主判据：在 LGC val 上相对 base 的 RMSE 改善，配对 Wilcoxon p<0.01 且 bootstrap CI 不跨 0。
- 泛化判据：LGC test（洪水）、CIA val/test 不退化，且至少两个 split 为正。
- 效率判据：参数增量 <0.3M，NFE 仍为 1。

**失败预案**：若 I/II/III 单独都 <1%，说明骨架容量不是瓶颈（数据/协议是），则停止魔改，转向"conditional identity projection"（用 ΔC 的可靠度动态决定投影强度），并如实写 limitation。

## 6. 实现与验证状态（2026-11-18 12:00）

### 已实现
- `repos/STF/src/model/stfdiff/model6/bi_pred_resnet_v2.py`：
  - 三个创新全部做成独立开关 `delta_enc / time_gate / reinject`
  - **全关时与原始 BiPredNoiseNet 权重逐位兼容，输出 max abs diff = 0.0**
  - 参数增量：delta +0.0145M，gate +0.0164M，reinject +0.0M（仅重排通道），full ≈ +0.031M（<0.5%）
  - 四种组合（delta/gate/reinject/full）前向与训练（1 epoch smoke）全部通过
- `tools/train_stfdiff.py` 增加 `--arch {base,delta,gate,reinject,full}`
- 链式训练 `tools/run_arch_chain.sh` 已启动，顺序 full → delta → gate → reinject，每个 40 epoch

### 预注册成功判据（跑之前写死）
1. **主判据**：LGC val RMSE 相对 base（0.0481 raw）改善，配对 Wilcoxon p<0.01 且 bootstrap CI 不跨 0。
2. **泛化判据**：LGC test（洪水）、CIA val/test 至少两个 split 为正，其余不退化超过 1%。
3. **效率判据**：参数增量 <0.5M，仍为 1 NFE。
4. **机制判据（针对创新 II）**：gate 值必须随 t 单调变化（高 t 低信任），否则 gate 退化为普通缩放，判定无效。

### 失败预案（如实登记，不硬凑）
- 若 full 单独都 <1% 改善：说明 1200 patch 的规模下骨架容量不是瓶颈，问题在数据多样性。此时停止魔改，把诊断结论（a≈0.54、R²≈0.28）写成 limitation，并转向"conditional identity projection"（用 ΔC 幅值动态决定投影强度）。
- 若某创新在 val 改善但 test 退化：按预注册判据 2 淘汰，不选择性报告。

## 7. 诊断发现的方法论意义（可直接写入论文 Discussion）

> We measure the relation between fine- and coarse-resolution temporal changes in LGC. The optimal affine slope is consistently ≈0.54 (GT) and ≈0.47–0.57 (network prediction), and coarse change explains only R²≈0.28 of fine low-frequency change. This shows that (i) the ≈0.5 attenuation is a **sensor MTF/resolution fact**, not a network bug — the backbone already learns it; and (ii) the majority of fine temporal change is **not observable** from coarse data, which is why inference-time projection has a hard ceiling. Consequently, improving fusion requires either better spatial priors or a backbone that explicitly represents temporal differences at multiple scales — which motivates our architectural changes rather than another post-hoc correction.
