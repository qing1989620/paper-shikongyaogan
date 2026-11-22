# 遥感时空融合_文献深读总表 — 结构化摘要

来源：D:\qing_zhuomian_\科研区\时空融合\遥感时空融合_文献深读总表.xlsx（1 sheet「文献深读总表」，A1:U6，表头1行+5篇论文，21列）

## 列结构（A→U）
A 题目 | B 年份/会议期刊 | C 核心问题 | D 核心发现 | E 核心假设 | F 研究思路 | G 输入→输出 | H 方法与核心模块 | I 损失函数 | J 数据集与实验设计 | K 主要实验结果 | L 消融说明 | M 主要结论 | N 亮点 | O 实验局限 | P 方法局限 | Q 发表当年新意 | R 2026年不能再当创新 | S 创新启发 | T 最值得深挖问题 | U 一句话概括

## 逐篇要点

### P1 DeepFDP（ICASSP 2025）
- 协议：{F0 fine, C1 coarse} → F1，无 C0，2 图范式。
- 核心：粗细分辨率 token 频率分布 low intra/high inter variance 跨域稳定 → frequency degradation prior；FPM（FFT + 可学习线性插值 α + 幅度/相位卷积 + IFFT）+ HFM（多尺度高频增强）+ HM；L1 + Focal Frequency Loss λ=0.5。
- 实验：LGC 10训/4测；CIA 5 对 zero-shot；指标 MAE/RMSE/SAM/SSIM；LGC in-domain SSIM 0.9345 / MAE 0.0125；LGC→CIA SSIM 0.8464 / RMSE 0.0328；~1.1M 参数。
- 局限：无 C0、时间建模弱、突变区旧纹理失效、频率先验理论弱、跨域仅 LGC→CIA、baseline 偏老。

### P2 GSINet（ICASSP 2025）
- 协议：{C1, F0} → F1，无 C0；6 波段固定拆 RGB / NIR+MIR1+MIR2。
- 核心：SEFM（learnable group token 无监督语义聚类 + cross-attention，用预测时刻 coarse 语义增强历史 fine）+ ICM 双向交叉注意力；Charbonnier pixel loss + MS-SSIM struct loss。
- 实验：CIA 17 对（5测）/ LGC 14 对（4测）；CIA 全面最优，LGC 与 MLFF-GAN 接近但 SSIM 更高；~3.0M 参数。
- 局限：无 C0、固定分组仅两农田数据集验证、语义不变假设在剧变区缺证、时序仅 t1/t2。

### P3 2019 Remote Sensing 综述
- 无固定协议；核心结论：无单一方法全场景最优；渐变 vs 突变必须区分；异质景观/传感器/光谱/配准是联合困难；DL 当时应用少。

### P4 SA-STF（CVPR 2026）★最高撞题风险
- 协议：{F0,C0,C1,F2,C2} 5 图双锚点 → F1。
- 核心：Residual Diffusion + Noise Encoder + TFA（时间特征对齐）+ SAF（语义自适应变化判定 + 语义匹配高频迁移）+ Taylor-inspired De-Res；Lrec(L1)+Lper(VGG16)+Ltime；DDIM 50 步。
- 结果：CIA CC 0.8950/SSIM 0.8924/ERGAS 0.8910/RMSE 0.0218/SAM 0.0538；LGC 0.9360/0.9440/0.6751/0.0169/0.0531；AHB 0.8729/0.8962/0.8620/0.0283/0.0709，三数据集总体最佳。
- 局限：无 zero-shot cross-domain、无跨 sensor pair、unmatched abrupt change 难恢复、changed/unchanged 判定启发式、TFA 线性时间模型不足、需前后两个 reference、50 步推理成本高。

### P5 2023 Elsevier 综述
- 归纳 4 类协议（3图/5图/2图/时序）；核心缺口：distribution shift、unseen reflectance change、泛化/鲁棒/迁移、benchmark 单一、物理约束不足、效率。

## 与 TCCR/Stage4 最相关方法
| 相关度 | 论文 | 相关点 |
|---|---|---|
| ★★★ | SA-STF CVPR2026 | TFA 时间特征对齐、Taylor 时间投影、coarse 约束 residual diffusion、SAF 变化感知迁移——与 temporal projection+自校准几乎同一问题空间 |
| ★★☆ | GSINet | SEFM 用预测时刻 coarse 语义增强历史 fine，本质是 coarse 引导历史细节校正 |
| ★★☆ | DeepFDP | FPM 学 coarse→fine 频率映射；突变区旧纹理失效问题意识 |
| ★☆☆ | 两篇综述 | generalization/uncertainty/arbitrary-time/physical prior 问题源 |

## 撞题风险点
1. 时间投影/对齐：与 SA-STF 的 TFA + Taylor 投影 + SAF + coarse 约束 diffusion 高度重合——最大撞题点。差异化：test-time 逐样本自校准、冻结 backbone、无 prediction-date GT、1 NFE。
2. coarse 引导历史细节校正：GSINet SEFM 已做，需强调自监督/物理约束/逐样本。
3. 频域投影：若转频域会撞 DeepFDP FPM。
4. change-aware/semantic-adaptive 迁移：SA-STF、GSINet 已覆盖，不宜单独宣称。
5. 泛化主张：DeepFDP 已占"频率先验→跨域泛化"。

## 尚未被占的差异化落点
unseen/novel change 重建、uncertainty-aware、arbitrary-time/irregular observation、cross-sensor+cross-region 泛化、physical sensor degradation + generative、self-supervised/test-time adaptation、可解释 invariant representation、few-step 高效生成、change-conditioned frequency prior。
