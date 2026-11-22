# 邻近方向文献调研：为 RSP（reference-pair 闭式自校准 + 首步输出投影）找可借鉴机制

调研日期：2026-11-18
时间窗：2026-01 ~ 2026-11（个别条目回溯到 2025 下半年，均标注）
检索渠道：WebSearch（英文/中文多组关键词）、IEEE Xplore 页面、CVF Open Access / cvpr.thecvf.com virtual、arXiv 摘要页、PMLR、MDPI、ScienceDirect、Google Scholar 条目、项目主页
用途：为 RSP 创新点寻找"邻近但不同"的机制来源；同时做撞题排查。

## 0. 阅读前必读：核验等级说明

| 标记 | 含义 |
|---|---|
| ✅ | 已在**出版方/官方**页面确认（IEEE Xplore、CVF virtual、PMLR、MDPI、arXiv abs、项目主页） |
| 🟡 | 已确认标题+摘要+年份，但**未拿到一手指纹**（DOI 或官方 PDF）；引用前需补 |
| ⚠️ | 仅见于聚合/解读站（graphastra、lacuna、chatpaper、alphaxiv 等），**元数据未经一手核实**，只能当线索，不可直接进参考文献 |

> 2026 年文献大量只存在于 arXiv/预印本或会议 virtual 页，部分期刊条目（TGRS Early Access）尚无卷期页码。凡标 🟡/⚠️ 的条目，**投稿前必须逐条抓官方 PDF 核对作者序、venue、DOI**。本文件刻意保留"未检出"与"不确定"，不做凑数。

---

## 1. 扩散少步/单步采样与轨迹优化

- **Dual-End Consistency Model (DE-CM)** — Linwei Dong, Ruoyu Guo, Ge Bai, Zehuan Yuan, Yawei Luo, Changqing Zou · arXiv 2602.10764 · 2026-06 ✅
  机制：指出 CM 训练不稳来自自监督项导致的 loss 发散、采样不灵活来自首步误差累积；把 PF-ODE 轨迹切成三个关键子轨迹簇分别做连续时间 CM 蒸馏，并用 flow matching 作边界正则项稳定训练，另提"noise-to-noisy (N2N)"映射让任意点都能映射，专门缓解**首步误差累积**。ImageNet 256 一步 FID 1.70。
  链接：https://arxiv.org/abs/2602.10764
  → 对我们的意义：它把"首步误差"当成一等公民来处理，正好是 RSP 的作用位置；可借其"首步误差主导"的论述为自己的动机站台。

- **Trajectory-Regularized Distribution Distillation (TRD)** — An Zhao 等 · arXiv 2611.12978 · 2026-11-08 ✅
  机制：指出纯轨迹对齐在极小采样预算下退化、纯分布匹配又需额外网络估 score；提出"动量混合策略"在一次学生前向中同时充当一步生成器、学生边缘速度估计器、以及沿教师 ODE 的**闭式时变速度场**，从而去掉第二个可训练网络。
  链接：https://arxiv.org/abs/2611.12978
  → 对我们的意义："把某个量写成闭式速度场/闭式解"是 2026 的主流叙事之一，RSP 的闭式仿射可以套进同一套修辞（closed-form 而非 learned）。

- **FlowSR: Fast Image SR via Consistency Rectified Flow** — arXiv 2605.12377 · 2026 ✅（代码 github.com/jiaqixuac/FlowSR）
  机制：把 SR 重构成 LR→HR 的 rectified flow，在一致性蒸馏中加入 **HR 正则**项，使学到的流不仅自一致而且精确收敛到 GT；再用"快-慢双调度器"采样相邻时间步（快调度器省步数、慢调度器抓细纹理），实现单步 SR。
  链接：https://arxiv.org/abs/2605.12377
  → 对我们的意义："单步 + 一个额外正则把输出拉回真值"与 RSP"单步 + 一个投影把输出拉回参考一致"结构同构，可直接借其消融设计（有/无正则）。

- **ScaleResfusion: Residual Rectified Flow** — Zhenning Shi, Chen Xu, Junhao Zhang, Kefei Zhang, Linjie Liu, Zhedong Zheng, Tao Li · arXiv 2607.25275 · 2026-09-10 ✅
  机制：把残差项 R 插进 rectified flow 的线性传输路径，使采样从"带噪 LQ"这一**精确加速点**出发，起点 SNR 由残差比例 γ 连续控制；由此得到的"残差向量场"不含 scheduler 系数，与预训练 RF 目标只差一个 γR 偏移，因此适配冻结的十亿级 backbone 只需 **LoRA 拟合这个紧致残差修正**，再蒸馏到 4 步。
  链接：https://arxiv.org/abs/2607.25275
  → 对我们的意义：这是"冻结 backbone + 只学/只解一个紧致残差"的最强同构工作；它用 LoRA，我们用闭式，正好形成"零参数 vs 极少参数"的对照，也是我们必须引用的近邻。

- **Acceleration-Aware Sampling (A²S) for Few-Step Rectified Flow** · 2026 ⚠️（仅聚合站，作者/venue 待核）
  机制：指出 Euler 采样器假设分段常速、忽略加速度，导致少步下离散误差主导；从二阶视角把加速度分解为时间项（时间平移的速度评估）与空间项（随时间的光滑增益调制步长），在**同样一次前向**的成本下补偿二者，无需重训。
  链接：https://lacuna.tiptreesystems.com/work/acceleration-aware-sampling-for-few-step-rectified-flow

---

## 2. Test-time adaptation / test-time training / 测试期自校准

- **DCTTA: Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration** — Ni Tang, Shenghao Nie, Xiaotong Luo, Yuan Xie, Yanyun Qu · CVPR 2026（poster 37745）✅
  机制：核心假设是"同一场景的多个退化版本应映射到同一张 clean 图"。三步：(1) 测试期用 RDDM 在线学出 ȳ→x_in 的退化过程并再退化出 x_sd，构造伪 paired 监督；(2) 学生网络对 x_in 与 x_sd 施加像素+感知一致性损失做域适应；(3) 用 Fisher 信息矩阵对角近似挑出"退化敏感参数"只更新它们、冻结关键参数，防止灾难遗忘。Rain100H +4.57 dB。
  链接：https://cvpr.thecvf.com/virtual/2026/poster/37745
  → 对我们的意义：**"用样本自身构造伪配对做一致性自监督"这个论证结构可以借**；但它是"更新少量参数 + 多步"，RSP 是"不更新参数 + 闭式 + 1 NFE"，差异清晰且对我们有利。

- **Uncertainty-Aware Test-Time Adaptation for Cross-Region Spatio-Temporal Fusion of LST** — IGARSS 2026 · DOI 10.1109/IGARSS60577.2026.11730220 ✅
  机制：只更新预训练 STF 模型的**融合模块**，更新信号由三部分组成：认知不确定性、LULC 一致性、偏差校正；不需要源数据与目标域标签，10 个 TTA epoch。Orléans 训练 → Rome/Cairo/Madrid/Montpellier 测试，RMSE/MAE 平均改善 24.2%/27.9%。
  链接：https://xplorestaging.ieee.org/document/11730220
  → 对我们的意义：**这是"STF + test-time"最直接占位者**。它是跨区域域偏移 + 更新参数 + 多 epoch；RSP 是同协议内逐样本、零更新、1 NFE。审稿人问"你和这篇什么关系"必须有备答。

- **PRISM: Prototype-Rectified Iterative Self-supervised Manifold Denoising** — arXiv 2608.15037 · 2026 ✅
  机制：提出"Affine Noise Hypothesis"——严重噪声在潜空间引起**低秩仿射偏移**，>90% 畸变能量集中在前 60 个主成分；于是用冻结的文本原型作几何锚，做三种**闭式几何校正**，再由"Affine Bias Regression"编译成**单一静态投影矩阵**，推理时只需一次矩阵-向量乘（0.0009 ms），零训练、source-free。
  链接：https://arxiv.org/abs/2608.15037
  → 对我们的意义：**机制上最像 RSP 的一篇**（闭式仿射 + 编译成单次投影 + 零训练 + 极低开销）。必须引用并明确区分：它是**音频-文本潜空间**、**分布偏移校正**、**无时间维**；RSP 是**像素/低频残差**、**时间仿射（coarse→fine）**、**用 reference pairs 定标**。

- **TAO: Test-Time Degradation Adaptation** — ICML 2024（PMLR v235:16167-16177）· 2024（时间窗外，作谱系引用）✅
  机制：冻结的退化无关 DDPM + 可学习的 TDA 适配器（测试期梯度更新）+ AIR 引导，逐去噪步执行；面向开集退化。
  链接：https://github.com/XLearning-SCU/2024-ICML-TAO
  → 对我们的意义：反例。它**更新参数**，且"test-time adaptation"一词被它占据，RSP 措辞必须写成 parameter-free test-time **self-calibration**。

---

## 3. 扩散逆问题的数据一致性投影与引导

- **DCDP: Decoupled Data Consistency via Diffusion Purification** — Xiang Li, Soo Min Kwon, Shijun Liang, Ismail Alkhouri, Saiprasad Ravishankar, Qing Qu · IEEE ICCP 2026, pp.1-15 ✅
  机制：把数据一致性从反向采样中**解耦**出来，改成两阶段交替：(i) 重建阶段强制 data consistency，(ii) 精化阶段用 diffusion purification 施加生成先验。因为不再需要每步算似然梯度，可兼容任意加速采样器与 consistency model，并适用于像素/潜空间。
  链接：https://www.computer.org/csdl/proceedings-article/iccp/2026/11668875/2jqtA6F1mGQ
  → 对我们的意义：**"把投影从采样轨迹里拆出来、只做一次/极少数次"正是 RSP 的结构**。DCDP 是我们的方法论祖先，必须引，并指出它的 A 是已知物理算子、我们的"算子"由样本自带 reference pairs 闭式拟合。

- **Exact and Calibrated Diffusion Reconstruction for Digital Breast Tomosynthesis** — arXiv 2607.12937 · 2026-07-14 🟡
  机制：把逐点 proximal 更新换成**精确欧氏投影到数据一致集**，用 m 维对偶系统 + 一次性 Gram 矩阵 AAᵀ 分解实现，每步投影仅 4.5 ms（248× 加速），残差降到 2.4e-13；证明该投影是 proximal 步的 ρ→0 极限并给出 no-harm 定理；还实证"投影放在 update 之后会掉点"，即**投影位置是决定性的**；再用 isotonic recalibration 把 ensemble 离散度校准到误差尺度（ECE 0.029→0.008）。
  链接：https://mlsanity.com/paper/2607.12937
  → 对我们的意义：**三个可直接借用的东西**：(a) "一次性 Gram 分解 → 之后每步只做廉价投影"的计算模板；(b) "投影位置敏感"的实验结论支持我们在首步输出后立即投影；(c) 投影 + 不确定性校准的组合，可为我们加一个几乎零成本的置信图。

- **Generative Spectrum Cartography (GSC)** — Yuntong Gu, Xiangming Meng, Zhiyuan Lin, Sheng Wu, Linling Kuang · arXiv 2512.20108 · 2026-06 ✅
  机制：把稀疏/量化观测下的图重建写成贝叶斯逆问题，对线性与量化两类测量模型分别推出**闭式后验均值更新**，把这些更新嵌进反向扩散，得到**无梯度、测量一致**的后验采样，避免昂贵的似然梯度引导；后验样本再用于空间不确定度估计与主动选点。
  链接：https://arxiv.org/abs/2512.20108
  → 对我们的意义："闭式后验均值 + 免梯度"是比"投影"更温和的表述，若审稿人觉得"projection"太硬，可改称 closed-form posterior-mean correction。

- **Getting Your Guidance Weights Right in Diffusion and Flow-Matching Posterior Sampling** — L. Moroy, J.-F. Giovannelli, Y. Altmann, S. McLaughlin, F. Champagnat, G. Bourmaud · arXiv 2610.03503 · 2026-10 ✅
  机制：观察到条件去噪 score-matching（或条件 flow-matching）目标在每步都是**最小二乘**，故当条件预测写成"无条件网络输出 + 测量引导项"的加权和时，最优权重退化成一个**二维线性最小二乘**；可离线用一个小 batch 的采样轨迹解出时变权重，无需重训；顺带让采样从 1000 步降到 50 步不掉点。
  链接：https://arxiv.org/abs/2610.03503
  → 对我们的意义：**这是"把调参变成闭式最小二乘"最干净的范式**，和 RSP 的 12 标量最小二乘同源。可直接借它的话术："我们不是发明新网络，而是把一个原本靠手调的标量组变成可解析求解的量"。

- **PC-DAPS: Prior-Consistent Decoupled Annealing Posterior Sampling** — Merve Kocabas, Benjamin A. Burns, Andreas Geiger, Kent Gauen, Sara Fridovich-Keil · arXiv 2610.31592 · 2026-10-22 ✅
  机制：针对"测量一致但先验不支持"的模糊非线性逆问题，在一次采样内维持多个候选解，随退火噪声下降按**先验一致性**剪枝/替换弱候选，无需 GT；减少 prior-inconsistent 样本。
  链接：https://arxiv.org/abs/2610.31592
  → 对我们的意义：如果 RSP 后期想扩到"多候选 + 按参考一致性选优"，这是现成的、无 GT 的选优准则。

---

## 4. 频域/小波/多尺度先验在复原与融合中的应用

- **HDW-SR: High-Frequency Guided Diffusion based on Wavelet Decomposition** — Chao Yang, Boqian Zhang, Jinghao Xu, Guang Jiang（西电）· CVPR 2026 · arXiv 2511.13175 ✅
  机制：**只在残差图上做扩散**，让网络专注高频；用**小波下采样替代 CNN 下采样**做多尺度频率分解，并在"预超分图的高频子带"与"扩散图低频子带"之间做稀疏 cross-attention 实现显式高频引导；动态阈值块（DTB）筛选高频；上采样利用小波逆变换实现低损重建。
  链接：https://cvpr.thecvf.com/virtual/2026/poster/39527
  → 对我们的意义：**"低频给定、只扩散高频残差"** 与 RSP"低频由参考对校正、高频交给扩散"是同一频率分工哲学，是我们最好的引用背书。

- **WaveFreqAnchor: Wave-Structural Anchoring and Frequency Correction Diffusion（training-free 人脸复原）** · 2026 ⚠️（仅 chatpaper，需核）
  机制：ASWG 用各向异性波响应一致性约束结构；MWFI 用多尺度小波-傅里叶注入，**替换预测低频子带的相位**使之与观测对齐，从而校正反向扩散中累积的不一致；SHE 对预测高频子带有界、空间掩蔽地细化。
  链接：https://chatpaper.com/zh-CN/chatpaper/paper/323896
  → 对我们的意义：如果 RSP 想在频域表述，**"替换低频子带相位/幅值使其与观测一致"**就是一个现成的、training-free 的投影写法，比空间域投影更贴合"低频时间残差"的说法。

- **WaFT: Wavelet-Factor Matrix Hybrid Transport Maps for Spatiotemporal Bayesian Fusion** — Jin Zhang, Shiyuan He, Lingxin Gongye, Jianhua Guo · IEEE TGRS, Vol.64, Art.4417421 · 2026-09-07 · DOI 10.1109/TGRS.2026.3731164 ✅
  机制：把 HR patch 多级小波分解成**时间不变的平稳空间细节 SSD** 与**时间演化的大尺度结构 DLS**；DLS 用双线性低秩矩阵因子模型压到紧致潜子空间以保几何；在高斯共轭下得到**闭式 MAP 推断**，无需在 HR 像素空间迭代优化。
  链接：https://xplorestaging.ieee.org/document/11683240/authors
  → 对我们的意义：**撞题风险最高的一篇（方法学层面）**：同一数据集、同一年、同样是"逐样本闭式解 + 只改低频"。区别必须写死：它是**贝叶斯生成模型**（SSD 直接搬、不生成），无扩散、无首步输出、无投影算子；RSP 是**修正一个已训练扩散器的首步输出**。

- **WDCDiff** — Weiquan Kong, Yurong Qian, Lu Bai, Yuanxu Wang, Sen Luo, S.N. Tereshchenko · IEEE GRSL, Vol.23 · 2026 · DOI 10.1109/LGRS.2025.3635659 ✅
  机制：两阶段——先做**小波去噪**抑制高频噪声导致的过平滑，再交给条件扩散增强。
  链接：https://doi.org/10.1109/LGRS.2025.3635659
  → 对我们的意义：说明"小波前置 + 扩散"在 STF 已被做过；RSP 不能宣称"首次用小波"，只能宣称"低频残差的可解析投影"。

---

## 5. 不确定性量化与可信融合

- **Assessing Conformal Prediction for Remote Sensing: LCZ Classification（So2Sat LCZ42）** — Christoph Schweden, Peter Jung, Xiao Xiang Zhu · IEEE TGRS · 2026-08-04 🟡（DOI 待补）
  机制：在带 10 组独立专家标注的大规模土地覆盖数据上系统评估 conformal prediction 的覆盖率保证与预测集大小（后者作为不确定度直观指标），并考察标注者分歧对 exchangeability 假设的影响。
  链接：https://scholar.google.co.id/citations?citation_for_view=CNakdIgAAAAJ:Huv_iBJ06lEC
  → 对我们的意义：若 RSP 想给"投影修正后仍不确定的区域"一个无分布假设的置信区间，这是 RS 领域最合适的引文与协议模板。

- **Conformal Prediction for Uncertainty-Aware SAR-Based Terrain-Aided Navigation** — Simon Bertrand, Cornelia Vacar, Lionel Bombrun · FUSION 2026（Trondheim）· DOI 10.23919/FUSION67272.2026.11726608 ✅
  机制：把神经网络地形配准与 conformal prediction 结合，产生**有有限样本覆盖保证**的位置预测区域；区域小=高置信可入滤波，区域膨胀=应拒识，从而把不确定度直接接进多传感器融合管线。
  链接：https://xplorestaging.ieee.org/document/11726608/metrics
  → 对我们的意义：提供"预测集大小 → 是否采纳该修正"的**决策接口**，正好可用来决定 RSP 在哪些 patch 上启用投影。

- **TCC: Transported Conformal Calibration** — arXiv 2609.10737 · 2026（在 SEN12MS 上验证）✅
  机制：当标定数据只在源空间、预测集却要在目标空间时，用**成对无标注观测**（如配对模态/换传感器）把源标定"运输"到目标空间，再用无标签目标输入做残差校正（TCC-KS 保守调整 / weighted-TCC 重加权），给出随可观测失配度自适应的目标域覆盖保证。
  链接：https://weekinpapers.com/paper/2609.10737v1
  → 对我们的意义：**这是"用成对数据做跨域标定迁移"的严格版本**，和 RSP"用样本自带 reference pairs 定标"在数据协议上高度一致；可借其"配对无标注 + 有限样本保证"的论证来强化 RSP 的合法性。

- **AdaPCL: Evidential Uncertainty Modulated Adaptive Predictive Contrastive Learning** — Qiuyu Mei, Hong Yu, Shijie Yu, Yan Yang · ICMR 2026, pp.487-496 · DOI 10.1145/3805622.3810771 ✅
  机制：用 evidential deep learning 为每个模态给出**认知不确定度**，构成"预测置信度 + 证据可靠性"双视角，再合成**连续、逐样本的软权重**去调制跨模态对比目标（可靠对做对称对齐、可靠→不可靠做定向蒸馏、双不可靠则松弛）。
  链接：https://dl.acm.org/doi/full/10.1145/3805622.3810771
  → 对我们的意义：如果要做"按可靠性加权融合参考对"（比如某些 patch 参考已失效），这是零额外网络的加权方案模板。

---

## 6. 最优传输 / 直方图匹配 / 辐射归一化用于跨传感器域适应

> **说明**：本方向在 2026 年**没有检索到"纯 OT 用于跨传感器遥感域适应"的高质量新作**——搜到的 OT+RS 内容多为 2025 及以前的 UDA 综述章节、工程复现（Houston13→18 等）或博客。以下 4 篇是"同一问题（跨传感器/跨时相辐射一致性）但非 OT"的 2026 新作，机制上更贴近 RSP 需要的"逐样本线性定标"。

- **GCCM: A Multi-Stage Cascaded Framework for RRN of High-Resolution RS Imagery in Karst Regions** — Jian Wang, Qingyun Tang, Zhao Li, Weiping Jiang, Zhiyuan An · Geo-spatial Information Science · 2026-08-17 · DOI 10.1080/10095020.2026.2710937 ✅
  机制：三级级联——GMM 概率筛选候选不变区 → 双分支 CNN 学稳健空-谱表征 → 在学到的特征空间里用 CCA-MAD 精修不变区选择；针对喀斯特地形阴影/岩-植被光谱耦合导致的 PIF 提取困难。GF-1 长时序 R²=0.800、RMSE=0.007。
  链接：https://www.informahealthcare.com/doi/full/10.1080/10095020.2026.2710937
  → 对我们的意义：**"先选稳定子集、再在其上拟合线性关系"正是 RSP 该有的稳健化步骤**；可直接借它的 PIF 筛选思路做 RSP 的加权最小二乘（不改参数量，只改权重）。

- **Diffusion-Driven Relative Radiometric Normalization with Spatial–Spectral Attention Residual Network (SSARN)** — Song 等（中科院西安光机所/西安理工）· Remote Sensing 2026, 18(13), 2156 · DOI 10.3390/rs18132156 ✅
  机制：把辐射不一致建模为"确定性残差 + 随机扰动"，前向注入结构化噪声+随机噪声，反向用双目标变分同时恢复辐射一致性与保留细节；核心是空-谱双注意力残差网络；并用 SSIM 引导的预处理自动筛稳定伪不变区。Sentinel-2 多时相验证，NDVI 一致性与纹理保持均更优。
  链接：https://doi.org/10.3390/rs18132156
  → 对我们的意义：**"残差 + 噪声"分解与 RSP 的"低频时间残差"是同一分解视角**，可引作"辐射域已证明该分解有效"的旁证。

- **spectralmatch: relative radiometric normalization toolkit** — Lindiwe 等 · JOSS 11(117):8974 · 2026 · DOI 10.21105/joss.08974 ✅
  机制：global-to-local 匹配——先用**重叠区最小二乘**估计全局 scale/offset 把所有影像拉向共同光谱中心，再按块用块均值做局部微调；配套 PIF 自动提取、云/植被掩膜、免像素级配准。
  链接：https://doi.org/10.21105/joss.08974
  → 对我们的意义：**"重叠区最小二乘估 gain/offset、只两个数/波段、可记录可复现"** 是 RSP 的工程合法性来源；也提醒我们必须报告"每个样本解出的 12 个标量"作为可审计产物。

- **Physics-Constrained Transfer Learning with a Spectral-Fidelity-Preserving (SFP) Model** — Min Min, Jun Li 等 · Atmospheric Chemistry and Physics · 2026 ✅
  机制：不做 loss 惩罚，而是**把物理烘进训练数据的构造**：用 MODTRAN 在 1 cm⁻¹ 分辨率模拟 TOA 辐亮度，卷积两颗卫星的真实 SRF 得到"已编码光谱差异"的匹配对，再用从一阶起、R²<0.95 才升阶的**多项式拟合**做传感器间转换，系数天然满足辐射传输；并给出 SRF 相似度阈值（JSD≤0.3）。
  链接：https://www.x-mol.com/paper/2109450330265038855
  → 对我们的意义：**"用物理生成配对样本 → 多项式/线性拟合 → 闭式系数"** 是 RSP 的物理侧最强背书；也可借它的"相似度阈值"思想给 RSP 加一个"参考对可用性判据"。

---

## 7. 遥感领域的 Mamba / 状态空间模型与线性复杂度骨干

- **B-Mamba: Burn Severity Mapping from Large-Scale Landsat Using a Mamba-Based Siamese Network** — IEEE TGRS · 2026 · DOI 10.1109/TGRS.2026.3746053 ✅
  机制：Siamese Mamba 双分支提特征 + 层级 change fusion + 可学习 patch 扩张解码器，用 SSM 的线性复杂度做中分辨率（Landsat）长程依赖；Landsat-BSA 上 mIoU 79.31%。
  链接：https://xplorestaging.ieee.org/document/11753888/keywords
  → 对我们的意义：若审稿人质疑"冻结的 UNet backbone 太老"，这是"换骨干"的对照路线；但注意它换的是**架构**，不是**推理期校正**，二者正交，可写成 future work。

- **SAMamba: State Attention Mamba for HSI Classification** — Shuyu Ye, Bing Tu, Bo Liu, Yan He, Lianhui Liang, Peng Wang, Jun Li · IEEE TGRS Vol.64, Art.5518021 · 2026-06-15 ✅
  机制：不走"预定义扫描策略"老路，而是显式干预 SSM 的**序列构造**与**状态演化**两个阶段：空间分支用幅-相协同语义序列构造（APC），双分支异构状态空间。
  链接：https://scholar.google.co.cr/citations?citation_for_view=iMuSewsAAAAJ:nVrZBo8bIpAC
  → 对我们的意义：提示"扫描顺序=时间顺序"这一隐喻可迁移到时序融合（把 t 维当作扫描轴），但属结构性改造，与 RSP 不冲突。

- **CFAS-Mamba: Cross-Frequency Adaptive Synergistic Mamba** — Fulin Xu, Gen Feng, Mingyang Ma, Jiao Shi, Shaohui Mei · IEEE TGRS Early Access · 2026-10-13 · DOI 10.1109/TGRS.2026.3743144 ✅
  机制：针对 Mamba 强制 1D 序列化破坏 2D 拓扑的问题，用小波引导的各向异性序列化（WGAS）按频带物理属性给不同方向感受野，再用跨频协同注意力（CFSA）做高低频噪声鲁棒协同演化。
  链接：https://xplorestaging.ieee.org/document/11735200/metrics
  → 对我们的意义：**"频带分工 + 方向化"** 与 RSP 的低频/高频分工同源，可作为"频域分解在 SSM 里同样成立"的跨架构证据。

- **Vision Mamba in Remote Sensing: A Comprehensive Survey** — Muyi Bao, Shuchang Lyu, Zhaoyang Xu, Huiyu Zhou, Jinchang Ren, Shiming Xiang, Xiangtai Li, Guangliang Cheng · Remote Sensing 2026, 18(4), 594 · DOI 10.3390/rs18040594 ✅
  机制：综述约 120 篇 RS-Mamba 工作，提出微观（SSM 公式改良、扫描策略、多模态/双时相交互）与宏观（CNN/Transformer 混合、频域适配）两级分类，并做跨任务基准。
  链接：https://doi.org/10.3390/rs18040594
  → 对我们的意义：写 related work 时的单一入口，避免逐篇堆砌。

---

## 8. 参考图引导的超分/复原（reference-based SR、retrieval-augmented、memory bank）

- **DS-DiT: Decoupled Siamese Diffusion Transformer for Reference-Based RS Image SR** — Bin Luo, Runmin Dong, Zhaoyang Luo, Jinxiao Zhang, Jiyao Zhao, Fan Wei, Haohuan Fu · ECCV 2026, pp.263-281 · arXiv 2605.17980 ✅
  机制：在 attention 内部**解耦** LR 结构条件与 Ref 纹理条件，让二者各自独立地与噪声潜变量交互以缓解条件竞争；Patch-Level Weighting（PLW）做局部自适应融合（在参考与目标之间发生土地覆盖变化的区域尤其能防"鬼影"）；孪生结构支持推理期 **autoguidance**：用强/弱 Ref 条件的预测差做外推（ω>1 放大参考带来的高频），零额外训练。
  链接：https://arxiv.org/abs/2605.17980
  → 对我们的意义：**"强/弱条件差分 = 免费引导"** 与 RSP 的"参考对残差"是同一思想的两个化身（它在线性外推，我们在闭式投影）；且它同样冻结训练、推理期才起作用。

- **WorldSR: Harnessing World Knowledge Search for Grounded Image SR** — Junyuan Deng, Xinyi Wu, Zhenyao Wu, Song Wang, Yu Wang, Congchao Zhu · arXiv 2610.28423 · 2026-10-20 ✅
  机制：用 MLLM 把输入分解成实体并画 ROI，逐实体做网络+本地库检索、质量/语义多步筛选；memory bank 用 IoU 复用已检索结果；重建端把 LR 与多张参考编码进同一 token 序列，并用**两个独立 LoRA** 分别处理 LR token 与 Ref token（区分两种质量分布），实现多参考融合。
  链接：https://arxiv.org/abs/2610.28423
  → 对我们的意义：**memory bank + 逐实体参考** 是"参考选择"这一环的现成方案；RSP 的 reference pairs 是"时间近邻"而非"语义实体"，可写成"我们不需要检索，因为协议已给出锚点"这一简洁性优势。

- **GraftSR: Grafting Authentic Textures via Identical-Instance Guidance** — Qifan Yu, Haoran Bai, Zongyao He, Weijie He, Sibin Deng, Honggang Qi, Ying Chen（阿里/国科大）· arXiv 2608.25334 · 2026 ✅
  机制：用同实例参考图锚定真实纹理，但参考与输入存在严重空间错位；提出**双掩膜参考引导**——一个掩膜决定"从参考取什么纹理"、另一个决定"贴到哪里"，从而不依赖脆弱的空间对齐；并构建 TexRefSR-141K 数据集与 TexRefSR-Eval 基准，LPIPS 降 20.2%。
  链接：https://arxiv.org/abs/2608.25334
  → 对我们的意义：**"解耦 what-to-take 与 where-to-put"** 正是 RSP 中"逐像素仿射系数 a(i,j) 与偏移 b" 的两段式；可作为我们 12 标量参数化的可解释性论据。

- **3M-TI: RGB 参考引导热成像超分** — CVPR 2026（上交/复旦/清华）⚠️（仅二手报道，元数据待核）
  机制：免标定的跨模态联合注意力（CSM，零新增参数）+ 错位增强 + **一步 SD-Turbo** 潜扩散，仅用 LoRA 微调 UNet 与 VAE 解码器，使未配准、不同步的 RGB 也能稳定助力热图重建。
  链接：https://m.toutiao.com/article/7677903589863408182
  → 对我们的意义：**"零参数注意力 + 一步扩散 + 未配准参考"** 与我们设定几乎重合（只是任务不同），是"一步 + 未对齐参考也 work"的有力旁证。

---

## 9. 参数高效微调与适配器在遥感/复原中的测试期使用

- **DROS / MM-DROS: Domain-aware Relaxed Orthogonal Subspace adaptation** — Han Luo, Ruoyu Yang, Yinhe Liu, Yanfei Zhong · arXiv 2609.13654 · 2026-09-12 ✅
  机制：指出 LoRA 的固定低秩子空间在 RS 域偏移下"坐标系选错"（subspace mismatch）；改为**数据条件化的子空间学习**——用下游训练分布的二阶激活统计量条件化权重分解，使初始化反映真实特征几何，再用松弛正交参数化做柔性几何变换；多模态版跨模态共享变换结构。**无额外推理开销**，多处 RS 基准超全量微调。
  链接：https://arxiv.org/abs/2609.13654
  → 对我们的意义：**"二阶统计量闭式定标"** 与 RSP 的最小二乘定标同族；若未来不得不加极少参数，这是首选方案（但会破坏"零参数"卖点，需权衡）。

- **RSAC: Remote Sensing Amodal Completion（Dual-Adaptive Diffusion Framework）** — 武汉大学 · Journal of Remote Sensing · 2026-04-07 · DOI 10.34133/remotesensing.1035 ✅
  机制：把 SD 通过 **LoRA** 适配到遥感域，四通道 ControlNet 用图+掩膜引导结构补全；关键是 **prior-enhanced initialization**——用可见物体的低频信息初始化，而不是从纯高斯噪声开始，从而保物理一致性；IoU 0.853、SSIM 0.930。
  链接：https://doi.org/10.34133/remotesensing.1035
  → 对我们的意义：**"不从纯噪声起、而是用观测的低频初始化"** 与 RSP"低频由参考决定"完全同构，是我们最好的频率分工引文；差别在于它是初始化、我们是在首步之后做投影修正。

- **ScaleResfusion（见 §1）** — 冻结十亿级 RF backbone，仅用 LoRA 拟合残差向量场，4 步采样 ✅
  → 对我们的意义：**"LoRA 残差"是最小侵入的对照方案**；RSP 的卖点正是"连 LoRA 都不要，直接闭式"。

- **WorldSR（见 §8）** — 用两个独立 LoRA 分别处理 LR token 与 Ref token ✅
  → 对我们的意义：说明"给不同来源的 token 不同适配器"已成惯例；RSP 用"不同频率不同处理"替代"不同 LoRA"，是更轻的版本。

---

## 10. 变化检测引导的时序建模（change-aware gating、bi-temporal interaction）

- **DCAFNet: Difference-Gated Interaction and Change-Aware Cross-Temporal Fusion** — Lele Li, Panpan Zheng, Liejun Wang, Yuqing Zhou（新疆大学）· Remote Sensing 2026, 18(16), 2763 · DOI 10.3390/rs18162763 ✅
  机制：DGFI 用**双时相差分幅值生成变化相关门**，抑制光照/季节伪变化、保留真变化；CCTF 把粗预测当语义先验做特征重校准，并用跨时注意力 + 可学习自适应融合恢复小尺度变化；粗到细统一框架，四个基准 SOTA。
  链接：https://doi.org/10.3390/rs18162763
  → 对我们的意义：**"用 |F_t − F_ref| 幅值当门控"** 是 RSP 可借的逐像素权重来源——用差分幅值决定投影强度（变化大处少投影、稳定处强投影），完全不增加参数。

- **AdaDINO: Pair-Aware In-Backbone Adaptation of Frozen DINO for Efficient RS Change Detection** — Xu Zhang, Xinqing Li, Jianpeng Xie, Zeshuai Zhu, Xin He, Yun Liu · arXiv 2608.07982 · 2026-08-08 ✅
  机制：核心是 Change-aware Gated Local Adaptation（CGLA）：在选定的冻结 block 之后耦合两支流，并向两路**注入等幅、反号的时间残差**，从而增强真变化响应同时**严格保持 pair midpoint 不变**；另有 BSCS 按 batch 共享通道块削减 FFN 计算（去掉 62.5% 宽度仍 F1 85.29、1.41× 吞吐）。
  链接：https://arxiv.org/abs/2608.07982
  → 对我们的意义：**"反号注入 → 中点守恒"是一个可证明的、零参数的残差参数化**。RSP 完全可以借用这个构造来定义"时间残差"：把校正写成 F̂ = F + δ、并让 δ 在两对参考上对称抵消，天然给出"不引入整体偏置"的保证。

- **BSGA: Bi-Temporal Semantic Gated Adapter** — IEEE TGRS · 2026 · DOI 10.1109/TGRS.2026.3752873 ✅
  机制：即插即用模块。PGSA 用一组共享可学习语义原型重建双时相语义特征以提升跨时语义一致性；DSIG 以**双时相特征差为门控先验**、结合重建的语义上下文做特征调制，突出真变化、压制伪变化；可插入 DINOv3、SAM2。
  链接: https://xplorestaging.ieee.org/document/11793217
  → 对我们的意义：**"特征差 → 门控 → 调制"** 是 §10 的标准配方；RSP 用"参考对拟合出的仿射残差 → 投影"是解析版，二者可并列作为一种"gate vs project"的方法论对照。

- **MIGA-Net: Graph Interaction and Gated Context Network** — Jingtong Yang, Xiaorong Xue, Yishuo Tian, Wen Zhang, Bingyan Lu, Xin Zhao, Wancheng Wang · Remote Sensing 2026, 18(13), 2155 · DOI 10.3390/rs18132155 ✅
  机制：ST-GIM 结合交互注意力与图推理抑制光照/季节伪变化；AGCP 做内容驱动的尺度选择 + 门控残差注入；HBAR 分层边界感知精修。
  链接：https://doi.org/10.3390/rs18132155
  → 对我们的意义：偏工程，价值在于"伪变化抑制"的评价口径（F1/IoU 在 LEVIR-CD/WHU-CD/SYSU-CD 上的稳定性），可作为 RSP 在变化区域的定性对照参考。

---

## 11. 物理约束 / 辐射传输先验嵌入深度模型

- **AnyBand-Diff: Unified RS Image Generation and Band Repair with Spectral Priors** · 2026 ✅（多源一致，arXiv 号待补）
  机制：MCD 主干 + **双随机掩码**（训练时从 SRF 库采样光谱响应函数并随机丢像素）使模型能从任意波段子集恢复完整光谱；核心是 **Physics-Guided Sampling（PGS）**——用可微分物理前向模型（辐射传输、光谱指数）对含噪潜变量求"时变物理一致性损失"的梯度，作为**修正信号注入噪声估计**，主动把去噪轨迹推向物理可行流形；另有像素/区域/全局三级多尺度物理损失（含 PROSAIL 的 MLP 代理）。
  链接：https://chatpaper.com/paper/328852
  → 对我们的意义：**"物理模型梯度 → 注入首步噪声估计"** 与 RSP"参考对仿射 → 投影首步输出"是同一插入点、两种约束源（物理模型 vs 数据自校准）。这是最强的方法论并置：**它是 model-based prior，我们是 data-instantiated prior**。

- **PINA: Physics-Informed Neural Network for Aerosols（逐小时吸收性气溶胶反演）** — Geophysical Research Letters · 2026 · DOI 10.1029/2026GL122825 ✅
  机制：用 6SV 模拟训练一个轻量全连接"代理辐射传输模型（SRTM）"，再把 SRTM 与反演网络耦合，用**可训练的不确定度因子 σ 自适应加权物理损失与数据损失**；并以"清洁参考（±15 天最低 AOD）→ 污染观测"的 RANSAC 线性关系作为辐射一致性约束的物理依据。
  链接：https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026GL122825
  → 对我们的意义：**"用时间窗内的清洁参考构造配对 → 估线性关系 → 作为一致性约束"** 与 RSP 的双锚点 reference pairs 在逻辑上惊人一致（只是它用于气溶胶、且训练期）。这是"参考对定标"在物理反演领域的先例，务必引用以证明思路不是凭空。

- **Physics-Constrained TL with SFP Model（见 §6）** — Min Min, Jun Li 等 · ACP 2026 ✅
  机制：物理烘进数据构造 → SRF 卷积配对 → 多项式闭式拟合 → 系数天然满足辐射传输；给出 SRF 相似度可用阈值。
  → 对我们的意义：**"闭式拟合 + 物理可解释系数"** 的范式背书；可直接借"系数可解释"这一点写 RSP 的 12 标量。

- **ReMD（CVPR 2026，科学计算超分，守恒律约束，2-5 步）** ⚠️（仅 graphastra 汇总，需核）
  机制：把守恒律作为硬/软约束嵌入扩散，使流体场重建满足物理一致性，采样压到 2-5 步。
  链接：https://graphastra.com/en/summaries/20260712-%E4%BB%A5%E6%89%A9%E6%95%A3%E4%B8%BA%E4%B8%BB%E5%B9%B2
  → 对我们的意义：仅作"物理约束 + 少步"的趋势证据，不建议直接引。

---

## 12. 混合专家 / 路由 / 动态计算

- **DAMP: Degradation-Aware Metric Prompting for HSI Restoration** — Binfeng Wang, Di Wang, Haonan Guo, Ying Fu, Jing Zhang · ICML 2026, PMLR 306:130375-130394 ✅
  机制：从 25 个候选度量里三阶段筛出 6 个**可解释物理统计量**（高频能量比、空间纹理均匀性、光谱曲率均值/标准差、梯度标准差、空间相关系数）组成 6 维"退化提示"；DAMoE 用该向量当**路由先验**（而非视觉特征）做 softmax+top-k 稀疏选专家，训练时给门控 logits 加高斯噪声促负载均衡；专家是并行的空间（窗口 MHSA）/光谱（1D conv）双分支 SSAM。零样本泛化到未见退化。
  链接：https://proceedings.mlr.press/v306/wang26hr.html
  → 对我们的意义：**"用少数几个可解释标量做路由/调制"** 与 RSP"用 12 个可解释标量做投影"是同一种"极低维自适应"哲学；若要做"按样本难度决定投影强度"，这是现成的度量清单。

- **Patch Forcing: Denoising, Fast and Slow（难度感知自适应采样）** — Johannes Schusterbauer, Ming Gui, Yusong Li, Pingchuan Ma, Felix Krause, Björn Ommer（TUM/CompVis）· CVPR 2026 ✅
  机制：让不同 patch 走不同时间步，但用 **LTG 采样器控制"训练时可见的最大信息量"**（而非平均信息量）避免 context leakage 造成的训练-推理失配；加一个 <0.01% 参数的不确定性头预测每 patch 难度；推理时 dual-loop（自信区大步走、困难区小步细修）与 look-ahead（把自信区推到未来时间步当干净上下文）两种采样器。
  链接：https://compvis.github.io/patch-forcing/
  → 对我们的意义：**"不确定性 → 决定哪里多算/少算"** 是 RSP 唯一的"按需修正"扩展方向（且可做到零新增参数，用参考对残差的方差代替不确定性头）。

- **Leveraging Inference-Time Compute via Global Scheduling of Denoising Trajectories** — Yuan Cao, Yifu Tang, Hangqi Li, Zeyu Zheng · arXiv 2609.22867 · 2026-09-19 ✅
  机制：把"固定 NFE 预算如何在轨迹各步分配搜索次数"形式化为可分离凹整数规划，解具 water-filling 结构；证明敏感度跨样本变化时任何自适应策略都有线性 worst-case regret，故建议"离线锚定分配 + 在线只回收实例级松弛"；实测省 20-50% NFE。
  链接：https://arxiv.org/abs/2609.22867
  → 对我们的意义：**"离线锚定 + 在线微调"的理论依据**：RSP 的 12 标量投影强度可以全局锚定一个默认值、逐样本只做小幅闭式调整，正好落在它推荐的 regime。

- **Instance-Level Sampling Schedules via REINFORCE with James-Stein Shrinkage** — Google · CVPR 2026 ⚠️（仅聚合站）
  机制：用策略梯度为每个 prompt/噪声种子学个性化采样日程，并用 James-Stein 收缩压制 REINFORCE 在高维策略空间的梯度方差。
  链接：https://graphastra.com/en/summaries/20260803-%E5%AE%9E%E4%BE%8B
  → 对我们的意义：只在"逐实例自适应"这条线上作趋势引证，不必细引。

---

## 13. 专项 A：2026 年 STF / 遥感图像融合直接相关新论文

> 这一节是**任务同域**清单（不是邻近方向），用于 baseline 与 positioning。已尽量取官方页。

| 方法 | 出处（2026） | 机制要点 | 核验 |
|---|---|---|---|
| **SA-STF**（Semantic-Adaptive Diffusion） | CVPR 2026；Jinsong Zhang, Ying Qu, Yuan Liao, Hairong Qi, Zhenzhou Shao | 官方双锚点 {C1,F0,C0,F2,C2}→F1；RDDM 式残差扩散 + TFA（线性回归逐像素 a,b 做时间特征对齐）+ SAF 语义自适应 + Taylor-inspired De-Res；DDIM 50 步 | ✅ CVF + GitHub `ZZZhangjinsong/SA-STF` |
| **WaFT** | TGRS 64, Art.4417421；DOI 10.1109/TGRS.2026.3731164 | 小波分解 SSD/DLS + DLS 双线性低秩因子模型 + 高斯共轭下**闭式 MAP** | ✅ |
| **GAUSTF** | TGRS 64, Art.5631223；DOI 10.1109/TGRS.2026.3711642 | 假设地物反射率时间变化服从高斯，混合像元=多个正态分布的组合，STF 转为**目标级参数推断**；仅需一对 fine 图；代码 Zenodo | ✅ |
| **FSATS** | TGRS；2026-06-17；Yiting Wang, Chengfa Wang, Rui Zou, Jinghao Liu, Donghui Xie, Guangjian Yan | 自自适应趋势面，针对稀疏/时间不均 fine 观测下"时间相似性 + 线性变化"假设失效的异质农区 | 🟡 |
| **ArtFusion** | TGRS；2026-07-16；Dizhou Guo, Qianqian Jia, Xuan Wang, Yinze Lian, Zhenhong Li | 任意数量参考图；ECHB 对比感知混合块 + ETIE 显式时间编码（用获取元数据）+ MCAF 多头跨参考注意力 | 🟡 |
| **DCAF** | TGRS；2026-08-10；Li Shuai, Yinghong Jing, Xiaojun She, Yao Li | 双 cross-attention 做多源多场景互校准 + 自适应权重分配 + 层级融合；**只需两幅交叉配对影像**，不用三幅以上 | 🟡 |
| **STCAFormer** | Remote Sensing of Environment, 2026-11；DOI 10.1016/j.rse.2026.115594 | 双输入编码器 + swin-temporal fusion block（cross-attention 抓时间变化）+ 多尺度空间融合块 + 双路上采样；在**特征空间**而非像素空间做融合 | ✅ |
| **DPDiff** | TGRS；2026-02-13；DOI 10.1109/TGRS.2026.3664495 | PENet 学 SMP（同日 coarse↔fine 空间映射）+ TCP（跨时 coarse↔fine 时间变化先验），二者互逆；RDF 残差扩散；DAM 对齐 | ✅（前次审计已确认） |
| **STF-FlowDiff** | TGRS；2026-04-28；DOI 10.1109/TGRS.2026.3687438 | 在 reference fine 与"由 coarse 构造的 proxy fine"间估**稠密光流**作显式先验；GFC 选稳定区 + MFI 多尺度流注入 | ✅（前次审计） |
| **WDCDiff** | GRSL 23；DOI 10.1109/LGRS.2025.3635659 | 小波去噪 + 条件扩散两阶段 | ✅ |
| **SuperSTF** | ISPRS J. P&RS 235；DOI 10.1016/j.isprsjprs.2026.02.037 | all-in-one 去云 + STF；潜在扩散 + Swin 自编码器 + 云位置编码 + 获取日期调制；非双锚点协议 | ✅（前次审计） |
| **MSRS-CDM** | 《航天返回与遥感》2026(4)；DOI 10.3969/j.issn.1009-8518.2026.04.008 | 多源条件扩散；CT-SSG 跨时结构-光谱引导 + DMG-Fusion 双流互导；两图输入可工作 | ✅ |
| **STF-Mamba** | 2024/2025（被 SA-STF 列为对比基线） | Mamba 骨干做 STF | 🟡（仅作 baseline 名，未核 2026 版） |

**读法**：
1. **2026 年 STF 的方法论主线不是"更强生成"，而是"更强的先验结构"**：光流（STF-FlowDiff）、时间变化先验（DPDiff）、语义（SA-STF）、小波低秩（WaFT）、对象级高斯（GAUSTF）、任意参考（ArtFusion）。**"结构化的、可解释的先验"已经是主赛道**，RSP 正好落在这一主线上，但必须把"可解释"做到位（12 个标量 + 闭式解 + 可复现）。
2. **闭式解在 STF 已被 WaFT/GAUSTF 占住**（都是 model-based Bayesian）。RSP 不能说"首次在 STF 引入闭式解"，只能说"首次把闭式解用于**修正已训练扩散器的输出**"。
3. **"两图输入"已成共识**（MSRS-CDM、DCAF、GAUSTF 都强调少输入），RSP 的"两对 reference"不是劣势。
4. IGARSS 2026 的 STF 侧重点是**跨区域泛化**（见 §2 的 uncertainty-aware TTA）而非新融合算子；CVPR 2026 是**扩散为主**；TGRS 2026 是**model-based 复兴**（WaFT/GAUSTF）。这三条线合起来就是我们的 positioning 图。

---

## 14. 专项 B：撞题排查——是否已有"逐样本闭式校准 + 投影修正扩散输出"

**结论：未检出完全撞题。** 检索覆盖 12 组关键词 × 中英文、CVF virtual、IEEE Xplore、arXiv、PMLR、MDPI、Google Scholar。但需防守 5 个"部分占位"，按危险度排序：

| # | 占位者 | 它已经占了什么 | RSP 必须守住的差异（写进 intro 与 rebuttal） |
|---|---|---|---|
| 1 | **WaFT**（TGRS 2026） | STF 任务 + **逐样本闭式解** + 只改低频（DLS） + 小波分解 + 免迭代 | 它是**贝叶斯生成模型**（SSD 直接搬、DLS 由低秩因子模型推断），**不含扩散**、无"首步输出"、无"投影算子"；且不需要任何已训练网络。RSP 的前提是"已有一个扩散 STF 器"，改的是它的**输出**，不是重建图像本身。可写：*"WaFT replaces the generator; RSP repairs it."* |
| 2 | **PRISM**（arXiv 2608.15037） | **闭式仿射校正 + 编译成单一静态投影矩阵 + 零训练 + 单次 matvec** + 极低延迟 | 任务是音频-文本潜空间去噪、校的是**分布偏移**、无时间维、无参考对协议；RSP 在像素/低频残差、校的是**coarse→fine 时间仿射**、由样本自带 pairs 定标。这是"机制同构、问题不同"的典型，**必须主动引用并显式对比**，否则容易被指认抄袭结构。 |
| 3 | **SA-STF 的 TFA**（CVPR 2026） | 逐像素 coarse→fine 时间仿射 `HR0(i,j)=a_h(i,j)·F0(i,j)+b` | 它是 **CNN 学出的逐像素系数**、作用于**深层特征**、是 backbone 的一部分、需 50 步 DDIM。RSP 是**闭式最小二乘的 12 个标量**、作用于**扩散输出的低频时间残差**、冻结 backbone、1 NFE。 |
| 4 | **DPDiff 的 TCP**（TGRS 2026） | 明确写"cross-time coarse↔fine temporal change prior" | 它由 PENet **训练期学习**、作为**条件注入**（生成前）；RSP 是**生成后**的投影，且逐样本闭式。RSP 措辞避免把 12 标量叫 "learned prior"。 |
| 5 | **Uncertainty-aware TTA for STF-LST**（IGARSS 2026） | "STF + test-time" 这一组合，且只更新融合模块 | 它**更新参数、10 epoch、跨区域域偏移**；RSP **零更新、1 NFE、同协议内逐样本**。术语上用 "parameter-free test-time self-calibration"，不用裸 "adaptation"。 |

**次一级需注意（非撞题但会被拿来类比）**：
- **Guidance weights as 2D least squares**（arXiv 2610.03503）：也把"标量组"变成闭式最小二乘，但解的是**引导权重**、离线、所有样本共享；RSP 解的是**时间仿射**、逐样本、作用在输出上。可借其修辞，但要强调"per-sample"。
- **GSC**（arXiv 2512.20108）：闭式后验均值 + 免梯度，但测量算子已知；RSP 的"算子"来自参考对。
- **Exact projection for DBT**（arXiv 2607.12937）：精确投影 + 一次性 Gram 分解，但算子是已知物理前向模型；RSP 无显式 A。
- **AdaDINO 的 CGLA**（arXiv 2608.07982）：反号时间残差保中点，零参数、backbone 冻结——**结构上最接近 RSP 的"零参数时间残差"**，但用于 CD 分类头，不是扩散输出。
- **RSAC 的 prior-enhanced init**（J. Remote Sensing 2026）：低频来自观测、不从纯噪声起——与 RSP 频率分工同构，但它是**初始化**、我们是在首步后**投影**。

**一句话防守**：
> 现有工作分别在三个位置各占一段——**WaFT 占"STF 里的闭式解"**、**PRISM 占"闭式仿射编译成单次投影"**、**SA-STF/DPDiff 占"coarse→fine 时间先验"**；但没有任何一篇把三者拼在"**冻结扩散器的首步输出**"这一位置上。RSP 的新颖性是**位置（output-space, first-step）+ 约束来源（样本自带 reference pairs）+ 参数状态（零更新、零新增）**的三元组，而不是任何单项技术。

---

## 15. 对我们最有借鉴价值的 8 个机制（排序）

排序标准：(a) 能否在**冻结 backbone + 1 NFE + 零新增参数**下直接落地；(b) 能否为 RSP 提供**理论或实验背书**；(c) 与 RSP 的差异化是否清晰（不会被指认为它的变体）。

| 排名 | 机制 | 出处 | 为什么适合我们的设定 | 可落地动作 |
|---|---|---|---|---|
| **1** | **SSD/DLS 频率分离 + 只对时间演化分量做闭式估计** | WaFT (TGRS 2026) | 直接给我们"只改低频时间残差"的正当性；且它证明"闭式 + 免迭代"在 STF 是 SOTA 级做法。同时它是我们最强的对比对象，引用它反而是加分（显示我们清楚边界）。 | 在方法节明确写出：RSP 的投影只作用于低频带（可用一个固定高斯/小波低通），高频完全交给扩散器；报告"只投低频 vs 全带投影"的消融。 |
| **2** | **闭式仿射 → 编译成单一静态投影矩阵 → 推理时一次 matvec** | PRISM (arXiv 2608.15037) | 这正是 RSP 的"工程形态"模板：把 12 个标量的最小二乘解写成 `F ← P·F + t`，一次矩阵乘，成本可忽略。它已经把"零训练 + 极低延迟"这条路走通并被接受，我们可以直接借其成本论证。 | 把 RSP 实现写成"解 12 标量 → 组装 P,t → 一次性作用"，并在论文里报 wall-clock（应为 μs 级）。同时显式引用 PRISM 并说明"我们校的是时间而非分布"。 |
| **3** | **精确投影到数据一致集 + 一次性 Gram 分解 + "投影位置敏感"结论** | Exact & Calibrated Diffusion Reconstruction (arXiv 2607.12937) | 三件事全都要：(i) 理论上把"投影"包装成 proximal 的极限（我们的投影可写成同样的极限形式）；(ii) 计算上"一次分解、之后廉价"的模板；(iii) 实验上"投影放在 update 之后会掉点"直接支持"首步输出后立刻投影"。 | 在附录给一个命题：当仿射约束的权重 →∞ 时，RSP 的加权最小二乘退化为到仿射子集的欧氏投影；并复现它的"投影位置"消融（首步前/首步后/多步后）。 |
| **4** | **把"标量组调参"重写成低维线性最小二乘** | Guidance Weights (arXiv 2610.03503) | 给 RSP 一套现成的合法性叙事：不是启发式，而是"每步目标本身是最小二乘，故最优标量有解析解"。而且它证明"解对了权重能把 1000 步压到 50 步"，反衬我们 1 NFE 的合理性。 | 论文里用"two-stage: closed-form fit + single projection"的表述；并做"逐样本解 vs 全局解 vs 网格搜索"三组对照，证明逐样本闭式严格更优。 |
| **5** | **反号注入时间残差 → 严格保持 pair midpoint** | AdaDINO CGLA (arXiv 2608.07982) | 给"时间残差"一个**可证明的、零参数的**参数化：若把校正写成在 F0 与 F2 上分别 +δ 与 −δ，则中点 (F0+F2)/2 不变，天然避免整体偏置/色偏。这几乎是免费的定理，却极大提升可信度。 | 设计一个"对称版 RSP"：δ 由两对参考残差的差解出，作用时对两锚点反号。作为主方法的可选变体（"bias-free variant"）报告。 |
| **6** | **低频来自观测初始化、高频交给生成（不从纯噪声起）** | RSAC prior-enhanced init (J. Remote Sensing 2026, DOI 10.34133/remotesensing.1035) | 与 RSP 的频率分工完全同构，且已在遥感扩散里被验证有效。它解决"生成器把低频也重画一遍导致色偏/结构漂移"，正是 RSP 要修的病。 | 在 motivation 里用它做"频率分工在 RS 扩散中已被证实"的引证；并做一个"RSP vs 低频初始化"的对照，说明二者可叠加（一个在输入端、一个在输出端）。 |
| **7** | **不确定性/难度 → 决定哪里多算、哪里少算（零参数版）** | Patch Forcing (CVPR 2026) | RSP 目前对所有 patch 用同一投影强度，这在变化区会过修。Patch Forcing 证明"按不确定性分配"在固定预算下严格更好；我们可以用"参考对残差在两锚点上的不一致度"当免费的不确定性代理，**不加任何参数**。 | 加一个"逐像素投影强度 = f(两锚点残差差异)"的变体；用 patch 级消融证明变化区少投影、稳定区强投影更好。 |
| **8** | **"同一场景多版本应映射到同一目标"的一致性自监督 + 选择性更新** | DCTTA (CVPR 2026) | 给 RSP 的自监督信号提供**独立于 STF 文献的合法性**：不是我们发明的技巧，而是图像复原 TTA 的公认假设。同时它是"更新少量参数"的代表，正好反衬我们"零更新"，形成清晰的对照组。 | 在 related work 写"我们的自监督来自协议自带的参考对，而非合成退化"；实验上与其思路做概念对照（不实现），并在 limitation 里讨论"若允许更新参数还能再涨多少"。 |

**未进前 8 但值得留档**：
- **AnyBand-Diff 的 Physics-Guided Sampling**（§11）：插入点与 RSP 完全相同（首步噪声估计），但约束源是物理模型。是"我们 vs 物理先验"的最佳并置对象，适合写在 discussion 而非方法。
- **TCC 的配对无标注标定迁移**（§5）：如果 RSP 未来要跨传感器，这是最严谨的框架（有有限样本覆盖保证）。
- **DAMP 的 6 维可解释度量**（§12）：若要做"按样本难度决定是否启用 RSP"，这是现成度量清单。
- **FSATS / ArtFusion / DCAF**（§13）：STF 侧的最新 baseline，投稿时必须进对比表（尤其 ArtFusion 的"任意参考"会直接质疑"为什么只用两对"）。
- **Global scheduling 的"离线锚定 + 在线松弛"**（§12）：如果 RSP 要做逐样本强度，这是理论依据（自适应策略有线性 regret 下界，故不宜全在线）。

---

## 16. 待办（投稿前必须完成）

1. 逐条核实 🟡/⚠️ 条目的 DOI、作者序、venue（约 8 条）。
2. 抓 **WaFT** 全文，逐条比对与 RSP 的数学形式（尤其它的 MAP 解是否等价于某个仿射投影）。这是最大风险点。
3. 抓 **PRISM** 全文，确认其"Affine Bias Regression"是否已经是"用参考对解仿射 + 单次投影"；若是，RSP 的表述必须改成"temporal specialization of affine bias regression"并引用。
4. 抓 **Exact projection for DBT** 全文，确认其 no-harm 定理的前提（无噪/有噪），判断能否直接搬到 RSP。
5. 补 **IGARSS 2026 / CVPR 2026** 的 STF 相关 poster 全列表（目前只覆盖到 SA-STF、SRINO、TexADiff、3M-TI、Patch Forcing 等），确认无遗漏的同题工作。
6. 若时间允许，检索 **ISPRS J. P&RS / RSE 2026** 是否有"STF + 不确定性"或"STF + 基础模型"的新文（本次仅检出 STCAFormer 与 SuperSTF）。
