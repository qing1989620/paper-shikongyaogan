# 2024–2026 扩散式时空融合最新文献与撞题边界审计

调研日期：2026-11-17
调研范围：扩散式 STF、test-time adaptation / self-calibration / data consistency、撞题排查、目标会议核实
用途：为 "reference-pair self-calibrated temporal projection"（下称 **RSP**）创新点与撞题边界提供依据

---

## 0. 执行摘要（先看这里）

1. **撞题总判定：没有发现完全撞题的论文，但存在 4 个"部分占位"必须防守。** 未找到任何一篇同时满足 RSP 全部七要素（双锚点协议 + 逐样本自监督拟合 coarse→fine temporal affine + 不用预测日 fine GT + 不更新 backbone + 零新增网络参数 + 1 NFE + 对第一步扩散输出低频 temporal residual 做投影修正）的工作。但 SA-STF（CVPR2026）的 TFA 模块已经做了"逐像素 coarse→fine 时间仿射系数 (a,b)"，DPDiff（TGRS2026）已经占了"temporal change prior"，传统 ESTARFM/FSDAF/Fit-FC 已经占了"coarse→fine 时间变化线性回归 + 残差补偿"，TAO/CCDF 已经占了"test-time adaptation / data consistency projection"。**RSP 的存活空间在于"closed-form 逐样本自校准 + 输出残差投影 + 参数冻结 + 1 NFE"这一组合，而不是任何一个单点。**

2. **最危险的单篇是 SA-STF（CVPR2026）**：它用的正是官方双锚点协议 {C1, F0, C0, F2, C2} → F1，且其 TFA 用线性回归把 fine 特征对齐到目标时刻，形式上是 `HR0(i,j) = a_h(i,j) × F0(i,j) + b`。审稿人极可能问"这和 TFA 有何区别"。**必须准备好逐条对比：它是 CNN 学出的逐像素系数、在深层特征空间、训练期学得、属于 backbone；RSP 是闭式解出的逐样本 12 标量、在扩散输出残差上、测试期即时拟合、不碰 backbone。** 内部总结表已标注 SA-STF 为"★最高撞题风险"，本次外部检索确认该判断。

3. **时间窗口警报（影响投稿计划）**：DLNN 2026 全文截稿 2026-10-08 **已过**；ICASSP 2027 全文截稿 2026-09-16 **已过**。截至今天（2026-11-17），DLNN 2026 仅剩最终提交（11-28，面向已录用论文）；DLNN 2027（同系列）尚未发布；ICASSP 2027 已进录用通知阶段。**如果目标是"新投一篇"，这两个会议当前都投不了**，需改投 DLNN 2027（预计 2027-10 前后开放）或 ICASSP 2028。详见第 5 节。

4. **效率主张要收窄**：SA-STF 用 DDIM 50 步、STFDiff 多步采样；RSP 的 1 NFE 是相对优势，但不要写"比所有 diffusion STF 都快"——SRINO（CVPR2026）用神经算子替代自注意力降 FLOPs 约 70%，是另一条效率路线。

---

## 1. 扩散式 STF 2024–2026 最新工作清单

### 1.1 逐篇核实表

| 方法 | 出处 / 时间 | 协议与核心方法 | 数据集 | 效率 / 代码 | 核实状态 |
|---|---|---|---|---|---|
| **STFDiff** | Information Fusion, 2024-06, DOI `10.1016/j.inffus.2024.102505`；作者 Huang He, He Wei, Zhang Hongyan, Xia Yu, Zhang Liangpei | 官方双锚点：{C1(预测日粗), (F0,C0), (F2,C2)} → F1。DS-Unet 双流编码器相减增强噪声特征，迭代去噪；早期恢复光谱、后期恢复空间细节 | CIA、LGC + 自建 E-SMILE；下游 K-means mIoU +2% | 代码：`github.com/prowDIY/STF`（统一 STF 库，权重逐步放出）；多步采样；Google Scholar 被引 89 | ✅ 真实，多源交叉确认 |
| **SA-STF** | CVPR 2026；Jinsong Zhang, Ying Qu, Yuan Liao, Hairong Qi, Zhenzhou Shao | 官方双锚点 Cond={C1,F0,C0,F2,C2}→F1。RDDM 式残差扩散 + **TFA（时间特征对齐，线性回归系数 a,b）** + SAF（语义自适应变化判定/高频迁移）+ Taylor-inspired De-Res。单网络统一 F1t 估计 | CIA / LGC / AHB，对比 ESTARFM、MLFF-GAN、SwinSTFM、STFDiff、STFMamba，三数据集总体最佳 | GitHub `ZZZhangjinsong/SA-STF`（MIT，三个数据集预训练权重已放出）；T=100、**DDIM 推理 50 步**，200 epoch | ✅ 真实；CVF 官方 PDF + GitHub 双确认。**最高撞题风险** |
| **DPDiff** | IEEE TGRS, 2026-02-13, DOI `10.1109/TGRS.2026.3664495`；Baoluo Zhu, Huihui Song, Kaihua Zhang, Guangcan Liu | 条件扩散。**PENet（轻量 CNN）学时空双先验：SMP（同日 coarse↔fine 空间映射先验）+ TCP（跨时 coarse↔fine 时间变化先验），二者互逆**；RDF 残差扩散学分布差异；DAM 对齐 CNN 预测空间与 GT 空间 | CIA、LGC、WaterBank、全球 E-SMILE | 代码：`github.com/paul623/DPDiff`（摘要明示 source code available）；效率数据未在检索摘要中给出 | ✅ 真实；**第二撞题风险（TCP 已占"时间变化先验"措辞）** |
| **STF-FlowDiff** | IEEE TGRS, 2026-04-28, DOI `10.1109/TGRS.2026.3687438`；Hui Shen, Hongjun Su, Xiaojuan Liu, Hui Yang, Qian Du | 在 reference fine 与"由 coarse 输入构造的 proxy fine"之间估计**稠密光流**，作为显式先验引导扩散；GFC（梯度流置信度）选稳定区、MFI（多尺度流注入）注入扩散编码器 | 3 个真实数据集 | 未检索到公开代码；强调 temporal consistency 与参数鲁棒性 | ✅ 真实；IEEE Xplore + Google Scholar 确认 |
| **WDCDiff** | IEEE GRSL, Vol.23, 2026, DOI `10.1109/LGRS.2025.3635659`；Weiquan Kong, Yurong Qian, Lu Bai, Yuanxu Wang, Sen Luo, S.N. Tereshchenko | 两阶段：先**小波去噪**（针对高频噪声导致的过平滑），再条件扩散增强 | 未在第 4 方检索片段中列全 | 未检索到公开代码 | ✅ 真实；dblp + IEEE Xplore 确认 |
| **SuperSTF** | ISPRS J. P&RS, Vol.235, 2026-03/05, DOI `10.1016/j.isprsjprs.2026.02.037`；Min Zhao, Jiajun Cai, Man Zhou, Bo Huang | **all-in-one 去云 + STF**：潜在扩散，Swin Transformer 作像素空间自编码器，cross-attention 多源融合，云位置编码 + 获取日期调制。输入是 coarse 序列 + 有云 fine 观测 | 多时相多源序列 | 未检索到公开代码 | ✅ 真实；ScienceDirect/Mendeley/Google Scholar 确认。**注意：它不做双锚点协议**，是去云+STF 联合任务 |
| **DiffSTF** | 2024；Ma 等（转引自两篇综述） | 条件扩散；参考日 fine 的**结构信息** + 预测日 coarse 的**光谱信息**作为双条件；encoder-decoder + residual/Transformer block | 见原文 | 代码未核实 | ⚠️ 仅由综述转引，未直接检出原文页面；引用前需查原文 |
| **DiffSTSF** | 2024；Wei 等（转引自两篇综述） | 融合 GF-1 的 2m PAN / 8m MS / 16m WFV；多条件扩散；前向=显式降质建模，反向=融合 | GF-1 多传感器 | 代码未核实 | ⚠️ 仅由综述转引；引用前需查原文 |
| **MSRS-CDM** | 《航天返回与遥感》2026年第4期, DOI `10.3969/j.issn.1009-8518.2026.04.008` | 多源条件扩散；CT-SSG（跨时结构-光谱引导）+ DMG-Fusion（双流互导融合）；两图输入也能工作 | LGC、天津数据集 | 中文期刊，代码未提及 | ✅ 真实（中文期刊网页） |
| **SRINO** | CVPR 2026（Same session as SA-STF） | Galerkin 神经算子替代自注意力；空间/光谱残差约束；面向星上资源受限部署 | WV3/GF2/QB（pansharpening 任务为主） | 512 分辨率 FLOPs 从约 5000 降到约 1500，约 -70%；推理快数倍 | ⚠️ 来自二手聚合站（graphastra），引用前需核 CVF 原文 |

### 1.2 对本项目的关键读法

- **官方双锚点协议已被三篇主流方法采用**：STFDiff（奠基）、SA-STF（CVPR2026）、DPDiff（TGRS2026）。因此"在双锚点协议下做 STF"本身**不构成创新**，只能作为实验设定陈述。
- **"coarse 约束残差扩散"也已被占**：SA-STF 用 `{C1 - F1t}` 项约束解空间；STFDiff 用双流相减。RSP 不要宣称"首次用 coarse 约束扩散"。
- **效率对比基线要选对**：SA-STF（DDIM 50 步）和 STFDiff（多步）是合适的 1 NFE 对照；不要与 SRINO 比 FLOPs（它换的是注意力算子，不是采样步数）。
- **数据集口径**：LGC/CIA 是 STFDiff/SA-STF/DPDiff/MSRS-CDM 的公共基准；AHB 是 SA-STF 引入；E-SMILE 是 STFDiff/DPDiff 的全球集。内部实验在 LGC/CIA 上有效，与主流口径一致；但若要强调泛化，需补 AHB 或 E-SMILE。

---

## 2. TTA / 自校准 / data consistency 工作的边界

| 工作 | 出处 | 做法 | 与 RSP 的边界 |
|---|---|---|---|
| **CCDF** | CVPR 2022, DOI `10.1109/CVPR52688.2022.01209`, arXiv `2112.05146` | "反向扩散 + 非扩张 data consistency 步骤"交替；用随机微分方程收缩理论解释加速；已知测量算子 A（SR/修复/MRI） | CCDF 的投影是**已知物理测量算子**的数据一致性；RSP 的投影约束是**从样本自带 reference pairs 拟合出的时间仿射**，无已知 A。可引用为"投影修正"的理论近亲，但必须点明约束来源不同 |
| **TAO (Test-Time Degradation Adaptation)** | ICML 2024 Spotlight, PMLR v235:16167-16177；GitHub `XLearning-SCU/2024-ICML-TAO` | 预训练退化无关 DDPM + **可学习 TDA 适配器（测试期梯度更新）** + AIR 引导；逐去噪步执行；去雾/低光/去噪 | TAO **确实更新参数**（适配器），且面向开集退化分布偏移；RSP **零参数更新、零新增网络**。不要把主定位叫"test-time adaptation"，否则直接落入 TAO 的领地 |
| **Uncertainty-Aware TTA for Cross-Region LST STF** | arXiv `2604.04153`, 录用 IGARSS 2026（2026-08-09~14, Washington DC） | 只更新预训练 STF 模型的**融合模块**，由认知不确定性 + LULC 一致性 + 偏差校正引导；无需源数据/标记目标样本；10 个 TTA epoch；Rome/Cairo/Madrid/Montpellier 四区域 RMSE/MAE 平均改善 24.2%/27.9% | 这是**跨区域域偏移**场景，且**更新模型参数、需 10 epoch**。RSP 是同协议内每样本即时闭式自校准、不更新任何权重、1 NFE。差异可守住，但"test-time"一词已被它占用，措辞需加限定 |
| **SATeCo** | CVPR（视频超分，Peeref 收录） | 冻结预训练 UNet/VAE，只优化 SFA（**逐像素估计仿射参数**）和 TFA 模块 | 它是"逐像素 affine 调制 + 微调模块"，RSP 是"逐样本 12 标量闭式仿射 + 输出残差投影"。可作为"affine 校准"近亲，但任务/粒度/是否训练都不同 |
| **CTCal (Cross-Timestep Self-Calibration)** | ~2026, arXiv `2603.20741`, GitHub `xiefan-guo/ctcal` | 用**小 t** 的 cross-attention 图校准大 t 的学习，训练期 fine-tuning，T2I 任务 | "self-calibration" 在扩散里已被用于"跨时间步自监督"，且是训练期调参。RSP 的 self-calibration 必须永远带 "reference-pair / per-sample / closed-form / parameter-free" 限定词 |
| **传统 STF 的线性回归谱系** | ESTARFM (Zhu 2010)、FSDAF (Zhu 2016)、Fit-FC (Wang & Atkinson 2018) | ESTARFM：移动窗口内对相似像元做**线性回归得转换系数 V**；FSDAF：按类别估计 coarse 时间变化 + TPS 残差再分配；Fit-FC：对两幅 coarse 做**局部线性回归**，系数作用于 fine + 残差补偿 | 这是最容易被审稿人翻出来的"现有技术"：coarse→fine 时间变化线性映射 + 残差补偿在 2010–2018 已是常识。RSP 必须强调不是"又一个线性回归"，而是"把 reference pairs 当作逐样本自监督信号去拟合一个作用在**扩散输出低频残差**上的投影算子" |

**边界小结（可用于 rebuttal 的三句话）：**

> CCDF 的 data consistency 依赖已知测量算子；TAO 与跨区域 LST TTA 都在测试期更新模型参数；传统 STF 的线性回归直接在图像域做预测。RSP 不更新任何权重、不使用预测日 fine GT，而是仅用目标样本自带的两对 reference 以闭式形式拟合一个时间仿射，并把该仿射作为投影算子作用于扩散首步输出的低频 temporal residual——约束来源、作用位置、参数状态三者均不同。

---

## 3. 最大撞题风险专项排查

### 3.1 检索结论

**未检出**题目/摘要明确写有"用 reference pairs 逐样本自校准 coarse-fine temporal mapping + 投影修正 diffusion 结果"的论文。检索覆盖：WebSearch 多组关键词（中英文）、Google Scholar 条目、arXiv、CVF Open Access、IEEE Xplore、dblp、Semantic Scholar、GitHub。

### 3.2 四个"部分占位"及防守要点

| # | 占位者 | 它占了什么 | RSP 必须守住的差异化 |
|---|---|---|---|
| 1 | **SA-STF TFA**（CVPR2026） | 逐像素 coarse→fine 时间仿射：`a_l=f_a(Concat(C1,C0))`, `b=f_b(F0−C0)`, `a_h=g(a_l,C0,F0)`, `HR0(i,j)=a_h(i,j)×F0(i,j)+b` | 它是 **CNN 卷积块学出的逐像素系数**（训练得到）、作用于**深层特征 F0⁵→HR0⁵**、是 **backbone 的一部分**、需 50 步 DDIM。RSP 是**闭式最小二乘逐样本 12 标量**、作用于**扩散输出的低频 temporal residual**、**冻结 backbone**、1 NFE |
| 2 | **DPDiff TCP**（TGRS2026） | "temporal change prior between coarse and fine images at different times" 的原话与概念 | 它由 **PENet 轻量 CNN 在训练期学习**，作为**条件信号注入扩散**（生成前）。RSP 在**生成后**做投影修正，且是逐样本闭式拟合，不是网络学到的全局先验。措辞上避免把 12 标量叫 "learned prior" |
| 3 | **ESTARFM / Fit-FC / FSDAF** | coarse↔fine 时间变化线性回归、转换系数、残差补偿 | 这些是**图像域直接预测**，无扩散、无"首步输出残差"概念。RSP 只在 diffusion 框架内成立，必须始终写出 "on the first-step diffusion output" 这个作用位置 |
| 4 | **CCDF / TAO / 跨区域 LST TTA** | "data consistency projection" / "test-time adaptation" 术语 | 见第 2 节。RSP 定位用 "parameter-free test-time **self-calibration**"，不用 "adaptation"，不用裸 "data consistency" |

### 3.3 若审稿人质疑"这不就是 TFA/Fit-FC 的逐样本版吗"的标准答复框架

1. **对象不同**：TFA/Fit-FC 对齐/回归的是*图像或深层特征*；RSP 修正的是*扩散首步输出的低频 temporal residual*（高频细节由 backbone 负责，RSP 不碰）。
2. **获取方式不同**：TFA 系数来自卷积网络的前向输出；RSP 的 12 标量来自对样本自带两对 reference 的闭式最小二乘，无梯度、无学习。
3. **状态不同**：TFA 随 backbone 训练并固化；RSP 在测试时对每个样本即时估计，用后即弃，不留下任何跨样本参数。
4. **代价不同**：TFA 随 50 步 DDIM 每步生效；RSP 只做 1 次投影，1 NFE。
5. **信息使用不同**：Fit-FC/ESTARFM 用 reference pairs 直接生成预测；RSP 用 reference pairs 只估计一个*时间变化仿射算子*，不生成任何像素。

---

## 4. 推荐 contribution 表述与必须避开的措辞

### 4.1 三条最安全的 contribution（建议按此顺序写）

1. **样本级自校准机制（主创新）**
   > "We propose **reference-pair self-calibrated temporal projection** for diffusion-based spatiotemporal fusion: for each test sample, a coarse-to-fine temporal-change affine (12 scalars) is estimated **in closed form** solely from its own dual-anchor reference pairs, and applied as a **single-step projection** to the low-frequency temporal residual of the first diffusion output — **without accessing the prediction-date fine ground truth**."

   关键限定词缺一不可：`per test sample` / `closed form` / `its own dual-anchor reference pairs` / `first diffusion output` / `without prediction-date fine GT`。

2. **参数与计算属性（第二创新，最硬）**
   > "The correction is **training-free and parameter-free**: the pretrained backbone is frozen, **no neural-network parameters are introduced**, and only **one additional network function evaluation (1 NFE)** is required at inference."

   这条最安全，因为 SA-STF/DPDiff/TAO/跨区域 LST TTA 全都违反至少一条。

3. **问题层面的洞察（第三条，拔高但不越界）**
   > "We show that the standard dual-anchor STF protocol itself provides sufficient **self-supervision** to calibrate the coarse–fine temporal mapping per sample, and that correcting the low-frequency temporal bias of the first diffusion step yields consistent gains on LGC and CIA under the official protocol."

   把"自监督信号来自协议本身"作为 insight，不碰"首次建模时间变化"这类已被占的断言。

### 4.2 必须避开的措辞（按危险程度排序）

| 禁用/慎用措辞 | 原因（被谁占了） | 替代表述 |
|---|---|---|
| "first to align/estimate coarse–fine temporal features/mapping" | SA-STF TFA 已做逐像素 affine 对齐；DPDiff TCP 已做 temporal change prior | "first **closed-form, per-sample** self-calibration of the coarse–fine temporal mapping **from the sample's own reference pairs**" |
| "test-time adaptation (TTA)" 作为主标签 | TAO (ICML2024)、跨区域 LST TTA (IGARSS2026) 已占，且它们都更新参数 | "training-free / parameter-free test-time **self-calibration**"，并在 related work 显式对比 |
| 裸用 "data consistency projection" | CCDF 已占，且它投影到已知测量算子 | "projection onto a **self-calibrated temporal-affine constraint**"，注明非测量一致性 |
| "learned temporal prior" / "temporal change prior" | DPDiff TCP 原话 | "**fitted** per-sample temporal affine (closed-form, not learned)" |
| "coarse-to-fine linear regression" 作为新意 | ESTARFM/Fit-FC 自 2010/2018 已有 | "closed-form self-calibration applied to diffusion-output residual" |
| "semantic-adaptive / change-aware fusion" | SA-STF SAF、GSINet SEFM 已覆盖 | 不单列 contribution，只作为可兼容模块 |
| "cross-domain / zero-shot generalization" | DeepFDP 占频率先验跨域，跨区域 LST TTA 占 TTA 跨域；内部只有 LGC/CIA | 不写；除非补 AHB/E-SMILE 实验 |
| "no reference fine images required" | 事实错误——RSP **需要**前后两对 reference fine/coarse | "no **prediction-date** fine GT required" |
| "self-calibration" 不加限定 | CTCal、相机自标定等大量文献稀释 | 永远写 "reference-pair self-calibrated / per-sample / closed-form / parameter-free" |

### 4.3 建议的 related-work 定位句（可直接用）

> "Unlike residual-diffusion STF methods that learn temporal alignment or change priors within the backbone (e.g., SA-STF's TFA, DPDiff's TCP), and unlike test-time adaptation methods that update model parameters (e.g., TAO, cross-region LST TTA), our method treats the two reference pairs already present in the official dual-anchor protocol as a **per-sample self-supervision signal**, fitting a closed-form coarse-to-fine temporal affine that corrects **only** the low-frequency temporal residual of the first diffusion output."

---

## 5. 目标会议核实（含准确 URL 与时间警报）

### 5.1 DLNN 2026（International Conference on Deep Learning and Neural Networks）

- **官网**：`http://www.dlnn.org` / `https://www.dlnn.org`
- **投稿/作者页**：`https://www.dlnn.org?p=9511/`
- **征稿页**：`http://www.dlnn.org/call-for-papers/`
- **会议**：2026 第二届，2026-12-11 至 12-13，中国武汉
- **关键日期**（官网原文）：
  - Full/Submission Deadline：**2026-10-08**（**已过**）
  - Acceptance Notification：2026-11-18
  - Final Submission Deadline：2026-11-28
  - Registration Deadline：2026-12-05
- **模板下载**：官网统一为一个 "Template" 链接（页面未分列 Word/LaTeX）：
  - `https://www.dlnn.org/?attachment_id=10430&download=1`
  - 下载后需自行确认是 .doc 还是 .tex/.zip；若确需 LaTeX 而链接只给 Word，发信 `dlnn_1@163.com` 索取。
  - Logo（可选）：`https://www.dlnn.org/?attachment_id=9892&download=1`
- **格式**：英文，最少 5 页，超出 5 页每页 42 USD / 300 CNY；单盲评审；录用并报告后进 proceedings（2025 届 proceedings 由 IOP Publishing 出版）。
- **联系**：`dlnn_1@163.com`

### 5.2 DLNN 2027（同系列）

- **截至 2026-11-17 尚未发布官网/截稿/模板**。第三方会议聚合站 myhuiban 基于上一届推测"下届截稿约 2027-10"，仅供参考，不可当作官方日期。
- **重要区分**：检索到的 "**ICDLNN 2027**"（International Conference on Deep Learning and Neural Networks，2027-01-15，美国亚特兰大，截稿 2026-12-19，`bigdataresearchforum.com` 主办）是**另一个组织的不同会议**，不是 dlnn.org 这个系列。投稿前务必确认目标到底是哪一个，避免投错。

### 5.3 ICASSP 2027

- **官网**：`https://2027.ieeeicassp.org/`
- **CFP 官方页**：`https://2027.ieeeicassp.org/call-for-papers/`（检索到的镜像路径 `https://2027.ieeeicassp.org/?p=3718/` 内容一致）
- **会议**：第 52 届，2027-05-16 至 05-21，加拿大多伦多
- **关键日期**（官网 CFP 原文）：
  - Full Paper Submission Deadline：**2026-09-16**（**已过**）
  - Published Journal Papers and Letters Due：2027-01-06
  - Acceptance Notification：2027-01-13
  - Final Paper Deadline：2027-01-27
  - Author Registration Deadline：2027-02-10
  - Open Preview Starts：2027-04-16
- **格式**：IEEEtran 双栏、US Letter；技术内容最多 4 页，第 5 页仅放参考文献/致谢/合规声明；OJSP track 为 8+1 页但进期刊不进 proceedings。
- **LaTeX 模板**：IEEE 官方模板页 `https://www.ieee.org/conferences/publishing/templates.html`（选 IEEEtran conference 模式）；不要依赖 call4papers.org 的第三方链接（该站把截稿误写为 2026-09-23，以官网 2026-09-16 为准）。
- **技术范围明确包含** "Remote Sensing & Signal Processing"、"Image, Video, and Multidimensional Signal Processing"、"Computational Imaging"，与 STF 主题契合。

### 5.4 时间线结论与建议

| 目标 | 状态（截至 2026-11-17） | 可行动项 |
|---|---|---|
| DLNN 2026 | 全文截稿已过；录用通知明天（11-18）；最终提交 11-28 | 仅适用于**已被录用**的稿件；新投不可能 |
| DLNN 2027 | 未发布 | 盯 dlnn.org，按 2026 节奏预计 2027 年 10 月截稿；先按 DLNN 2026 模板排版备用 |
| ICDLNN 2027（亚特兰大） | 截稿 2026-12-19 | 若期刊/DOI 要求可接受，可作为备选，但须核对主办方与 indexing 口碑 |
| ICASSP 2027 | 截稿已过，等 2027-01-13 通知 | 新投不可能 |
| ICASSP 2028 | 预计 2027 年 9 月截稿 | 按往年节奏准备，模板通用 IEEEtran |

---

## 6. 来源可靠性说明

- **一手确认**（可用于正式引用）：STFDiff（ScienceDirect + GitHub + DOI）、SA-STF（CVF 官方 PDF + GitHub）、DPDiff（IEEE Xplore/Scilit + DOI + GitHub）、STF-FlowDiff（IEEE Xplore + DOI）、WDCDiff（dblp + DOI）、SuperSTF（ScienceDirect/Mendeley + DOI）、CCDF（CVF + DOI）、TAO（PMLR + ICML virtual + GitHub）、跨区域 LST TTA（arXiv 原文 + IGARSS 声明）、DLNN/ICASSP（各自官网）。
- **二手/待核**（引用前必须查原文）：DiffSTF、DiffSTSF（仅由 2025 Sensors 综述与 arXiv 综述转引）；SRINO（graphastra 聚合站）；DLNN 2027 日期（myhuiban 推测）。
- **未检出公开代码**：STF-FlowDiff、WDCDiff、SuperSTF（不等于没有，只是本轮检索未发现）。

---

## 7. 后续建议（给团队）

1. **立刻把 SA-STF TFA 公式放进 related work 并写死对比表**（第 3.2 节表格可直接改写进论文），这是最高优先级防守动作。
2. 所有"self-calibration"出现处统一加 `reference-pair / per-sample / closed-form / parameter-free` 限定，避免与 CTCal 及相机标定文献混淆。
3. 补一组 **SA-STF（50 步 DDIM）vs RSP（1 NFE）** 的效率对比，强化第 2 条 contribution。
4. 若预算允许，在 AHB 或 E-SMILE 上加一组实验，堵住"只在 LGC/CIA 有效"的泛化质疑（但不要主动宣称 cross-domain）。
5. 投稿目标尽快决策：两个原定会议都已错过截稿，需明确改投 DLNN 2027 / ICASSP 2028 / ICDLNN 2027 或其它。
