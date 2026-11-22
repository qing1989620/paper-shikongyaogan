# 2026 创新来源与撞题边界

核验日期：2026-11-19。区分期刊、顶会与直接近邻；不把所有来源统称“顶刊”。以下机制摘要是对一手来源的简述，研究候选中的新方程与适配设计是我们的假设。没有全文阅读的项明确限定证据深度。

| 编号 | 论文与出处 | 阅读/核验范围 | 可借鉴的思想及边界 |
|---|---|---|---|
| J01 | [From Zero to Detail，TPAMI 2026；作者原文](https://arxiv.org/html/2604.15654v1)，[作者出版列表](https://tyshiwo.github.io/pub.html) | 原文方法、动机与消融章节；作者确认 TPAMI | DC/LF/HF 分工；2026 版是此前会议工作的扩展，不把思想全算 2026 首发 |
| J02 | [TABG-Net，TGRS 2026](https://ieeexplore.ieee.org/abstract/document/11658961)，[作者代码](https://github.com/hgxcxdg/TABG-Net) | 出版商摘要检索可见，直接全文访问受限 | 高频与边界引导解混；不能说首次边界/双频融合 |
| J03 | [Temporally-Similar Structure-Aware STF，TGRS 2026；作者原文](https://arxiv.org/abs/2508.11259)，[作者出版页](https://sites.google.com/view/ryosuke-isono-jap/publications) | 作者原文摘要与作者期刊元数据 | 时序引导 TV/边缘约束；预印本为 2025，期刊为 2026，不能伪装全新年份 |
| J04 | [STF-FlowDiff，TGRS 2026](https://doi.org/10.1109/TGRS.2026.3687438) | 出版商摘要，未完成源码和全部超参审计 | 光流置信度和多尺度注入已用于 STF，光流本身不新 |
| J05 | [MPGU-Net，TGRS 2026；作者代码与公式说明](https://github.com/xuanwentao/MPGU-Net)，[作者实验室出版页](https://visielab.uantwerpen.be/publications/mpgu-net-multi-model-physics-guided-network-nonlinear-hyperspectral-unmixing) | 作者方法说明、期刊信息；未复现 | 显式非线性混合和成分估计；六波段适配的可辨识性需另证 |
| J06 | [Physics-Guided Generative Pansharpening under Unknown Sensor Degradation，JSTARS 2026](https://ieeexplore.ieee.org/abstract/document/11718946/) | 出版商摘要 | 从观测自标定退化与响应；不等于知道 STF 的真实成像算子 |
| J07 | [MSDCN-LSTF，TGRS 2026](https://ieeexplore.ieee.org/abstract/document/11483138) | 出版商摘要 | 稳定性权重、传感器偏差和时变建模；任务为 LST，不能直接搬用其评分 |
| J08 | [Bridging Physics and Priors / Water2Air，TMM 2026](https://ieeexplore.ieee.org/abstract/document/11538173) | 出版商摘要 | 物理退化条件、多先验和跨域适应；水下光学先验不是遥感传感器事实 |
| J09 | [Spatiotemporal Fusion Based on Cloudy Time-Series，TGRS 2026](https://ieeexplore.ieee.org/abstract/document/11480169/) | 出版商摘要 | 有效时序观测的增量填补；我们的固定五观测协议不额外加时相 |
| C01 | [UDBM，ICML 2026，官方 PMLR](https://proceedings.mlr.press/v306/tu26d.html) | 正式摘要与方法概述 | 像素不确定性引导 diffusion bridge；桥和单步本身不新 |
| C02 | [CARD，CVPR 2026 官方论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Nezakati_CARD_Correlation_Aware_Restoration_with_Diffusion_CVPR_2026_paper.pdf) | 官方搜索结果中的相关噪声、协方差/Cholesky 方法片段；直接下载受限 | 相关性与白化恢复，完整理论与代码仍需细读再作原创性判断 |
| C03 | [FAPE-IR，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_FAPE-IR_Frequency-Aware_Planning_and_Execution_Framework_for_All-in-One_Image_Restoration_CVPR_2026_paper.html) | 正式摘要 | 频率规划与专家执行；不沿用其 MLLM 或自称首次专家恢复 |
| C04 | [Efficient and Training-Free Single-Image Diffusion Models，CVPR 2026；作者原文](https://arxiv.org/abs/2606.04299)，[作者实现](https://github.com/haojunqiu/efficient-SID-code) | 作者摘要、官方论文方法片段、README | 图像内部 patch 先验；我们的跨时传输不是该方法复现 |
| C05 | [Restore, Assess, Repeat，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.html)，[作者实现](https://github.com/saic-fi/RAR) | 正式摘要与方法概述、作者实现说明 | 恢复与质量反馈耦合；原型固定递推不等于已实现自适应停止 |
| C06 | [DCTTA，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.html) | 正式摘要与三组件说明 | 退化一致性和选择性适应；借其问题与约束，不把 EMA 训练冒充其算法 |

## 必须正面对比的直接近邻

- [SA-STF，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Semantic-Adaptive_Diffusion_for_Dynamic_Spatiotemporal_Fusion_CVPR_2026_paper.html)：时间特征对齐和语义自适应融合。已获取作者源码及 LGC 权重，补充材料 Table 3 的训练/测试日期已核对。我们的 March02 val 属于其训练日期；flood 日期一致，可按共同 patch 新实测，不能合并论文自报分数。详见 [共同基准](COMMON_BENCHMARK_2026.md)。
- [WPFM-STF，Remote Sensing，2026-11-07](https://www.mdpi.com/2072-4292/18/21/3832)：学习先验、prior-to-target flow matching、小波细节约束已直接用于同一任务。**“换 flow matching + 加小波”不能作为本项目大创新的充分理由。**
- [CLAW-GAN，IJCAI 2026](https://www.ijcai.org/proceedings/2026/163)：云污染 STF 中已有变化感知门控和频率分离。云鲁棒、门控、小波也都不能裸称首次。
- [DPDiff 作者仓库](https://github.com/paul623/DPDiff)：双先验思想是近邻，当前代码未完整发布。不把未复跑的方法填写成我们的实测对照。

本轮未使用二手博客作为机制事实的最终依据。对于只能读摘要或片段的条目，候选晋级后须补全文公式和作者代码比较；当前记录的“区别”是待验证的研究设计，不是排除了所有同类论文的新颖性证明。
