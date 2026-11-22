# 2026-11-22 阶段定稿

中英文稿围绕同一个问题展开。保持观测输入、共享去噪主干和单次去噪调用，先验中心和空间噪声尺度的改变，能否降低同一 LGC 划分上的重建误差？

| 读稿与复核 | 文件 |
|---|---|
| 中文论文 | [STF_stage_zh.pdf](output/pdf/STF_stage_zh.pdf) |
| 英文论文 | [STF_stage_en.pdf](output/pdf/STF_stage_en.pdf) |
| 可编辑机制图 | [STF_bridge_editable.pptx](source/figures/STF_bridge_editable.pptx) |
| LaTeX 和文献库 | [source](source/) |
| 数值、原始运行记录与摘要哈希 | [evidence](evidence/) |
| 可独立解包的阶段包 | [STF_stage_20261122.zip](https://github.com/qing1989620/paper-shikongyaogan/releases/download/backup-20261123-seed42/STF_stage_20261122.zip) |

## 冻结范围与结论

- 2026 年 11 月 22 日 22:20（北京时间）完成 I05 后暂停。从零训练计划完成 8/45 组，含基线和七个变体，均为种子 42、40 轮。另有 14 项六轮初筛，均未通过相对于原始强基线的改善门槛。
- I02 最佳检查点的平均 patch RMSE 为 0.024445，较基线最佳值 0.025080 下降 2.53%。共同第 40 轮的比较为 0.028603 对 0.025618，I02 高 11.65%。I01 最佳检查点下降 1.36%，第 40 轮高 1.21%。已完成变体中，I03 是共同第 40 轮 RMSE 唯一低于基线的方案。
- I02 在 120 个配对 patch 中改善了 63 个。以基线减 I02 计算的平均绝对差为 0.635 × 10⁻³，场景内分块重采样区间 [−0.005, 1.256] × 10⁻³ 包含零。粗观测偏离从低到高三组的相对 RMSE 降幅为 +7.71%、+5.39%、−5.19%。分组依据为观测量，结果限于一个训练种子、一个开发验证日期。
- 主表采用同一 LGC 开发划分的项目基线，分别保留最小验证 RMSE 与共同训练终点。外部方法的统一协议比较、跨场景结果和机制消融仍需独立证据。这一版采用项目已有双栏版式，署名暂匿名。

## 核对与复编

遵循科研区《科研论文写作与绘图规范》。主表与统计图取自冻结记录，图中以正值表示 RMSE 改善。原始日历清单、完整性标记、配置、逐轮日志和每个最佳检查点的逐 patch 验证输出存放在 evidence，数值复算以这些记录和预提取的 120 个粗偏离量为输入。

Windows PowerShell 在解包后的根目录执行以下命令。编译需要 TeX Live（latexmk、pdfLaTeX、XeLaTeX、BibTeX）以及宋体、楷体、黑体。重绘使用 Arial、Microsoft YaHei 和 Matplotlib 自带 DejaVu Sans。已验证 Python 3.12、NumPy 2.5.5、Matplotlib 3.11.2、python-pptx 1.0.2。

```powershell
python scripts/freeze_evidence.py --archived
python scripts/plot_results.py
python scripts/build_editable_figure.py
.\build.ps1
```

只编译 PDF 时执行最后一行即可。压缩包包含论文和图稿、编译源码、复算记录及核验说明。原始影像与训练权重保留在项目中。若需从项目文件重新核对权重哈希与计算粗偏离，可使用 freeze_evidence.py 的 --project 参数指定原工作区。这些命令处理留存记录和图文。
