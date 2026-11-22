# 工具索引

| 组别 | 脚本 | 当前定位 |
|---|---|---|
| 数据 | prepare_dataset_v2.py / prepare_train.py | 新的 10 组有界内存准备流程，30 个原始 patch 匹配通过；CIA 自动排除含验证/测试目标 fine 日期的训练组 |
| 训练与骨干 | train_stfdiff.py / backbone_stfdiff.py | 历史流程；随机种子、验证口径、采样配置待修后作为正式基线 |
| 旧评测 | eval_methods.py / exp20.py / exp20_final.py / ablate_rsp.py | 保留复核，不直接当新统一基准 |
| 新指标 | benchmark_metrics.py | reflectance_v1；已作解析例子及六个真实缓存检查，尚未接入所有模型适配器 |
| 2026 研究原型 | research_methods_2026.py | 14 个独立方向与原基础模型，功能/梯度/显存已检查 |
| 新配对训练 | train_research_2026.py | 三噪声验证、共享初始权重、数据/代码 SHA、逐轮续训，正式初筛正在运行 |
| 旧汇总 | final_aggregate.py / lock_numbers.py / leaderboard.py / stats_tests.py | 有单位、统计与来源问题；不要覆盖旧 FINAL 后直接用于投稿 |
| 诊断 | diagnose_backbone.py | 探索诊断，统计斜率不等同物理常数 |
| 下载辅助 | download_lgc.py / download_cia.py / fast_download.py / par_dl.py / dl2.py / dl3.py 等 | 历史脚本；重新下载先核对原始文件大小与空间预算 |
| 旧出图 | make_figure.py / make_method_fig*.py / make_pptx_figure.py / export_plot_data.py | 历史图；新图必须从新版本结果产生 |
| 缓存盘点 | cleanup.py | 现在只报告缓存大小，不删除；旧原件在 archive/legacy_tools |

从项目根目录运行的新工具位于 `../../scripts/`。当前已启动开发筛选训练；完整 SOTA 与确认性实验尚未完成。
