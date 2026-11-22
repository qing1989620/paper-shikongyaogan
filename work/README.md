# 可运行工作区

当前研究入口：[项目 README](../README.md)。计划与审计已集中到 `../docs/`，散落的旧日志移至 `logs/legacy/`。

核心数据、模型代码、权重、CSV、缓存和论文相对路径保持不变。原有脚本多数要求从本目录执行；新项目审计脚本可从任意目录执行。

```text
data/       原始影像与 patch；CIA 原始数据需要恢复
repos/      外部方法及现有本地架构修改
tools/      训练、评测、指标、聚合与诊断脚本
runs/       旧权重保留；本轮实际运行在 research_2026/<run_id>
results/    旧结果保留；research_2026 是新开发评测，benchmark_v1 为后续正式基准
logs/       legacy/ 是迁移后的旧文本日志
configs/    新协议配置；benchmark_v1 当前为 DRAFT
paper/      历史论文稿、图和模板
TCCR_STF/   早期方法包及项目旧训 checkpoint，保留供复核
```
