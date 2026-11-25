# 实验操作入口

2026-11-25 更新。当前队列在 seed 123 的 15 组全部完成后自动暂停。进度见 [实验实况](RESEARCH_LIVE_STATUS.md) 和 [本地监测状态](../work/runs/research_2026/MONITOR_STATUS.json)。启动器检测到监测进程时沿用现有任务。暂停后的实验顺序见 [强基线实验计划](I10_EXPERIMENT_PLAN.md)。

在项目根目录使用：

```powershell
# 仅在 seed 123 尚有未完成组且监督器已退出时接续
& '.\scripts\start_research_2026.ps1' -Resume -StopAfterSeed 123 -HealthCheckHours 2
# 只更新报告，不训练、不打开测试集
& '.\work\envs\research_v1\Scripts\python.exe' -B -X utf8 scripts/update_research_report.py
```

当前阶段采用原探索方案的冻结配置。每个种子包含基线和 14 个变体，每组从零训练 40 轮。第二个种子完成后，依据自动生成的 `work/runs/research_2026/PAUSE_STATUS.json` 和 15 组完成标记核对收尾。

每个运行目录在 `work/runs/research_2026/`，含 config、环境、源码快照、逐轮日志、最佳权重、续训优化器及逐样本验证。普通断电/进程退出后从上个完整 epoch 恢复；不保证保留未结束的当前 epoch。不要手动覆盖运行中的源码或冻结 manifest，否则续训的来源检查将拒绝混用版本。

本地监测进程每分钟更新监测状态。当前每两小时记录训练日志进展、最近检查点、GPU 温度和显存以及磁盘空间。检查结果追加到 [巡检记录](../work/logs/research_2026/hourly_health.jsonl)。Windows 临时文件锁会先重试写入，若该错误仍导致任务退出，监测进程确认前一个训练进程结束后可接续，最多自动重试两次。

`attention_required` 包含退出码、日志及可用诊断。每组约需 115–145 MiB 持久检查点，新组开跑前留 512 MiB 供检查点及临时文件写入。机器重启后可用上述启动命令从最近保存的轮次恢复。监测期间请求 Windows 保持系统唤醒，退出后释放该请求。

冻结配置见 [followup_v2.json](../work/configs/screening_2026/followup_v2.json)。其中的第三个种子和旧全量复验后触发的洪水评测属于 [原探索方案](EXPERIMENT_PLAN.md)。新阶段先复现并诊断强基线，再为主对照和关键消融安排复验。
