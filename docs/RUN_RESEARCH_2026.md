# 实验操作入口

2026-11-21 从第 12 组初筛接续。进度见 [实验实况](RESEARCH_LIVE_STATUS.md) 和 [本地监测状态](../work/runs/research_2026/MONITOR_STATUS.json)。启动器检测到监测进程时沿用现有任务。

在项目根目录使用：

```powershell
# 查看/启动/断点恢复本次有限实验流程
& '.\scripts\start_research_2026.ps1'
# 只更新报告，不训练、不打开测试集
& '.\work\envs\research_v1\Scripts\python.exe' -B -X utf8 scripts/update_research_report.py
```

初筛 15 组 → 基线与全部 14 个方向从零训练 40 轮、三个种子（45 组）→ 依验证结果选最多两个外部候选 → 共同 flood patch 评测。外部测试在充分训练和验证选型后访问。

每个运行目录在 `work/runs/research_2026/`，含 config、环境、源码快照、逐轮日志、最佳权重、续训优化器及逐样本验证。普通断电/进程退出后从上个完整 epoch 恢复；不保证保留未结束的当前 epoch。不要手动覆盖运行中的源码或冻结 manifest，否则续训的来源检查将拒绝混用版本。

本地监测进程依次运行初筛与复验。它每分钟更新监测状态，每小时记录训练日志进展、最近检查点、GPU 温度和显存以及磁盘空间。检查结果追加到 [每小时记录](../work/logs/research_2026/hourly_health.jsonl)。Windows 临时文件锁会先重试写入，若该错误仍导致任务退出，监测进程确认前一个训练进程结束后可接续，最多自动重试两次。

`attention_required` 包含退出码、日志及可用诊断。每组约需 115–145 MiB 持久检查点，新组开跑前留 512 MiB 供检查点及临时文件写入。机器重启后可用上述启动命令从最近保存的轮次恢复。监测期间请求 Windows 保持系统唤醒，退出后释放该请求。

2026-11-22 晚上的暂停条件更新为 I05（种子 42）完整训练并导出验证结果后暂停。暂停结果记录在 [暂停状态](../work/runs/research_2026/PAUSE_STATUS.json)，当时已完成的配对结果见 [实验实况](RESEARCH_LIVE_STATUS.md)。需要继续时，在项目根目录执行 `.\scripts\start_research_2026.ps1 -Resume`。

当前后续配置是 [followup_v2.json](../work/configs/screening_2026/followup_v2.json)，全量训练粗估 110–140 GPU 小时，CIA 域内、机制消融和同训练协议外部主表仍需后续完成。已有论文工作稿只包含可证实的任务和协议，不填虚构结果。
