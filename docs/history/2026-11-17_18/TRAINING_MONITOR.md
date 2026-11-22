
## 2026-11-18 16:26 即时巡检（手动触发）
- full: 40/40 完成，BEST 0.05327（已评测，淘汰：val 比 base 差 11.7%）
- delta: 40/40 完成，BEST 0.05221（**尚未评测**，待评测）
- gate: 28/40 进行中，ep28 val 0.05300，日志刚刚更新（正常）
- reinject: 未开始（排队）
- delsup: 未开始（全跑完后）
- GPU: 3374MiB / 61%  磁盘: 11GB 可用
- 异常: 无。`ps aux` 显示 0 个训练进程是 Windows 下 Git Bash 看不到子进程所致，日志实时更新 + GPU 占用证明训练正常，不重启。

## 2026-11-18 17:28:30 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 40/40 epoch，最后 ep 40 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（235.0 分钟前）
- **delta** (log_arch_delta.txt): 40/40 epoch，最后 ep 40 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（131.5 分钟前）
- **gate** (log_arch_gate.txt): 40/40 epoch，最后 ep 40 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（29.0 分钟前）
- **reinject** (log_arch_reinject.txt): 10/40 epoch，最后 ep 10 loss 0.0228 val 0.05800，无 BEST 行；日志更新 2026-11-18 17:25:58（2.5 分钟前）
- **delsup** (log_train_delsup.txt): 日志不存在（尚未开始）
### GPU / 磁盘
- GPU: used 3246MiB / total 6141MiB, util 61%
- 磁盘: D: 可用 9.15GB / 总 263.67GB
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  255G  9.2G  97% /d
### 异常与处理
- 未发现异常（卡死 / 断链）。
- 磁盘可用空间充足（>=3GB），无需清理。

## 2026-11-18 18:30:35 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 40/40 完成，最后 ep 40 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（297.1 分钟前，已完成无需更新）
- **delta** (log_arch_delta.txt): 40/40 完成，最后 ep 40 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（193.6 分钟前，已完成无需更新）
- **gate** (log_arch_gate.txt): 40/40 完成，最后 ep 40 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（91.1 分钟前，已完成无需更新）
- **reinject** (log_arch_reinject.txt): 34/40 进行中，最后 ep 34 loss 0.0154 val 0.05639，无 BEST 行；日志更新 2026-11-18 18:29:17（1.3 分钟前，训练活跃），较上次巡检 10/40 已推进 24 epoch
- **delsup** (log_train_delsup.txt): 日志不存在（尚未开始）
### GPU / 磁盘
- GPU: used 3247MiB / total 6141MiB, util 85~99%（训练活跃）
- 磁盘: D: 可用 9.14GB / 总 263.67GB，使用率 96.5%
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  255G  9.2G  97% /d
### 异常与处理
- 卡死检测：无。唯一未完成的 reinject（34/40）日志 1.3 分钟前刚更新，远未达 20 分钟阈值。
- 断链检测：无。full/delta/gate 均有 BEST 行且后继日志存在；reinject 尚无 BEST（未完成），其后的 delsup 未启动属正常排队，非断链。
- 磁盘可用 9.14GB（>=3GB），未触发清理，未截断/删除任何文件。
- 未重启任何训练任务，未改变实验顺序或超参数。
### 提醒
- gate 已完成但 reinject 尚未完成（34/40），故"gate+reinject 均完成但 delsup 未跑"提醒暂不触发。
- 全部实验尚未完成，暂不能定稿。

## 2026-11-18 19:32:31 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 40/40 完成，最后 ep 40 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（359.0 分钟前，已完成无需更新）
- **delta** (log_arch_delta.txt): 40/40 完成，最后 ep 40 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（255.5 分钟前，已完成无需更新）
- **gate** (log_arch_gate.txt): 40/40 完成，最后 ep 40 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（153.0 分钟前，已完成无需更新）
- **reinject** (log_arch_reinject.txt): 40/40 完成，最后 ep 40 loss 0.0151 val 0.05658，BEST 0.05245439840138597；日志更新 2026-11-18 18:45:08（47.4 分钟前，已完成无需更新）。较上次巡检 34/40 已跑完最后 6 epoch 并产出 BEST。
- **delsup** (log_train_delsup.txt): 16/40 进行中，最后 ep 16 loss 0.0416 val 0.04764，无 BEST 行；日志更新 2026-11-18 19:30:48（1.7 分钟前，训练活跃）。日志已出现，训练链按序推进到最后一个模型。
### GPU / 磁盘
- GPU: used 3191MiB / total 6141MiB, util 99%（训练活跃）
- 磁盘: D: 可用 7.4GB / 总 264GB，使用率 98%（较上次 9.14GB 下降约 1.7GB，仍充足，持续留意）
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  257G  7.4G  98% /d
### 异常与处理
- 卡死检测：无。唯一未完成的 delsup（16/40）日志 1.7 分钟前刚更新，远未达 20 分钟阈值；其余 4 个模型均 40/40 完成。
- 断链检测：无。full/delta/gate/reinject 均有 BEST 行且后继日志依次出现（full→delta→gate→reinject→delsup），链完整。
- 磁盘可用 7.4GB（>=3GB），未触发清理，未截断/删除任何文件；未重启任何训练任务，未改变实验顺序或超参数。
- 未触碰用户曾取消的 onestep、lfnet 任务。
### 提醒
- "gate+reinject 均完成但 delsup 未跑"提醒不触发：gate、reinject 均已完成，且 delsup 日志已出现并推进到 16/40，训练链正常进入最后一个模型，无需提醒。
- 全部实验尚未完成（delsup 16/40），暂不能运行 tools/final_aggregate.py 定稿；待 delsup 跑满 40 epoch 并产出 BEST 后再汇总。

## 2026-11-18 20:34:06 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 40/40 完成，最后 ep 40 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（420.6 分钟前，已完成无需更新）
- **delta** (log_arch_delta.txt): 40/40 完成，最后 ep 40 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（317.1 分钟前，已完成无需更新）
- **gate** (log_arch_gate.txt): 40/40 完成，最后 ep 40 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（214.6 分钟前，已完成无需更新）
- **reinject** (log_arch_reinject.txt): 40/40 完成，最后 ep 40 loss 0.0151 val 0.05658，BEST 0.05245439840138597；日志更新 2026-11-18 18:45:08（109.0 分钟前，已完成无需更新）
- **delsup** (log_train_delsup.txt): 40/40 完成，最后 ep 40 loss 0.0368 val 0.04948，BEST 0.04728194893395739；日志更新 2026-11-18 20:31:37（2.5 分钟前）。较上次巡检 16/40 已跑完最后 24 epoch 并产出 BEST。
### GPU / 磁盘
- GPU: used 727MiB / total 6141MiB, util 23%（训练已全部结束，GPU 基本空闲）
- 磁盘: D: 可用 7.4GB / 总 264GB，使用率 98%（与上次 7.4GB 持平，未继续下降）
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  257G  7.4G  98% /d
### 异常与处理
- 卡死检测：无。5 个模型均 40/40 完成并产出 BEST；delsup 最后更新 2.5 分钟前，属正常收尾。其余 4 个日志 mtime 虽超 20 分钟，但 epoch 已达 40（训练完成），不判定卡死。
- 断链检测：无。full→delta→gate→reinject→delsup 五个日志依次出现且均有 BEST 行，训练链完整闭合。
- 磁盘可用 7.4GB（>=3GB），未触发清理，未截断/删除任何文件；未重启任何任务，未改变实验顺序或超参数。
- 未触碰用户曾主动取消的 onestep、lfnet 任务。
### 提醒
- "gate+reinject 均完成但 delsup 未跑"提醒不触发：delsup 已跑完 40 epoch 并产出 BEST。
- **全部实验（full/delta/gate/reinject/delsup）均已完成！可以运行 `python tools/final_aggregate.py` 汇总并定稿了。**（脚本 tools/final_aggregate.py 已确认存在）
## 2026-11-18 21:15:52 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 完成，最后 ep 040 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（462.4 分钟前）。
- **delta** (log_arch_delta.txt): 完成，最后 ep 040 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（358.9 分钟前）。
- **gate** (log_arch_gate.txt): 完成，最后 ep 040 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（256.4 分钟前）。
- **reinject** (log_arch_reinject.txt): 完成，最后 ep 040 loss 0.0151 val 0.05658，BEST 0.05245439840138597；日志更新 2026-11-18 18:45:08（150.7 分钟前）。
- **delsup** (log_train_delsup.txt): 完成，最后 ep 040 loss 0.0368 val 0.04948，BEST 0.04728194893395739；日志更新 2026-11-18 20:31:37（44.2 分钟前）。
### GPU / 磁盘
- GPU: used 846MiB / total 6141MiB, util 9%（训练已全部结束，GPU 空闲）
- 磁盘: D: 可用 7.4GB / 总 264GB，使用率 98%（与上次 7.4GB 持平，未继续下降）
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  257G  7.4G  98% /d
### 异常与处理
- 卡死检测：无。5 个模型均 40/40 完成并产出 BEST；各日志 mtime 虽超 20 分钟，但 epoch 均已达 40（训练正常完成），不判定卡死。
- 断链检测：无。full→delta→gate→reinject→delsup 五个日志依次出现且均有 BEST 行，训练链完整闭合。
- 磁盘可用 7.4GB（>=3GB），未触发清理，未截断/删除任何文件；未重启任何任务，未改变实验顺序或超参数。
- 未触碰用户曾主动取消的 onestep、lfnet 任务。
### 提醒
- "gate+reinject 均完成但 delsup 未跑"提醒不触发：delsup 已跑完 40 epoch 并产出 BEST。
- **全部实验（full/delta/gate/reinject/delsup）均已完成并保持完成状态！可以运行 `python tools/final_aggregate.py` 汇总并定稿了。**（脚本已确认存在；较上次 20:34 无新增变化）
## 2026-11-18 21:35:49 自动巡检（训练任务健康监控）
### 各模型进度（预期 40 epoch）
- **full** (log_arch_full.txt): 完成，最后 ep 040 loss 0.0148 val 0.05342，BEST 0.05327242584326076；日志更新 2026-11-18 13:33:29（482.1 分钟前）。
- **delta** (log_arch_delta.txt): 完成，最后 ep 040 loss 0.0151 val 0.05373，BEST 0.05220732670185858；日志更新 2026-11-18 15:17:01（378.6 分钟前）。
- **gate** (log_arch_gate.txt): 完成，最后 ep 040 loss 0.0153 val 0.05305，BEST 0.05113424188743622；日志更新 2026-11-18 16:59:28（276.2 分钟前）。
- **reinject** (log_arch_reinject.txt): 完成，最后 ep 040 loss 0.0151 val 0.05658，BEST 0.05245439840138597；日志更新 2026-11-18 18:45:08（170.5 分钟前）。
- **delsup** (log_train_delsup.txt): 完成，最后 ep 040 loss 0.0368 val 0.04948，BEST 0.04728194893395739；日志更新 2026-11-18 20:31:37（64.0 分钟前）。
### GPU / 磁盘
- GPU: used 755MiB / total 6141MiB, util 22%（训练已全部结束，GPU 基本空闲）
- 磁盘: D: 可用 7.4GB / 总 264GB，使用率 98%（与上次 7.4GB 持平，未继续下降）
- df -h /d:
  Filesystem      Size  Used Avail Use% Mounted on
  D:              264G  257G  7.4G  98% /d
### 异常与处理
- 卡死检测：无。5 个模型均 40/40 完成并产出 BEST；各日志 mtime 虽超 20 分钟，但 epoch 均已达 40（训练正常完成），不判定卡死。
- 断链检测：无。full→delta→gate→reinject→delsup 五个日志依次出现且均有 BEST 行，训练链完整闭合。
- 磁盘可用 7.4GB（>=3GB），未触发清理，未截断/删除任何文件；未重启任何任务，未改变实验顺序或超参数。
- 未触碰用户曾主动取消的 onestep、lfnet 任务。
### 提醒
- "gate+reinject 均完成但 delsup 未跑"提醒不触发：delsup 已跑完 40 epoch 并产出 BEST。
- **全部实验（full/delta/gate/reinject/delsup）均已完成并保持完成状态！可以运行 `python tools/final_aggregate.py` 汇总并定稿了。**（脚本已确认存在；较上次 21:15 巡检无新增变化）
