# 恢复实验备份

## 复原代码和运行环境

使用 Windows、Python 3.12.10 和 PyTorch 2.5.1（CUDA 12.1 wheel）。训练虚拟环境建立在项目的 work/envs/research_v1 下。已有等价 Torch 环境时执行后两行。

```powershell
py -3.12 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
py -3.12 -m venv --system-site-packages work/envs/research_v1
& '.\work\envs\research_v1\Scripts\python.exe' -m pip install -r work/configs/research_v1-requirements.txt
```

在干净的仓库克隆中恢复外部代码。STF、STFMamba 和 SA-STF 直接从原仓库取得并固定到已登记的 commit。STF 的项目扩展放在 work/upstream_overlays，恢复器检查训练使用的骨干字节哈希。SA-STF 使用其上游 MIT 许可，STF 与 STFMamba 的上游仓库未声明可再分发许可，引用固定源码来源。

```powershell
& '.\work\envs\research_v1\Scripts\python.exe' -B -X utf8 scripts/restore_upstreams.py
```

## 复原影像输入

从官方 LGC 和 CIA 集合获取主影像，将对应目录分别置于 work/data/raw/LGC 和 work/data/raw_cia/CIA。下载文件的层级、字节数及哈希见 [原始数据校验表](audit/raw_file_integrity.csv)。数据版本应与清单一致。保留的本地 ZIP 可直接解压到克隆根目录，也可以从原始主影像重建切片。掩膜用于另外的区域评测，可从原集合单独取得。 [LGC 集合](https://data.csiro.au/collection/csiro:5847) · [CIA 集合](https://data.csiro.au/collection/csiro:5846)

```powershell
$researchPython = '.\work\envs\research_v1\Scripts\python.exe'
# 从官方影像生成首种子训练所需的 LGC 切片，并逐文件核对哈希
& $researchPython -B -X utf8 scripts/restore_frozen_data.py --dataset LGC --split train --split val --generate
# 使用本地切片副本时直接核对
& $researchPython -B -X utf8 scripts/restore_frozen_data.py --dataset CIA
```

恢复器使用冻结清单的日期、坐标和二进制布局。省略 split 可选择该数据集的所有切片，加入 generate 时对已存在的切片只做核对。

## 恢复检查点与接续实验

从对应 GitHub Release 下载检查点 ZIP。用 Get-FileHash 核对 ZIP 的 SHA-256，核对值存于 [备份状态](../work/runs/research_2026/BACKUP_STATE.json)。将 ZIP 解压到克隆根目录，保留其 work/runs/research_2026 相对路径。完成组的最佳模型与摘要标记配对，进行中实验另有包含优化器的 last saved epoch。当前完成快照含 15 个最佳权重及 I14 的第 40 轮续训状态。如需恢复早期进度，请检出该 Release 对应的源码标签再解包。首次快照的 I04 从第 10 轮接续。

```powershell
& '.\scripts\start_research_2026.ps1' -Resume -StopAfterSeed 42 -HealthCheckHours 2
```

控制器读取已完成标记、跳过已完成组，从最近的完整轮次继续当前组。完成 seed 42 后写入暂停状态。以仓库快照覆盖正在运行的工作区会混合不同时间的状态，请从独立克隆恢复。
