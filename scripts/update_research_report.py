"""Render actual queue state without promoting exploratory numbers to a paper."""
from __future__ import annotations
import csv,json
import os
from datetime import datetime,timezone,timedelta
from pathlib import Path
from research_runtime import replace_with_retry

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'work/runs/research_2026'

def replication_rows():
    rows = []
    for marker in RUNS.glob('replicate_*/COMPLETE.json'):
        summary = marker.with_name('summary.json')
        if summary.exists():
            rows.append(json.loads(summary.read_text(encoding='utf-8')))
    return sorted(rows, key=lambda r: (r['seed'], r['method'] != 'base', r['mean_patch_rmse']))


def main():
    cfg=json.loads((ROOT/'work/configs/screening_2026/queue_v1.json').read_text(encoding='utf-8'))
    state=json.loads((RUNS/'QUEUE_STATUS.json').read_text(encoding='utf-8'))
    entries=[]
    basepath=RUNS/'screen_base_seed42_v1'
    initial=json.loads((basepath/'initial_validation.json').read_text(encoding='utf-8')) if (basepath/'initial_validation.json').exists() else {}
    reference=initial.get('mean_patch_rmse')
    if (basepath/'summary.json').exists():
        s=json.loads((basepath/'summary.json').read_text(encoding='utf-8'))
        reference=min(reference,s['mean_patch_rmse']) if reference else s['mean_patch_rmse']
    for job in state['jobs']:
        p=RUNS/job['run_id'];row={'method':job['method'],'state':job['state'],'epoch':0,
                                'best_validation_rmse':None,'strongest_base_rmse':reference,'improvement_vs_strongest_base_pct':None}
        if (p/'train_log.csv').exists():
            with (p/'train_log.csv').open(encoding='utf-8',newline='') as f:logs=list(csv.DictReader(f))
            if logs:row.update(epoch=int(logs[-1]['epoch']),best_validation_rmse=min(float(r['val_rmse']) for r in logs))
        if (p/'COMPLETE.json').exists():
            s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
            row.update(state='complete',best_validation_rmse=s['mean_patch_rmse'])
            if reference:row['improvement_vs_strongest_base_pct']=100*(reference-s['mean_patch_rmse'])/reference
        entries.append(row)
    out=ROOT/'work/results/research_2026';out.mkdir(parents=True,exist_ok=True)
    with (out/'live_screening.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    complete=sum(e['state']=='complete' for e in entries)
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    lines=['# 2026 研究实验实况','',f'更新：{stamp}；完成 {complete}/{len(entries)} 个初筛 run。',
           '', '**阶段：开发筛选。没有把这些结果作为 SOTA 或独立测试结论。**',
           '',f'共享起点的第 0 轮 RMSE：{initial.get("mean_patch_rmse", "待评测")}。',
           '收益相对于已知最强基础控制（包括第 0 轮），避免把微调退化的 base 用作较弱参照。',
           '尚未完成的行只显示进度，不报告最终增益。', '',
           '| 方法 | 实际状态 | 完成轮数 | 当前最好验证 RMSE | 完成后相对最强 base 改善 |',
           '|---|---|---:|---:|---:|']
    for e in entries:
        score='—' if e['best_validation_rmse'] is None else f'{e["best_validation_rmse"]:.7f}'
        gain='—' if e['improvement_vs_strongest_base_pct'] is None else f'{e["improvement_vs_strongest_base_pct"]:+.2f}%'
        lines.append(f'| {e["method"]} | {e["state"]} | {e["epoch"]}/{cfg["epochs"]} | {score} | {gain} |')
    lines+=['','仅以完整结束的 run 做探索性排序；次指标与噪声重复见各 run 的 summary.json。',
            '原始队列 screening_progress.csv 相对的是继续微调后的 base；本报告增加了更强的第 0 轮参照。',
            '', '下一阶段：充分训练、多种子、机制控制、CIA 与外部强对照；论文结果主张以最终冻结证据为准。']
    original=ROOT/'work/results/research_2026/original_base_control/summary.json'
    lines+=['', '第 0 轮基础模型多指标：'+('已完整导出。' if original.exists() else '尚待完整独立导出，未完成前不作全面优越性判断。')]
    follow=RUNS/'FOLLOWUP_STATUS.json'
    f={}
    results = replication_rows()
    if follow.exists():
        f=json.loads(follow.read_text(encoding='utf-8'))
        lines+=['', '## 有限后续执行状态', '',f'实际阶段：`{f["state"]}`；控制进程 PID：{f.get("pid", "未知")}。',
                f'已结束从零复验：{len(results)}/{f.get("total_replication_jobs",45)}；选择方式：`{f.get("decision_mode", "等待全部初筛")}`。']
        if f.get('current'):lines.append('当前复验：`'+json.dumps(f['current'],ensure_ascii=False)+'`。')
        if f.get('error'):lines.append('需处理的问题：'+f['error'])
    monitor=RUNS/'MONITOR_STATUS.json'
    if monitor.exists():
        m=json.loads(monitor.read_text(encoding='utf-8'))
        lines+=['',f'本地监测：{m["state"]}，状态刷新于 {m["checked_cst"]}。详见 [运行状态](../work/runs/research_2026/MONITOR_STATUS.json) 和 [巡检记录](../work/logs/research_2026/hourly_health.jsonl)。']
        if m.get('next_health_check_cst'):
            lines.append(f'下次巡检：{m["next_health_check_cst"]}。')
    pause_path = RUNS / 'PAUSE_STATUS.json'
    if pause_path.exists():
        pause = json.loads(pause_path.read_text(encoding='utf-8'))
        lines += ['', '## 暂停状态', '']
        if pause.get('mode') == 'after_run':
            lines.append(f'{pause["target_run_id"]} 完成后暂停。[暂停记录](../work/runs/research_2026/PAUSE_STATUS.json)')
        else:
            lines.append(f'计划暂停时间 {pause["scheduled_for_cst"]}。[暂停记录](../work/runs/research_2026/PAUSE_STATUS.json)')
        if pause.get('paused_cst'):
            lines.append(f'实际暂停时间 {pause["paused_cst"]}。')
        boundary = pause.get('boundary')
        if boundary:
            lines.append(f'{pause["run_id"]} 已保存至第 {boundary["epoch"]} 轮。')
    if results:
        reference_by_seed = {r['seed']: r['mean_patch_rmse'] for r in results if r['method'] == 'base'}
        lines += ['', '## 从零复验阶段结果', '',
                  '以下是完成 40 轮并导出验证结果的组。降幅为相对同种子基线的验证 RMSE 降幅，正值表示改善。',
                  '', '| 方法 | 训练种子 | 最佳轮 | RMSE ↓ | 相对基线降幅 | SSIM ↑ | SAM ↓ (rad) | ERGAS ↓ | CC ↑ |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in results:
            reference = reference_by_seed.get(r['seed'])
            change = f'{100 * (reference - r["mean_patch_rmse"]) / reference:+.2f}%' if reference and r['method'] != 'base' else '—'
            metrics = ' | '.join(f'{r[k]:.5f}' if r.get(k) is not None else '—' for k in ('ssim','sam_rad','ergas','cc'))
            lines.append(f'| {r["method"]} | {r["seed"]} | {r["best_epoch"]} | {r["mean_patch_rmse"]:.7f} | {change} | {metrics} |')
        lines += ['', '这些是 LGC 开发验证结果。多种子和外部测试结论待相应实验完成。']
    report=ROOT/'docs/RESEARCH_LIVE_STATUS.md'
    temp_report=report.with_suffix(f'.{os.getpid()}.tmp')
    temp_report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    replace_with_retry(temp_report,report)
    plan=ROOT/'docs/EXPERIMENT_QUEUE.csv'
    if plan.exists():
        with plan.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames;milestones=list(reader)
        lookup={f'work/runs/research_2026/{j["run_id"]}':j['state'] for j in state['jobs']}
        for row in milestones:
            if row['artifact'] in lookup:row['status']=lookup[row['artifact']]
            if row['id']=='R15':
                row['status']='complete' if len(results)==45 else f.get('state','queued_after_screening')
            if row['id']=='R16':row['status']='complete_validation_selection' if (RUNS/'FOLLOWUP_DECISION.json').exists() else 'pending_all_45_runs'
            if row['id']=='R17':
                external=ROOT/'work/results/research_2026/common_flood_v1/STATUS.json'
                row['status']=json.loads(external.read_text(encoding='utf-8'))['state'] if external.exists() else 'pending_postselection'
        temporary=plan.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(milestones)
        replace_with_retry(temporary,plan)
    print(f'Updated live report: {complete}/{len(entries)} complete; queue={state["state"]}')

if __name__=='__main__':main()
