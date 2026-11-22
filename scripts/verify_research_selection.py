"""Verify that degraded controls or harmful metrics cannot manufacture promotion."""
import json
from pathlib import Path
from continue_research_2026 import select_candidates,select_replicated

ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg={'minimum_gain_pct':1.,'maximum_candidates':2,'run_diagnostic_if_no_promotion':True,'diagnostic_candidate_count':2}
    good={'mean_patch_rmse':.02,'ssim':.9,'cc':.9,'sam_rad':.05,'ergas':1.}
    weakened={**good,'mean_patch_rmse':.025}
    false_win={**good,'method':'false_win','mean_patch_rmse':.024}
    bad_sam={**good,'method':'harmful','mean_patch_rmse':.018,'sam_rad':.06}
    actual_win={**good,'method':'actual_win','mean_patch_rmse':.0195}
    decision=select_candidates(good,weakened,[false_win,bad_sam,actual_win],cfg)
    assert decision['reference_rmse']==.02
    assert [r['method'] for r in decision['selected']]==['actual_win']
    assert not next(r for r in decision['all_candidates'] if r['method']=='false_win')['passes_screen']
    assert 'sam_rad' in next(r for r in decision['all_candidates'] if r['method']=='harmful')['harm_metrics']
    diagnostic=select_candidates(good,weakened,[false_win,bad_sam],cfg)
    assert diagnostic['mode']=='diagnostic_full_budget_no_positive_screen_claim'
    assert not any(r['passes_screen'] for r in diagnostic['selected'])
    cfg.update(seeds=[42,123,2026],quality_metric_relative_harm_limit_pct=1.)
    rows=[]
    for seed in cfg['seeds']:
        rows.extend([{**good,'method':'base','seed':seed},
                     {**actual_win,'seed':seed},
                     {**actual_win,'method':'one_seed_regression','seed':seed,'mean_patch_rmse':.0201 if seed==42 else .018},
                     {**bad_sam,'seed':seed}])
    replicated=select_replicated(rows,cfg)
    assert [r['method'] for r in replicated['selected']]==['actual_win']
    assert not next(r for r in replicated['all_candidates'] if r['method']=='one_seed_regression')['passes_replication']
    try:select_replicated(rows[:-1],cfg)
    except ValueError:pass
    else:raise AssertionError('Incomplete seeds accepted')
    report={'passed':True,'tests':['strong_epoch0_control','secondary_metric_harm','genuine_gain','diagnostic_is_not_promotion',
                                 'three_seed_paired_control','one_seed_regression_not_promoted','incomplete_seeds_rejected']}
    (ROOT/'docs/audit/research_selection_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: stronger baseline, quality gates, and diagnostic labeling')

if __name__=='__main__':main()
