"""Compare the three-epoch replays made on AutoDL (2026-11-11) with the stored records, field by field.

Replays (downloaded to 02_当前实验/原始证据/train_20261109/replay_autodl_20261111/server):
  vastjob_*            a job trained on Vast in November, started again on AutoDL with the same command
  vastjob_*__avx2sort  the same, with numpy's AVX-512 kernels disabled (NPY_DISABLE_CPU_FEATURES), so that
                       numpy sorts with its AVX2 kernel; nothing else differs
  septjob_*            a job trained on AutoDL in September, started again on AutoDL (now a different host)
Reference: the results.csv of the stored run. Every column except the elapsed time is compared as text for the
first three epochs, and the columns are also grouped: training (losses and learning rates), validation losses,
validation metrics. Output: replay_comparison.txt and replay_comparison.json next to the downloaded replays.
Usage: python compare_replay.py     (standard library only)
"""
import csv, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
EVID = Path('D:/ring3/02_当前实验/原始证据')
TRAIN = EVID / 'train_20261109'
REP = TRAIN / 'replay_autodl_20261111'
SERVER = REP / 'server'
VAST = TRAIN / 'results/extracted/root/autodl-tmp'
V_BASE = VAST / 'refsoft_extension_20261109/release_v1/runs/WSODD_baseline_seed2/results.csv'
V_CBAM = VAST / 'refsoft_extension_20261109/release_v1/runs/WSODD_cbam_only_seed0/results.csv'
V_PRE = VAST / 'pretrained_copies_20261109/runs/WSODD_pretrained_clean_seed0/results.csv'
V_FLOW = VAST / 'flow_source_retrain_20260920/release_v1/runs/FloW_E2C_seed1/results.csv'
PAIRS = [
    ('septjob_WSODD_bundle_seed0', 'AutoDL, September', EVID / 'refsoft_reproduction_20260910/final_evidence_20260913/runs/WSODD_bundle_seed0/results.csv'),
    ('septjob_WSODD_E0_seed42', 'AutoDL, September', SERVER / 'september_reference/WSODD_E0_seed42/results.csv'),
    ('vastjob_WSODD_baseline_seed2', 'Vast, November', V_BASE),
    ('vastjob_WSODD_cbam_only_seed0', 'Vast, November', V_CBAM),
    ('vastjob_WSODD_pretrained_clean_seed0', 'Vast, November', V_PRE),
    ('vastjob_FloW_E2C_seed1', 'Vast, November', V_FLOW),
    ('vastjob_WSODD_baseline_seed2__avx2sort', 'Vast, November', V_BASE),
    ('vastjob_WSODD_pretrained_clean_seed0__avx2sort', 'Vast, November', V_PRE),
    ('vastjob_FloW_E2C_seed1__avx2sort', 'Vast, November', V_FLOW),
]
EPOCHS = 3
GROUPS = {'training': ('train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2'),
          'validation losses': ('val/box_loss', 'val/cls_loss', 'val/dfl_loss'),
          'validation metrics': ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)')}


def rows(path):
    return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(path, encoding='utf-8'))]


out, lines = {}, []
for name, where, ref_path in PAIRS:
    rep_path = SERVER / name / 'results.csv'
    if not rep_path.exists():
        out[name] = dict(status='no replay record'); lines.append(f'{name}: no replay record'); continue
    rep, ref = rows(rep_path), rows(ref_path)
    n = min(EPOCHS, len(rep))
    cols = [c for c in ref[0] if c != 'time']
    assert set(cols) == {'epoch'} | {c for g in GROUPS.values() for c in g}, cols
    diffs = [dict(epoch=i + 1, column=c, replay=rep[i].get(c), reference=ref[i][c], abs_difference=abs(float(rep[i][c]) - float(ref[i][c])))
             for i in range(n) for c in cols if rep[i].get(c) != ref[i][c]]
    by_group = {g: dict(fields=n * len(cs), differing=sum(d['column'] in cs for d in diffs),
                        largest_abs_difference=max([d['abs_difference'] for d in diffs if d['column'] in cs], default=0.0)) for g, cs in GROUPS.items()}
    m = [d['abs_difference'] for d in diffs if d['column'] == 'metrics/mAP50-95(B)']
    out[name] = dict(reference=where, reference_file=str(ref_path), epochs_compared=n, fields_compared=n * len(cols), fields_differing=len(diffs),
                     identical=bool(n == EPOCHS and set(rep[0]) == set(ref[0]) and not diffs), groups=by_group,
                     largest_abs_difference_mAP50_95=max(m, default=0.0), differences=diffs)
    lines.append(f"{name} against {where}: {n * len(cols)} fields, differing {len(diffs)}" + ('  -> identical' if out[name]['identical'] else '  -> not identical'))
    for g, v in by_group.items():
        lines.append(f"    {g}: {v['differing']} of {v['fields']} differ" + (f", largest absolute difference {v['largest_abs_difference']:.5f}" if v['differing'] else ''))
    if m:
        lines.append(f"    validation mAP50-95: largest absolute difference {max(m):.5f} ({100 * max(m):.3f} pp)")
    for i in range(n):
        lines.append(f"    epoch {i + 1}  replay    " + ' '.join(rep[i].get(c, '?') for c in cols[1:]))
        lines.append(f"    epoch {i + 1}  reference " + ' '.join(ref[i][c] for c in cols[1:]))
done = {k: v for k, v in out.items() if 'identical' in v}
lines.append('summary')
for kind, label in (('septjob', 'September jobs replayed on AutoDL'), ('vastjob', 'Vast jobs replayed on AutoDL')):
    for ctl in (False, True):
        sel = [v for k, v in done.items() if k.startswith(kind) and k.endswith('__avx2sort') == ctl]
        if sel:
            lines.append(f"  {label}{' with numpy AVX-512 kernels disabled' if ctl else ''}: {sum(v['identical'] for v in sel)} of {len(sel)} identical in every field; "
                         f"training fields differing {sum(v['groups']['training']['differing'] for v in sel)} of {sum(v['groups']['training']['fields'] for v in sel)}; "
                         f"validation fields differing {sum(v['groups'][g]['differing'] for v in sel for g in ('validation losses', 'validation metrics'))} of "
                         f"{sum(v['groups'][g]['fields'] for v in sel for g in ('validation losses', 'validation metrics'))}")
(REP / 'replay_comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(REP / 'replay_comparison.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('\n'.join(lines))
