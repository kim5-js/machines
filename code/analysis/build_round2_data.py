"""Round-2 data of revision 20261109: multi-seed strategies, matched controls and the added controls. CPU only.

Inputs (read only)
  sealed unified evaluation CSV (基础依赖/20260924_统一评价修订/unified_evaluations.csv)
  02_当前实验/原始证据/recompute_20261109/vast_results/output/evaluations/*/evaluation.json
  02_当前实验/原始证据/train_20261109/results/extracted/output/evaluations/*/evaluation.json
  FloW training records (runtime_evidence.json) for the small-object weighting share
Outputs (version 20261109 only)
  正文源/20261109/tables/round2_values.json       every summary used by the text, full precision
  正文源/20261109/tables/table4_strategies.csv, table5_matched.csv, table1_coverage.csv
  图表/20261109/绘图数据/paired_effects.json       per-seed paired differences for Figures 7 and 8
All three record sets were scored by the same evaluator (hash checked). A key present in an earlier set is never replaced.
Run:  python build_round2_data.py      (scientific environment; needs scipy)
"""
import collections, csv, glob, json, math, statistics, sys
from pathlib import Path
from scipy import stats as st

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3')
PAPER = RING3 / '01_当前论文'
EVID = RING3 / '02_当前实验/原始证据'
SEALED = PAPER / '基础依赖/20260924_统一评价修订/unified_evaluations.csv'
REC_EV = EVID / 'recompute_20261109/vast_results/output/evaluations'
NEW_EV = EVID / 'train_20261109/results/extracted/output/evaluations'
FLOW_OLD = EVID / 'flow_source_retrain_20260920/final_recovery/evidence/runs'
FLOW_NEW = EVID / 'train_20261109/results/extracted/root/autodl-tmp/flow_source_retrain_20260920/release_v1/runs'
TABLES = PAPER / '正文源/20261109/tables'
PLOT = PAPER / '图表/20261109/绘图数据'
EVALUATOR = '988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267'
M = ('mAP50-95', 'APsmall', 'APmedium', 'APlarge', 'mAP50')
CONDS = ('normal', 'low_light_strong', 'low_light_L4', 'low_light_L5', 'glare_strong')
S6 = ['0', '1', '2', '42', '123', '3407']
S3 = ['0', '1', '2']
ARM = {'A_baseline': 'A', 'B_weights_only': 'B', 'C_CBAM_Focal_EIoU': 'C', 'D_clean_duplicate': 'D', 'E_refsoft': 'E',
       'baseline': 'A', 'weights_only': 'B', 'bundle': 'C', 'duplicate': 'D', 'refsoft': 'E', 'cbam_only': 'B+CBAM', 'focal_only': 'B+Focal',
       'pretrained_clean': 'P-clean', 'pretrained_glaresoft': 'P-glare'}

val, src = {}, {}
counts = collections.Counter()
weights = collections.defaultdict(set)
for r in csv.DictReader(open(SEALED, encoding='utf-8-sig')):
    assert r['evaluator_code_sha256'] == EVALUATOR
    counts['sealed records'] += 1; weights['sealed'].add(r['checkpoint_sha256'])
    if r['family'] in ('matched', 'strategy'):
        k = (r['dataset'], ARM.get(r['arm_raw'], r['arm_raw']), r['seed'], r['condition_raw'])
        val[k] = {m: 100 * float(r[m + '_fraction']) for m in M}; src[k] = 'sealed'
for p in glob.glob(str(REC_EV / '*/evaluation.json')):
    e = json.load(open(p, encoding='utf-8')); j = e['job']
    assert e['status'] == 'completed' and e['evaluator_code_sha256'] == EVALUATOR
    if j.get('block') == 'gate':
        counts['repeat evaluations of sealed keys (not counted)'] += 1; continue
    last = '_last__' in j['key']
    counts['records added for conditions and last.pt'] += 1
    weights['last.pt' if last else 'sealed'].add(j.get('weights_sha256') or j['weights'])
    if last:
        continue
    k = (j['dataset'], ARM.get(j['arm'], j['arm']), str(j['seed']), j['condition'])
    if k not in val:
        val[k] = {m: 100 * e['metrics'][m] for m in M}; src[k] = 'conditions'
for p in glob.glob(str(NEW_EV / '*/evaluation.json')):
    e = json.load(open(p, encoding='utf-8')); j = e['job']
    assert e['status'] == 'completed' and e['evaluator_code_sha256'] == EVALUATOR
    base, seed = j['experiment_id'].rsplit('_seed', 1); ds, arm = base.split('_', 1)
    k = (ds, ARM.get(arm, arm), seed, j['condition']); assert k not in val, k
    val[k] = {m: 100 * e['metrics'][m] for m in M}; src[k] = 'added runs'
    counts['records of the added runs'] += 1; weights['added'].add(j['weights_sha256'])


def summ(v):
    n = len(v); mean = sum(v) / n; sd = statistics.stdev(v); hw = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci_low=mean - hw, ci_high=mean + hw, positive=sum(x > 0 for x in v), negative=sum(x < 0 for x in v),
                excludes_zero=bool(mean - hw > 0 or mean + hw < 0), per_seed=v)


def seeds(ds):
    return S3 if ds == 'SeaDronesSee' else S6


def arm_stats(ds, arm, cond='normal'):
    return {m: summ([val[(ds, arm, s, cond)][m] for s in seeds(ds)]) for m in M}


def contrast(ds, new, ref, cond='normal'):
    out = {}
    for m in M:
        d = [val[(ds, new, s, cond)][m] - val[(ds, ref, s, cond)][m] for s in seeds(ds)]
        row = dict(diff=summ(d))
        if cond != 'normal':
            n0 = [val[(ds, new, s, 'normal')][m] - val[(ds, ref, s, 'normal')][m] for s in seeds(ds)]
            row['interaction'] = summ([a - b for a, b in zip(d, n0)])
        out[m] = row
    return out


ARMS = [('SeaDronesSee', 'E0', 'Standard training'), ('SeaDronesSee', 'E1', 'Fixed single view'),
        ('WSODD', 'E0', 'Standard training'), ('WSODD', 'E1', 'Fixed single view'),
        ('FloW', 'E0', 'Standard training'), ('FloW', 'E2', 'Paired views'), ('FloW', 'E3', 'Paired views + small-object weighting'),
        ('FloW', 'E2C', 'Identical paired views'),
        ('WSODD', 'A', 'Scratch reference'), ('WSODD', 'B', 'Adjusted loss weights'), ('WSODD', 'B+CBAM', 'Attention only'),
        ('WSODD', 'B+Focal', 'Localization loss only'), ('WSODD', 'C', 'Attention + localization loss'), ('WSODD', 'D', 'Clean copies'),
        ('WSODD', 'E', 'GlareSoft copies'), ('WSODD', 'P-clean', 'Pretrained + clean copies'), ('WSODD', 'P-glare', 'Pretrained + GlareSoft copies')]
CONTRASTS = [('SeaDronesSee', 'E1', 'E0', 'strategy', 'Fixed single view'), ('WSODD', 'E1', 'E0', 'strategy', 'Fixed single view'),
             ('FloW', 'E2', 'E0', 'strategy', 'Paired views'), ('FloW', 'E3', 'E0', 'strategy', 'Paired views + small-object weighting'),
             ('WSODD', 'B', 'A', 'chain', 'Loss-weight adjustment'), ('WSODD', 'C', 'B', 'chain', 'Attention + localization-loss addition'),
             ('WSODD', 'D', 'C', 'chain', 'Clean-copy addition'), ('WSODD', 'E', 'D', 'chain', 'Matched localized glare increment'),
             ('WSODD', 'E', 'C', 'chain', 'GlareSoft vs component configuration'), ('WSODD', 'E', 'A', 'chain', 'Complete configuration vs scratch reference'),
             ('WSODD', 'B+CBAM', 'B', 'component', 'Attention only vs adjusted weights'), ('WSODD', 'B+Focal', 'B', 'component', 'Localization loss only vs adjusted weights'),
             ('WSODD', 'C', 'B+CBAM', 'component', 'Localization loss added to attention'), ('WSODD', 'C', 'B+Focal', 'component', 'Attention added to localization loss'),
             ('WSODD', 'P-glare', 'P-clean', 'pretrained', 'Pretrained: GlareSoft copies vs clean copies'),
             ('WSODD', 'P-clean', 'E0', 'pretrained', 'Pretrained: clean copies vs standard training'),
             ('WSODD', 'P-glare', 'E0', 'pretrained', 'Pretrained: GlareSoft copies vs standard training'),
             ('FloW', 'E2', 'E2C', 'view', 'Paired views vs identical paired views'), ('FloW', 'E2C', 'E0', 'view', 'Identical paired views vs standard training'),
             ('FloW', 'E3', 'E2', 'view', 'Small-object weighting vs paired views')]
values = dict(arms={}, contrasts={}, conditions={})
for ds, arm, label in ARMS:
    assert all((ds, arm, s, 'normal') in val for s in seeds(ds)), (ds, arm)
    values['arms'][f'{ds}|{arm}'] = dict(label=label, seeds=seeds(ds), **arm_stats(ds, arm))
for ds, new, ref, group, label in CONTRASTS:
    key = f'{ds}|{new}|{ref}'
    values['contrasts'][key] = dict(label=label, group=group, seeds=seeds(ds), **contrast(ds, new, ref))
    if ds != 'SeaDronesSee':
        values['conditions'][key] = {c: contrast(ds, new, ref, c) for c in CONDS[1:]}
did = {}
for m in M:
    x = [(val[('WSODD', 'C', s, 'normal')][m] - val[('WSODD', 'B+Focal', s, 'normal')][m]) - (val[('WSODD', 'B+CBAM', s, 'normal')][m] - val[('WSODD', 'B', s, 'normal')][m]) for s in S6]
    did[m] = summ(x)
values['attention_by_loss_interaction'] = did

# FloW small-object weighting: share of degraded-view targets that received the weight, six training runs
share = {}
for s in S6:
    root = FLOW_NEW if s in ('42', '123', '3407') else FLOW_OLD
    t = json.load(open(root / f'FloW_E3_seed{s}/runtime_evidence.json', encoding='utf-8'))['total_target_counts']
    share[s] = 100 * t['degraded_small_targets'] / t['degraded_targets']
values['flow_weighted_target_share_percent'] = dict(per_seed=share, low=min(share.values()), high=max(share.values()))

n_best = len(weights['sealed']) + len(weights['added'])
values['record_counts'] = dict(counts, best_checkpoints_original=len(weights['sealed']), best_checkpoints_added=len(weights['added']), best_checkpoints=n_best,
                               last_checkpoints=len(weights['last.pt']), records_total=counts['sealed records'] + counts['records added for conditions and last.pt'] + counts['records of the added runs'])
TABLES.mkdir(parents=True, exist_ok=True); PLOT.mkdir(parents=True, exist_ok=True)
(TABLES / 'round2_values.json').write_text(json.dumps(values, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def pm(s, nd=3):
    return f"{s['mean']:.{nd}f} ± {s['sd']:.{nd}f}"


def spm(s, nd=3):
    return f"{s['mean']:+.{nd}f} ± {s['sd']:.{nd}f}".replace('-', '−')


def ci(s, nd=3):
    lo, hi = f"{s['ci_low']:+.{nd}f}", f"{s['ci_high']:+.{nd}f}"
    return (lo + ' to ' + hi).replace('-', '−')


def write(path, header, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


t4a = [['a', ds.replace('SeaDronesSee', 'SDS'), label, values['arms'][f'{ds}|{arm}']['mAP50-95']['n'], pm(values['arms'][f'{ds}|{arm}']['mAP50-95']), pm(values['arms'][f'{ds}|{arm}']['APsmall'])]
       for ds, arm, label in ARMS[:7]]
t4b = []
for ds, new, ref, group, label in CONTRASTS[:4]:
    c = values['contrasts'][f'{ds}|{new}|{ref}']
    t4b.append(['b', ds.replace('SeaDronesSee', 'SDS'), label, c['mAP50-95']['diff']['n'], spm(c['mAP50-95']['diff']), ci(c['mAP50-95']['diff']), spm(c['APsmall']['diff']), ci(c['APsmall']['diff'])])
write(TABLES / 'table4_strategies.csv', ['panel', 'dataset', 'configuration', 'n', 'mAP50-95 or change', 'APsmall or 95% CI', 'APsmall change', '95% CI'], t4a + t4b)
t5 = []
for ds, arm, label in ARMS[8:15]:
    a = values['arms'][f'{ds}|{arm}']; t5.append(['a', label, pm(a['mAP50-95']), pm(a['APsmall']), pm(a['mAP50']), '', ''])
for ds, new, ref, group, label in CONTRASTS[4:14]:
    c = values['contrasts'][f'{ds}|{new}|{ref}']
    t5.append(['b', label, spm(c['mAP50-95']['diff']), ci(c['mAP50-95']['diff']), f"{c['mAP50-95']['diff']['positive']}/6", spm(c['APsmall']['diff']), ci(c['APsmall']['diff']), f"{c['APsmall']['diff']['positive']}/6"])
for ds, new, ref, group, label in CONTRASTS[14:16] + CONTRASTS[17:19]:
    c = values['contrasts'][f'{ds}|{new}|{ref}']
    t5.append(['c', label, spm(c['mAP50-95']['diff']), ci(c['mAP50-95']['diff']), f"{c['mAP50-95']['diff']['positive']}/6", spm(c['APsmall']['diff']), ci(c['APsmall']['diff']), f"{c['APsmall']['diff']['positive']}/6"])
write(TABLES / 'table5_matched.csv', ['panel', 'configuration or comparison', 'mAP50-95', 'APsmall or 95% CI', 'mAP50 or positive seeds', 'APsmall change', '95% CI', 'positive seeds'], t5)
write(TABLES / 'table1_coverage.csv', ['analysis', 'dataset', 'configurations', 'seeds', 'conditions'],
      [['Detector reference', 'SDS, WSODD, FloW', 'YOLOv8s, YOLO11s, YOLO26s, RT-DETR-L', '0', 'normal'],
       ['Multi-seed strategies', 'SDS', 'standard training; fixed single view', '0, 1, 2', 'normal; low light and localized glare, levels 1-5'],
       ['Multi-seed strategies', 'WSODD', 'standard training; fixed single view', ', '.join(S6), 'normal; low light and localized glare, levels 1-5'],
       ['Multi-seed strategies', 'FloW', 'standard training; paired views; paired views + small-object weighting; identical paired views', ', '.join(S6), 'normal; low light levels 3-5; localized glare level 3'],
       ['Matched controls (from scratch)', 'WSODD', 'seven configurations', ', '.join(S6), 'normal; low light levels 3-5; localized glare level 3'],
       ['Matched controls (pretrained)', 'WSODD', 'clean copies; GlareSoft copies', ', '.join(S6), 'normal; low light levels 3-5; localized glare level 3']])
plot = dict(figure7=[], figure8=[])
for ds, new, ref, group, label in CONTRASTS:
    c = values['contrasts'][f'{ds}|{new}|{ref}']
    row = dict(dataset=ds, new=new, ref=ref, group=group, label=label, seeds=seeds(ds),
               **{m: {k: c[m]['diff'][k] for k in ('per_seed', 'mean', 'sd', 'ci_low', 'ci_high', 'positive', 'n')} for m in ('mAP50-95', 'APsmall')})
    plot['figure7' if group == 'strategy' else 'figure8'].append(row)
(PLOT / 'paired_effects.json').write_text(json.dumps(plot, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('keys', len(val), '| counts', dict(values['record_counts']))
for ds, new, ref, group, label in CONTRASTS:
    c = values['contrasts'][f'{ds}|{new}|{ref}']
    print(f"{ds:12s} {new:8s}-{ref:8s} mAP {spm(c['mAP50-95']['diff'], 2)} [{ci(c['mAP50-95']['diff'], 2)}] {c['mAP50-95']['diff']['positive']}/{c['mAP50-95']['diff']['n']} | "
          f"APs {spm(c['APsmall']['diff'], 2)} [{ci(c['APsmall']['diff'], 2)}] {c['APsmall']['diff']['positive']}/{c['APsmall']['diff']['n']}")
print('weighted share', {k: round(v, 2) for k, v in share.items()})

# FloW single-component runs under the pretrained recipe (three seeds; same official evaluator settings). Read-only archive file.
FLOW_SC = RING3 / '03_历史归档/独立补充实验/flow_source_cbam_20260920/FINAL_OFFICIAL_COCO_PER_SEED_20260921.csv'
sc = {}
for r in csv.DictReader(open(FLOW_SC, encoding='utf-8-sig')):
    if r['split'] == 'test':
        assert r['evaluator'] == 'official_pycocotools_COCOeval_2.0.10' and int(r['images']) == 541 and int(r['boxes']) == 1090
        sc[(r['arm'], r['seed'])] = {m: 100 * float(r[m]) for m in M}
anchor = max(abs(sc[(a, s)][m] - val[('FloW', a, s, 'normal')][m]) for a in ('E0', 'E2', 'E3') for s in S3 for m in M)
assert anchor < 1e-9, anchor
fsc = {}
for arm, label in (('cbam_only', 'Attention only'), ('focal_eiou_only', 'Localization loss only'), ('cbam_focal', 'Attention + localization loss')):
    fsc[arm] = dict(label=label, seeds=S3, **{m: summ([sc[(arm, s)][m] - sc[('E0', s)][m] for s in S3]) for m in M})
values['flow_single_component_vs_standard'] = dict(anchor_max_abs_difference_pp=anchor, **fsc)
(TABLES / 'round2_values.json').write_text(json.dumps(values, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
for arm, v in fsc.items():
    print('FloW', arm, '- E0: mAP', spm(v['mAP50-95'], 2), ci(v['mAP50-95'], 2), f"{v['mAP50-95']['positive']}/3", '| APs', spm(v['APsmall'], 2), ci(v['APsmall'], 2), f"{v['APsmall']['positive']}/3")
