"""Combine the train_20261109 evaluations with the existing seeds and compute the paired contrasts. CPU only.

Inputs (read only):
  sealed unified CSV (20260924)                          -> original seeds, normal (+ WSODD E0/E1 strong conditions)
  recompute_20261109/vast_results evaluations            -> original seeds, further conditions (same evaluator)
  train_20261109/results/extracted/output/evaluations    -> new runs (same evaluator, checked below)
Outputs: <out_dir> (default 02_当前实验/原始证据/train_20261109/analysis): contrasts.csv, arm_means.csv, summary.json.
Sealed values keep priority for every key they cover; recompute values next; new runs last.
Usage: python analyze_train_results.py [out_dir]
"""
import sys, csv, json, math, statistics, itertools, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'recompute_20261109'))
from common import RING3, OUT as RECOMPUTE_OUT, UNIFIED_CSV, read  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TRAIN = RING3 / '02_当前实验/原始证据/train_20261109'
NEW_EV = TRAIN / 'results/extracted/output/evaluations'
OLD_EV = RECOMPUTE_OUT / 'vast_results/output/evaluations'
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN / 'analysis'
RES.mkdir(parents=True, exist_ok=True)
METRICS = ('mAP50-95', 'APsmall', 'mAP50')
CONDS = ('normal', 'low_light_strong', 'low_light_L4', 'low_light_L5', 'glare_strong')
EVALUATOR = '988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267'
SEEDS6 = ['0', '1', '2', '42', '123', '3407']
MATCHED_NEW = {'baseline': 'A_baseline', 'weights_only': 'B_weights_only', 'bundle': 'C_CBAM_Focal_EIoU', 'duplicate': 'D_clean_duplicate',
               'refsoft': 'E_refsoft', 'cbam_only': 'B+CBAM', 'focal_only': 'B+FocalEIoU'}


def tcrit(n):
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def summary(values):
    n = len(values); m = sum(values) / n
    sd = statistics.stdev(values) if n > 1 else float('nan')
    hw = tcrit(n) * sd / math.sqrt(n) if n > 1 else float('nan')
    flips = [sum(s * v for s, v in zip(signs, values)) / n for signs in itertools.product((1, -1), repeat=n)]
    return dict(n=n, mean=m, sd=sd, ci_low=m - hw, ci_high=m + hw, excludes_zero=bool(m - hw > 0 or m + hw < 0),
                positive=sum(v > 0 for v in values), signflip_p=sum(abs(f) >= abs(m) - 1e-12 for f in flips) / len(flips))


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


# (dataset, arm, seed, condition) -> {metric: percentage points}; source recorded for each key
val, src = {}, {}


def put(key, metrics, source):
    if key not in val:
        val[key] = metrics; src[key] = source


for r in csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')):
    if r['family'] not in ('matched', 'strategy'):
        continue
    put((r['dataset'], r['arm_raw'], r['seed'], r['condition_raw']),
        {m: 100 * float(r[m + '_fraction']) for m in METRICS if r.get(m + '_fraction') not in ('', None)}, 'sealed')

old_ident = {}
for r in csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')):
    old_ident[r['experiment_id']] = (r['dataset'], r['arm_raw'], r['seed'])
evaluators = collections.Counter()
for p in OLD_EV.glob('*/evaluation.json'):
    e = read(p); j = e['job']
    if j.get('block') == 'gate' or '_last__' in j['key'] or j['experiment_id'] not in old_ident:
        continue
    evaluators[('recompute', e['evaluator_code_sha256'])] += 1
    ds, arm, seed = old_ident[j['experiment_id']]
    put((ds, arm, seed, j['condition']), {m: 100 * e['metrics'][m] for m in METRICS}, 'recompute_20261109')

new_count = 0
for p in NEW_EV.glob('*/evaluation.json'):
    e = read(p); j = e['job']
    evaluators[('train', e['evaluator_code_sha256'])] += 1
    base, seed = j['experiment_id'].rsplit('_seed', 1)
    ds, arm = base.split('_', 1)
    if ds == 'WSODD' and arm in MATCHED_NEW:
        arm = MATCHED_NEW[arm]
    key = (ds, arm, seed, j['condition'])
    if key in val:
        raise RuntimeError(f'New run collides with an existing key: {key}')
    put(key, {m: 100 * e['metrics'][m] for m in METRICS}, 'train_20261109'); new_count += 1
if set(h for _, h in evaluators) != {EVALUATOR}:
    raise RuntimeError(f'Evaluator differs: {evaluators}')


def seeds_with(ds, arms, conds):
    return [s for s in SEEDS6 + ['0', '1', '2'] if all((ds, a, s, c) in val for a in arms for c in conds)]


def ordered(seeds):
    return sorted(set(seeds), key=lambda s: SEEDS6.index(s) if s in SEEDS6 else 99)


rows, means = [], []
missing = []


def contrast(block, ds, new, ref, label, cond, metric, seeds=None):
    need = ['normal', cond]
    avail = ordered(seeds_with(ds, [new, ref], need))
    use = [s for s in (seeds or SEEDS6) if s in avail]
    if len(use) < 2:
        missing.append(dict(block=block, contrast=label, condition=cond, available=avail)); return
    d = [val[(ds, new, s, cond)][metric] - val[(ds, ref, s, cond)][metric] for s in use]
    row = dict(block=block, dataset=ds, contrast=label, condition=cond, metric=metric, seeds=' '.join(use),
               **{f'diff_{k}': v for k, v in summary(d).items()})
    if cond != 'normal':
        inter = [x - (val[(ds, new, s, 'normal')][metric] - val[(ds, ref, s, 'normal')][metric]) for x, s in zip(d, use)]
        row.update({f'inter_{k}': v for k, v in summary(inter).items()})
    rows.append(row)


def factorial(block, ds, a, b, c, d, label, cond, metric):
    """(d - c) - (b - a): difference-in-differences on matched seeds."""
    use = ordered(seeds_with(ds, [a, b, c, d], [cond]))
    if len(use) < 2:
        missing.append(dict(block=block, contrast=label, condition=cond, available=use)); return
    x = [(val[(ds, d, s, cond)][metric] - val[(ds, c, s, cond)][metric]) - (val[(ds, b, s, cond)][metric] - val[(ds, a, s, cond)][metric]) for s in use]
    rows.append(dict(block=block, dataset=ds, contrast=label, condition=cond, metric=metric, seeds=' '.join(use), **{f'diff_{k}': v for k, v in summary(x).items()}))


M = 'matched'
plan = [
    # 1. Matched controls, now six seeds
    (M, 'WSODD', 'B_weights_only', 'A_baseline'), (M, 'WSODD', 'C_CBAM_Focal_EIoU', 'B_weights_only'), (M, 'WSODD', 'D_clean_duplicate', 'C_CBAM_Focal_EIoU'),
    (M, 'WSODD', 'E_refsoft', 'D_clean_duplicate'), (M, 'WSODD', 'E_refsoft', 'C_CBAM_Focal_EIoU'), (M, 'WSODD', 'E_refsoft', 'A_baseline'),
    # 2. Component split of C (2x2: CBAM x loss, both on the B weights)
    ('component', 'WSODD', 'B+CBAM', 'B_weights_only'), ('component', 'WSODD', 'B+FocalEIoU', 'B_weights_only'),
    ('component', 'WSODD', 'C_CBAM_Focal_EIoU', 'B+CBAM'), ('component', 'WSODD', 'C_CBAM_Focal_EIoU', 'B+FocalEIoU'),
    # 3. FloW, now six seeds, plus the identical-view control
    ('flow', 'FloW', 'E2', 'E0'), ('flow', 'FloW', 'E3', 'E0'), ('flow', 'FloW', 'E3', 'E2'),
    ('flow', 'FloW', 'E2', 'E2C'), ('flow', 'FloW', 'E2C', 'E0'),
    # 4. Pretrained fine-tuning with appended copies (E0 = pretrained standard training without copies)
    ('pretrained', 'WSODD', 'pretrained_glaresoft', 'pretrained_clean'), ('pretrained', 'WSODD', 'pretrained_clean', 'E0'),
    ('pretrained', 'WSODD', 'pretrained_glaresoft', 'E0'),
]
for block, ds, new, ref in plan:
    for cond in CONDS:
        for metric in METRICS:
            contrast(block, ds, new, ref, f'{new} - {ref}', cond, metric)
            if block in (M, 'flow') and (new, ref) in (('C_CBAM_Focal_EIoU', 'B_weights_only'), ('E_refsoft', 'D_clean_duplicate'), ('E2', 'E0'), ('E3', 'E0')):
                orig = ['0', '1', '42'] if block == M else ['0', '1', '2']
                contrast(block + '_original_seeds', ds, new, ref, f'{new} - {ref}', cond, metric, seeds=orig)
for cond in CONDS:
    for metric in METRICS:
        factorial('component', 'WSODD', 'B_weights_only', 'B+CBAM', 'B+FocalEIoU', 'C_CBAM_Focal_EIoU', 'CBAM x loss interaction: (C - Focal) - (CBAM - B)', cond, metric)

arms = collections.defaultdict(set)
for (ds, arm, seed, cond) in val:
    arms[(ds, arm, cond)].add(seed)
for (ds, arm, cond), ss in sorted(arms.items()):
    if ds not in ('WSODD', 'FloW') or cond not in CONDS:
        continue
    for metric in METRICS:
        xs = [val[(ds, arm, s, cond)][metric] for s in ordered(ss)]
        means.append(dict(dataset=ds, arm=arm, condition=cond, metric=metric, n=len(xs), seeds=' '.join(ordered(ss)), mean=sum(xs) / len(xs),
                          sd=statistics.stdev(xs) if len(xs) > 1 else float('nan'),
                          sources=' '.join(sorted({src[(ds, arm, s, cond)] for s in ss}))))
write_csv(RES / 'contrasts.csv', rows)
write_csv(RES / 'arm_means.csv', means)
json.dump(dict(new_evaluations=new_count, evaluator=EVALUATOR, evaluator_counts={f'{k[0]}:{k[1][:12]}': v for k, v in evaluators.items()},
               missing=missing, files=['contrasts.csv', 'arm_means.csv']), open(RES / 'summary.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('new evaluations', new_count, 'contrast rows', len(rows), 'missing', len(missing))
for m in missing[:20]:
    print('missing', m)
print('written to', RES)
