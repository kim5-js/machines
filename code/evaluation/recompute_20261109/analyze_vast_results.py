"""Summarize the Vast evaluations together with the sealed 20260924 records. CPU only.

Inputs: OUT/vast_results/output/evaluations/*/evaluation.json (Vast), the sealed unified CSV, and the
local evaluations for the cross-environment check. Existing sealed evaluations stay the authority for
every key they cover; new conditions come from Vast.
Outputs under OUT/results_vast/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import csv, itertools, math, statistics, collections

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
EV = OUT / 'vast_results/output/evaluations'
RES = OUT / 'results_vast'
RES.mkdir(parents=True, exist_ok=True)
METRICS = ('mAP50-95', 'APsmall', 'mAP50', 'APmedium', 'APlarge')


def tcrit(n):
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def summary(values):
    n = len(values); m = sum(values) / n
    sd = statistics.stdev(values) if n > 1 else float('nan')
    hw = tcrit(n) * sd / math.sqrt(n) if n > 1 else float('nan')
    flips = [sum(s * v for s, v in zip(signs, values)) / n for signs in itertools.product((1, -1), repeat=n)]
    return dict(n=n, mean=m, sd=sd, ci_low=m - hw, ci_high=m + hw, excludes_zero=(m - hw > 0 or m + hw < 0), positive=sum(v > 0 for v in values),
                signflip_p=sum(abs(f) >= abs(m) - 1e-12 for f in flips) / len(flips))


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


# Values: sealed first, then Vast
rows = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
val = {}
ident = {}
for r in rows:
    ident[(r['family'], r['dataset'], r['arm_raw'], r['seed'])] = r['experiment_id']
    val[(r['experiment_id'], r['condition_raw'])] = {m: 100 * float(r[m + '_fraction']) for m in METRICS if r[m + '_fraction'] not in ('', None)}
vast = {}
for p in EV.glob('*/evaluation.json'):
    e = read(p)
    if e['job'].get('block') == 'gate':
        continue
    vast[e['job']['key']] = e
    key = (e['job']['experiment_id'] + ('_last' if '_last__' in e['job']['key'] else ''), e['job']['condition'])
    if key not in val:
        val[key] = {m: 100 * e['metrics'][m] for m in METRICS if e['metrics'][m] is not None}
print('vast evaluations', len(vast))

# 1. Cross-environment check against the local RTX 3060 runs
cross = []
for p in (OUT / 'evaluations').glob('*/evaluation.json'):
    loc = read(p); k = loc['job']['key']
    if k in vast:
        d = {m: 100 * (vast[k]['metrics'][m] - loc['metrics'][m]) for m in METRICS if vast[k]['metrics'][m] is not None}
        cross.append(dict(key=k, pixels_identical=vast[k]['input_pixels_sha256'] == loc['input_pixels_sha256'], max_abs_diff_pp=max(abs(v) for v in d.values()), **{f'diff_{m}': v for m, v in d.items()}))
write_csv(RES / 'cross_environment_check.csv', cross)
print('cross-env', len(cross), 'pixels identical', sum(c['pixels_identical'] for c in cross), 'max diff pp', max((c['max_abs_diff_pp'] for c in cross), default=None))


def paired(family, ds, new, ref, seeds, cond, metric, interaction=False):
    out = []
    for s in seeds:
        a, b = ident[(family, ds, new, s)], ident[(family, ds, ref, s)]
        if (a, cond) not in val or (b, cond) not in val:
            return None
        d = val[(a, cond)][metric] - val[(b, cond)][metric]
        if interaction:
            d -= val[(a, 'normal')][metric] - val[(b, 'normal')][metric]
        out.append(d)
    return out


SEEDS = {'WSODD': ['0', '1', '2', '42', '123', '3407'], 'SeaDronesSee': ['0', '1', '2'], 'FloW': ['0', '1', '2'], 'matched': ['0', '1', '42']}
LEVELS = [('low', ['normal', 'low_light_L1', 'low_light_L2', 'low_light_strong', 'low_light_L4', 'low_light_L5']),
          ('glare', ['normal', 'glare_G1', 'glare_G2', 'glare_strong', 'glare_G4', 'glare_G5'])]

# 2. Standard training vs fixed single view: every level, both datasets, all seeds
sev = []
for ds in ('WSODD', 'SeaDronesSee'):
    for kind, conds in LEVELS:
        for lvl, cond in enumerate(conds):
            for metric in ('mAP50-95', 'APsmall'):
                row = dict(dataset=ds, kind=kind, level=lvl, condition=cond, metric=metric)
                for arm in ('E0', 'E1'):
                    xs = [val[(ident[('strategy', ds, arm, s)], cond)][metric] for s in SEEDS[ds]]
                    row[f'{arm}_mean'] = sum(xs) / len(xs); row[f'{arm}_sd'] = statistics.stdev(xs)
                d = paired('strategy', ds, 'E1', 'E0', SEEDS[ds], cond, metric)
                row.update({f'diff_{k}': v for k, v in summary(d).items()})
                if cond != 'normal':
                    row.update({f'inter_{k}': v for k, v in summary(paired('strategy', ds, 'E1', 'E0', SEEDS[ds], cond, metric, True)).items()})
                sev.append(row)
write_csv(RES / 'severity_strategy.csv', sev)

# 3. SDS best.pt vs last.pt
last = []
for metric in ('mAP50-95', 'APsmall', 'mAP50'):
    for which, suffix in (('best', ''), ('last', '_last')):
        d = [val[(ident[('strategy', 'SeaDronesSee', 'E1', s)] + suffix, 'normal')][metric] - val[(ident[('strategy', 'SeaDronesSee', 'E0', s)] + suffix, 'normal')][metric] for s in SEEDS['SeaDronesSee']]
        last.append(dict(metric=metric, checkpoint=which, **summary(d)))
    for arm in ('E0', 'E1'):
        g = [val[(ident[('strategy', 'SeaDronesSee', arm, s)] + '_last', 'normal')][metric] - val[(ident[('strategy', 'SeaDronesSee', arm, s)], 'normal')][metric] for s in SEEDS['SeaDronesSee']]
        last.append(dict(metric=metric, checkpoint=f'{arm} last minus best', **summary(g)))
write_csv(RES / 'sds_last_vs_best.csv', last)

# 4. Matched block and FloW under the two existing degraded conditions
cond_rows = []
contrasts = [('matched', 'WSODD', 'B_weights_only', 'A_baseline'), ('matched', 'WSODD', 'C_CBAM_Focal_EIoU', 'B_weights_only'), ('matched', 'WSODD', 'D_clean_duplicate', 'C_CBAM_Focal_EIoU'),
             ('matched', 'WSODD', 'E_refsoft', 'D_clean_duplicate'), ('matched', 'WSODD', 'E_refsoft', 'C_CBAM_Focal_EIoU'), ('matched', 'WSODD', 'E_refsoft', 'A_baseline'),
             ('strategy', 'FloW', 'E2', 'E0'), ('strategy', 'FloW', 'E3', 'E0'), ('strategy', 'FloW', 'E3', 'E2')]
for fam, ds, new, ref in contrasts:
    seeds = SEEDS['matched'] if fam == 'matched' else SEEDS['FloW']
    for cond in ('normal', 'low_light_strong', 'low_light_L4', 'low_light_L5', 'glare_strong'):
        for metric in ('mAP50-95', 'APsmall'):
            d = paired(fam, ds, new, ref, seeds, cond, metric)
            if d is None:
                continue
            row = dict(family=fam, dataset=ds, contrast=f'{new} - {ref}', condition=cond, metric=metric, **{f'diff_{k}': v for k, v in summary(d).items()})
            if cond != 'normal':
                row.update({f'inter_{k}': v for k, v in summary(paired(fam, ds, new, ref, seeds, cond, metric, True)).items()})
            cond_rows.append(row)
for fam, ds, arms in (('matched', 'WSODD', ['A_baseline', 'B_weights_only', 'C_CBAM_Focal_EIoU', 'D_clean_duplicate', 'E_refsoft']), ('strategy', 'FloW', ['E0', 'E2', 'E3'])):
    seeds = SEEDS['matched'] if fam == 'matched' else SEEDS['FloW']
    for arm in arms:
        for cond in ('low_light_strong', 'low_light_L4', 'low_light_L5', 'glare_strong'):
            if any((ident[(fam, ds, arm, s)], cond) not in val for s in seeds):
                continue
            for metric in ('mAP50-95', 'APsmall'):
                drop = [val[(ident[(fam, ds, arm, s)], cond)][metric] - val[(ident[(fam, ds, arm, s)], 'normal')][metric] for s in seeds]
                cond_rows.append(dict(family=fam, dataset=ds, contrast=f'{arm}: condition - normal', condition=cond, metric=metric, **{f'diff_{k}': v for k, v in summary(drop).items()}))
write_csv(RES / 'conditions_matched_flow.csv', cond_rows)
write(RES / 'summary.json', dict(vast_evaluations=len(vast), cross_environment=dict(n=len(cross), pixels_identical=sum(c['pixels_identical'] for c in cross), max_abs_diff_pp=max((c['max_abs_diff_pp'] for c in cross), default=None)),
                                 files=['cross_environment_check.csv', 'severity_strategy.csv', 'sds_last_vs_best.csv', 'conditions_matched_flow.csv']))
print('analysis written to', RES)
