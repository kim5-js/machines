"""Collect the numbers used by the 20261109 revision units. CPU only; reads results, writes summaries.

Inputs (read-only)
  sealed unified evaluation CSV (20260924) and the Vast inference evaluations of recompute_20261109
Outputs
  <figures>/绘图数据/severity_strategy.json   per-seed values and summaries for every severity level
  <source>/tables/*.csv                        table drafts for the manuscript and supplement

Run:
  python build_revision_data.py
Paths are fixed to version 20261109 below; nothing is written outside that version's folders.
"""
from pathlib import Path
import csv, itertools, json, math, statistics, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3')
VERSION = '20261109'
PAPER = RING3 / '01_当前论文'
RECOMPUTE = RING3 / '02_当前实验/原始证据/recompute_20261109'
UNIFIED_CSV = PAPER / '基础依赖/20260924_统一评价修订/unified_evaluations.csv'
EV = RECOMPUTE / 'vast_results/output/evaluations'
FIG_DATA = PAPER / '图表' / VERSION / '绘图数据'
TABLES = PAPER / '正文源' / VERSION / 'tables'
METRICS = ('mAP50-95', 'APsmall', 'APmedium', 'APlarge', 'mAP50')
SEEDS = {'WSODD': ['0', '1', '2', '42', '123', '3407'], 'SeaDronesSee': ['0', '1', '2']}
LOW = ['normal', 'low_light_L1', 'low_light_L2', 'low_light_strong', 'low_light_L4', 'low_light_L5']
GLARE = ['normal', 'glare_G1', 'glare_G2', 'glare_strong', 'glare_G4', 'glare_G5']
# Level parameters: levels 1-3 are the training-band midpoints, 4-5 lie beyond the training range.
LEVEL_PARAMS = {
    'low': [None, (0.925, 1.085), (0.88, 1.165), (0.84, 1.255), (0.70, 1.50), (0.50, 2.00)],
    'glare': [None, (0.02275, 26.0), (0.0335, 34.5), (0.0445, 43.5), (0.08, 80.0), (0.15, 120.0)],
}


def tcrit(n):
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def summary(values):
    from scipy import stats
    n = len(values); m = sum(values) / n
    sd = statistics.stdev(values)
    hw = tcrit(n) * sd / math.sqrt(n)
    flips = [sum(s * v for s, v in zip(signs, values)) / n for signs in itertools.product((1, -1), repeat=n)]
    return dict(n=n, mean=m, sd=sd, ci_low=m - hw, ci_high=m + hw, excludes_zero=bool(m - hw > 0 or m + hw < 0),
                positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values),
                t_p=float(stats.ttest_1samp(values, 0).pvalue) if sd > 0 else None,
                signflip_p=sum(abs(f) >= abs(m) - 1e-12 for f in flips) / len(flips), values=values)


def load_values():
    rows = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
    val, ident = {}, {}
    for r in rows:
        ident[(r['family'], r['dataset'], r['arm_raw'], r['seed'])] = r['experiment_id']
        val[(r['experiment_id'], r['condition_raw'])] = {m: 100 * float(r[m + '_fraction']) for m in METRICS if r.get(m + '_fraction') not in ('', None)}
    n_vast = 0
    for p in EV.glob('*/evaluation.json'):
        e = json.loads(p.read_text(encoding='utf-8'))
        if e['job'].get('block') == 'gate':
            continue
        n_vast += 1
        key = (e['job']['experiment_id'] + ('_last' if '_last__' in e['job']['key'] else ''), e['job']['condition'])
        if key not in val:  # sealed records stay the authority for the keys they cover
            val[key] = {m: 100 * e['metrics'][m] for m in METRICS if e['metrics'][m] is not None}
    return rows, val, ident, n_vast


def main():
    rows, val, ident, n_vast = load_values()
    FIG_DATA.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    out = dict(version=VERSION, vast_evaluations=n_vast, seeds=SEEDS, level_parameters=LEVEL_PARAMS, records=[])
    records_used = set()
    for ds in ('WSODD', 'SeaDronesSee'):
        for kind, conds in (('low', LOW), ('glare', GLARE)):
            for lvl, cond in enumerate(conds):
                for metric in METRICS:
                    arm_vals = {}
                    for arm in ('E0', 'E1'):
                        arm_vals[arm] = [val[(ident[('strategy', ds, arm, s)], cond)][metric] for s in SEEDS[ds]]
                        records_used.update((ident[('strategy', ds, arm, s)], cond) for s in SEEDS[ds])
                    diff = [a - b for a, b in zip(arm_vals['E1'], arm_vals['E0'])]
                    rec = dict(dataset=ds, kind=kind, level=lvl, condition=cond, metric=metric,
                               standard=dict(mean=statistics.mean(arm_vals['E0']), sd=statistics.stdev(arm_vals['E0']), values=arm_vals['E0']),
                               augmented=dict(mean=statistics.mean(arm_vals['E1']), sd=statistics.stdev(arm_vals['E1']), values=arm_vals['E1']),
                               diff=summary(diff))
                    if lvl:
                        base = [val[(ident[('strategy', ds, 'E1', s)], 'normal')][metric] - val[(ident[('strategy', ds, 'E0', s)], 'normal')][metric] for s in SEEDS[ds]]
                        rec['interaction'] = summary([d - b for d, b in zip(diff, base)])
                        norm0 = [val[(ident[('strategy', ds, 'E0', s)], 'normal')][metric] for s in SEEDS[ds]]
                        rec['standard_change_from_normal'] = summary([a - b for a, b in zip(arm_vals['E0'], norm0)])
                        norm1 = [val[(ident[('strategy', ds, 'E1', s)], 'normal')][metric] for s in SEEDS[ds]]
                        rec['augmented_change_from_normal'] = summary([a - b for a, b in zip(arm_vals['E1'], norm1)])
                    out['records'].append(rec)
    out['condition_records'] = len(records_used)
    (FIG_DATA / 'severity_strategy.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # Table drafts
    def f(x, n=2, sign=False):
        return ('{:+.%df}' % n).format(x) if sign else ('{:.%df}' % n).format(x)

    table = []
    for rec in out['records']:
        if rec['metric'] not in ('mAP50-95', 'APsmall') or (rec['kind'] == 'glare' and rec['level'] == 0):
            continue
        d = rec['diff']; i = rec.get('interaction')
        table.append(dict(dataset='SDS' if rec['dataset'] == 'SeaDronesSee' else rec['dataset'], metric=rec['metric'],
                          condition='Normal' if rec['level'] == 0 else ('Low light' if rec['kind'] == 'low' else 'Localized glare') + f" L{rec['level']}",
                          n=d['n'], standard=f"{f(rec['standard']['mean'])} ± {f(rec['standard']['sd'])}", augmented=f"{f(rec['augmented']['mean'])} ± {f(rec['augmented']['sd'])}",
                          paired_difference=f"{f(d['mean'], 2, True)} ± {f(d['sd'])}", difference_ci=f"{f(d['ci_low'], 2, True)} to {f(d['ci_high'], 2, True)}", positive_seeds=f"{d['positive']}/{d['n']}",
                          interaction='—' if not i else f"{f(i['mean'], 2, True)} ± {f(i['sd'])}", interaction_ci='—' if not i else f"{f(i['ci_low'], 2, True)} to {f(i['ci_high'], 2, True)}",
                          interaction_positive='—' if not i else f"{i['positive']}/{i['n']}"))
    with (TABLES / 'table6_condition_severity.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    ladder = [dict(level=l, low_light_brightness=LEVEL_PARAMS['low'][l][0], low_light_gamma=LEVEL_PARAMS['low'][l][1],
                   glare_target_area_pct=100 * LEVEL_PARAMS['glare'][l][0], glare_center_intensity=LEVEL_PARAMS['glare'][l][1],
                   basis=('Midpoint of training band %d' % l) if l <= 3 else 'Beyond the training range') for l in range(1, 6)]
    with (TABLES / 'tableA5_severity_levels.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(ladder[0])); w.writeheader(); w.writerows(ladder)

    # WSODD real-condition subsets, five-row summary for the response letter. Kept out of the main text; no separate
    # supplementary table: the per-seed records go into Tables S1 and S2 when the supplement is regenerated.
    res = RECOMPUTE / 'results'
    counts = json.loads((res / 'real_subsets.json').read_text(encoding='utf-8'))['counts']
    ap = [r for r in csv.DictReader(open(res / 'real_subset_ap.csv', encoding='utf-8-sig')) if r['family'] == 'strategy' and r['dataset'] == 'WSODD']
    paired = {r['subset']: r for r in csv.DictReader(open(res / 'paired_by_real_subset.csv', encoding='utf-8-sig'))
              if r['family'] == 'strategy' and r['contrast'] == 'E1 - E0' and r['metric'] == 'all'}
    definitions = {'low_luminance_lt110': ('Low luminance', 'Mean gray level below 110'),
                   'saturated_gt0p3pct': ('High saturation', 'Pixels with gray level of at least 250 exceed 0.3% of the image'),
                   'captured_18h': ('Captured 18:00-18:59', 'File-name timestamp'),
                   'captured_19h': ('Captured 19:00-19:59', 'File-name timestamp'),
                   'rest_of_test': ('Remaining test images', 'In none of the four subsets')}
    subset_rows = []
    for key, (label, rule) in definitions.items():
        arm = {a: {r['seed']: float(r['all']) for r in ap if r['arm'] == a and r['subset'] == key} for a in ('E0', 'E1')}
        diffs = [arm['E1'][s] - arm['E0'][s] for s in SEEDS['WSODD']]
        s6 = summary(diffs); p = paired[key]
        assert abs(s6['mean'] - float(p['mean'])) < 1e-9 and abs(s6['ci_low'] - float(p['ci_low'])) < 1e-9, key  # same numbers as the recompute tables
        row = dict(subset=label, definition=rule, images=counts[key]['images'], objects=counts[key]['boxes'], small_objects=counts[key]['small_boxes'], seeds=s6['n'],
                   standard_mAP50_95=f"{f(statistics.mean(arm['E0'].values()))} ± {f(statistics.stdev(arm['E0'].values()))}",
                   augmented_mAP50_95=f"{f(statistics.mean(arm['E1'].values()))} ± {f(statistics.stdev(arm['E1'].values()))}",
                   paired_difference_pp=f"{f(s6['mean'], 2, True)} ± {f(s6['sd'])}", difference_ci=f"{f(s6['ci_low'], 2, True)} to {f(s6['ci_high'], 2, True)}",
                   positive_seeds=f"{s6['positive']}/{s6['n']}")
        row.update({f'difference_seed_{s}_pp': f(d, 2, True) for s, d in zip(SEEDS['WSODD'], diffs)})
        subset_rows.append(row)
        print('real subset', label, row['images'], row['objects'], row['small_objects'], row['paired_difference_pp'], row['difference_ci'], row['positive_seeds'])
    with (TABLES / 'real_condition_subsets_summary.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(subset_rows[0])); w.writeheader(); w.writerows(subset_rows)

    # Console digest for drafting
    def get(ds, kind, lvl, metric):
        return next(r for r in out['records'] if (r['dataset'], r['kind'], r['level'], r['metric']) == (ds, kind, lvl, metric))
    print('condition records used', len(records_used), '| vast evaluations', n_vast)
    for ds in ('WSODD', 'SeaDronesSee'):
        for kind in ('low', 'glare'):
            for metric in ('mAP50-95', 'APsmall', 'APmedium', 'APlarge'):
                line = []
                for lvl in range(6):
                    r = get(ds, kind, lvl, metric)
                    line.append(f"L{lvl}: std {r['standard']['mean']:.2f} aug {r['augmented']['mean']:.2f} d {r['diff']['mean']:+.2f}[{r['diff']['ci_low']:+.2f},{r['diff']['ci_high']:+.2f}] {r['diff']['positive']}/{r['diff']['n']}"
                                + (f" stdΔ {r['standard_change_from_normal']['mean']:+.2f} augΔ {r['augmented_change_from_normal']['mean']:+.2f}" if lvl else ''))
                print(ds, kind, metric, '\n   ' + '\n   '.join(line))
    # Reference dispersion of standard training (normal inputs) for Section 4.1 wording
    for ds, seeds in (('SeaDronesSee', ['0', '1', '2']), ('WSODD', SEEDS['WSODD']), ('FloW', ['0', '1', '2'])):
        xs = [val[(ident[('strategy', ds, 'E0', s)], 'normal')]['mAP50-95'] for s in seeds]
        print('standard-training seed SD', ds, 'n', len(xs), f'mean {statistics.mean(xs):.3f} sd {statistics.stdev(xs):.3f}')
    det = [r for r in rows if r['family'] == 'detector' and r['condition_raw'] == 'normal']
    for r in sorted(det, key=lambda r: (r['dataset'], -float(r['mAP50-95_fraction']))):
        print('detector', r['dataset'], r['arm_raw'], f"{100 * float(r['mAP50-95_fraction']):.2f}", f"small {100 * float(r['APsmall_fraction']):.2f}")


if __name__ == '__main__':
    main()
