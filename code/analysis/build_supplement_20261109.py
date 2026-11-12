"""Supplementary data of revision 20261109 (Tables S1-S5). CPU only; nothing is trained, inferred or re-scored here.

Inputs (read only)
  sealed unified evaluation CSV; evaluation.json of recompute_20261109 and of the added runs
  FloW single-component official evaluation file (archive)
  offline analysis tables in 02_当前实验/原始证据/recompute_20261109/results (size bins, classes, real-condition subsets)
  补充材料/20261001_r03/Supplementary_Data (previous S1 for the record numbers R001-R078; S4 is copied unchanged)
Outputs: 补充材料/20261109/Supplementary_Data/S1 ... S5 and supplement_audit.json one level above.
Record numbers are neutral (R001 ...); no dates, paths, hashes or batch labels are written into the tables.
Run:  python build_supplement_20261109.py      (scientific environment; needs scipy)
"""
import collections, csv, glob, json, math, shutil, statistics, sys
from pathlib import Path
from scipy import stats as st

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3'); PAPER = RING3 / '01_当前论文'; EVID = RING3 / '02_当前实验/原始证据'
OLD = PAPER / '补充材料/20261001_r03/Supplementary_Data'
OUT = PAPER / '补充材料/20261109/Supplementary_Data'
RES = EVID / 'recompute_20261109/results'
FULL = ('mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge')
M5 = ('mAP50-95', 'APsmall', 'APmedium', 'APlarge', 'mAP50')
S6 = ['0', '1', '2', '42', '123', '3407']; S3 = ['0', '1', '2']
CONFIG = {'yolov8s': 'YOLOv8s', 'yolo11s': 'YOLO11s', 'yolo26s': 'YOLO26s', 'rtdetr_l': 'RT-DETR-L', 'E0': 'Standard training', 'E1': 'Fixed single view',
          'E2': 'Paired views', 'E3': 'Paired views + small-object weighting', 'E2C': 'Identical paired views',
          'A_baseline': 'Scratch reference', 'baseline': 'Scratch reference', 'B_weights_only': 'Adjusted loss weights', 'weights_only': 'Adjusted loss weights',
          'cbam_only': 'Attention only', 'focal_only': 'Localization loss only', 'C_CBAM_Focal_EIoU': 'Attention + localization loss', 'bundle': 'Attention + localization loss',
          'D_clean_duplicate': 'Clean copies', 'duplicate': 'Clean copies', 'E_refsoft': 'GlareSoft copies', 'refsoft': 'GlareSoft copies',
          'pretrained_clean': 'Pretrained + clean copies', 'pretrained_glaresoft': 'Pretrained + GlareSoft copies'}
FLOW_SC = {'cbam_only': 'Pretrained + attention only', 'focal_eiou_only': 'Pretrained + localization loss only', 'cbam_focal': 'Pretrained + attention + localization loss',
           'cbam_focal_e2': 'Pretrained + attention + localization loss + paired views', 'cbam_focal_e3': 'Pretrained + attention + localization loss + weighted paired views'}
COND = {'normal': 'Normal', 'low_light_L1': 'Low light, level 1', 'low_light_L2': 'Low light, level 2', 'low_light_strong': 'Low light, level 3', 'low_light_L4': 'Low light, level 4',
        'low_light_L5': 'Low light, level 5', 'glare_G1': 'Localized glare, level 1', 'glare_G2': 'Localized glare, level 2', 'glare_strong': 'Localized glare, level 3',
        'glare_G4': 'Localized glare, level 4', 'glare_G5': 'Localized glare, level 5'}
CORDER = list(COND)
EVAL = 'pycocotools 2.0.10 COCOeval'

# ---------------------------------------------------------------- records
rec = {}   # (dataset, configuration, seed, checkpoint, condition) -> row


def add(key, split, images, boxes, metrics, group):
    assert key not in rec, key
    rec[key] = dict(dataset=key[0], split=split, configuration=key[1], seed=key[2], checkpoint=key[3], condition=COND[key[4]], images=images, boxes=boxes, evaluator=EVAL,
                    imgsz=960, actual_batch=1, half=False, rect=False, **{m + '_fraction': metrics.get(m, '') for m in FULL},
                    **{m + '_percent': 100 * metrics[m] for m in ('mAP50-95', 'APsmall', 'mAP50')}, _group=group, _cond=key[4])


for r in csv.DictReader(open(PAPER / '基础依赖/20260924_统一评价修订/unified_evaluations.csv', encoding='utf-8-sig')):
    add((r['dataset'], CONFIG[r['arm_raw']], r['seed'], 'best.pt', r['condition_raw']), r['split'], int(r['images']), int(r['boxes']), {m: float(r[m + '_fraction']) for m in FULL}, 0)
for p in sorted(glob.glob(str(EVID / 'recompute_20261109/vast_results/output/evaluations/*/evaluation.json'))):
    e = json.load(open(p, encoding='utf-8')); j = e['job']
    if j.get('block') == 'gate':
        continue
    last = '_last__' in j['key']
    key = (j['dataset'], CONFIG[j['arm']], str(j['seed']), 'last.pt' if last else 'best.pt', j['condition'])
    add(key, j['split'], e['images'], e['ground_truth_boxes'], e['metrics'], 1)
for p in sorted(glob.glob(str(EVID / 'train_20261109/results/extracted/output/evaluations/*/evaluation.json'))):
    e = json.load(open(p, encoding='utf-8')); j = e['job']; base, seed = j['experiment_id'].rsplit('_seed', 1); ds, arm = base.split('_', 1)
    add((ds, CONFIG[arm], seed, 'best.pt', j['condition']), j['split'], e['images'], e['ground_truth_boxes'], e['metrics'], 1)
for r in csv.DictReader(open(RING3 / '03_历史归档/独立补充实验/flow_source_cbam_20260920/FINAL_OFFICIAL_COCO_PER_SEED_20260921.csv', encoding='utf-8-sig')):
    if r['split'] == 'test' and r['arm'] in FLOW_SC:
        add(('FloW', FLOW_SC[r['arm']], r['seed'], 'best.pt', 'normal'), 'test', int(r['images']), int(r['boxes']), {m: float(r[m]) for m in ('mAP50-95', 'mAP50', 'APsmall', 'APmedium', 'APlarge')}, 2)
# record numbers: the 78 earlier records keep R001-R078; the rest follow in a fixed order
old_ids = {}
for r in csv.DictReader(open(OLD / 'S1/Table_S1_all_evaluations.csv', encoding='utf-8-sig')):
    old_ids[(r['dataset'], r['seed'], r['mAP50-95_fraction'], r['images'])] = r['record_id']
order_ds = {'SeaDronesSee': 0, 'WSODD': 1, 'FloW': 2}
config_order = list(dict.fromkeys(list(CONFIG.values()) + list(FLOW_SC.values())))
numbered = {}
for key, row in rec.items():
    rid = old_ids.get((row['dataset'], row['seed'], repr(row['mAP50-95_fraction']), str(row['images']))) if row['_group'] == 0 else None
    if rid:
        numbered[key] = rid
assert len(numbered) == 78 and len(set(numbered.values())) == 78, len(numbered)
rest = sorted((k for k in rec if k not in numbered), key=lambda k: (rec[k]['_group'], order_ds[k[0]], config_order.index(k[1]), k[3], S6.index(k[2]) if k[2] in S6 else 9, CORDER.index(k[4])))
for i, k in enumerate(rest, 79):
    numbered[k] = f'R{i:03d}'
rows1 = []
for key in sorted(rec, key=lambda k: numbered[k]):
    row = {k: v for k, v in rec[key].items() if not k.startswith('_')}
    rows1.append(dict(record_id=numbered[key], **row))


def write(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = header or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


write(OUT / 'S1/Table_S1_all_evaluations.csv', rows1)


def summ(v):
    n = len(v); mean = sum(v) / n; sd = statistics.stdev(v); hw = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, sd, mean - hw, mean + hw, sum(x > 0 for x in v)


def pvals(v):
    """Two-sided p values of the paired differences: paired t test, and exact sign-flip test over all 2^n sign assignments."""
    import itertools
    t = st.ttest_1samp(v, 0.0).pvalue if statistics.stdev(v) > 0 else float('nan')
    observed = abs(sum(v)); total = hits = 0
    for signs in itertools.product((1, -1), repeat=len(v)):
        total += 1; hits += abs(sum(a * b for a, b in zip(signs, v))) >= observed - 1e-12
    return float(t), hits / total


def pct(key, m):
    return 100 * rec[key][m + '_fraction']


# ---------------------------------------------------------------- S2 paired differences, S3 condition interactions
PLAN = [('stability_contrasts', 'SeaDronesSee', 'Fixed single view', 'Standard training', S3), ('stability_contrasts', 'WSODD', 'Fixed single view', 'Standard training', S6),
        ('stability_contrasts', 'FloW', 'Paired views', 'Standard training', S6), ('stability_contrasts', 'FloW', 'Paired views + small-object weighting', 'Standard training', S6),
        ('matched_contrasts', 'WSODD', 'Adjusted loss weights', 'Scratch reference', S6), ('matched_contrasts', 'WSODD', 'Attention + localization loss', 'Adjusted loss weights', S6),
        ('matched_contrasts', 'WSODD', 'Clean copies', 'Attention + localization loss', S6), ('matched_contrasts', 'WSODD', 'GlareSoft copies', 'Clean copies', S6),
        ('matched_contrasts', 'WSODD', 'GlareSoft copies', 'Attention + localization loss', S6), ('matched_contrasts', 'WSODD', 'GlareSoft copies', 'Scratch reference', S6),
        ('component_contrasts', 'WSODD', 'Attention only', 'Adjusted loss weights', S6), ('component_contrasts', 'WSODD', 'Localization loss only', 'Adjusted loss weights', S6),
        ('component_contrasts', 'WSODD', 'Attention + localization loss', 'Attention only', S6), ('component_contrasts', 'WSODD', 'Attention + localization loss', 'Localization loss only', S6),
        ('pretrained_copy_contrasts', 'WSODD', 'Pretrained + GlareSoft copies', 'Pretrained + clean copies', S6), ('pretrained_copy_contrasts', 'WSODD', 'Pretrained + clean copies', 'Standard training', S6),
        ('pretrained_copy_contrasts', 'WSODD', 'Pretrained + GlareSoft copies', 'Standard training', S6),
        ('paired_view_contrasts', 'FloW', 'Paired views', 'Identical paired views', S6), ('paired_view_contrasts', 'FloW', 'Identical paired views', 'Standard training', S6),
        ('paired_view_contrasts', 'FloW', 'Paired views + small-object weighting', 'Paired views', S6),
        ('flow_component_contrasts', 'FloW', 'Pretrained + attention only', 'Standard training', S3), ('flow_component_contrasts', 'FloW', 'Pretrained + localization loss only', 'Standard training', S3),
        ('flow_component_contrasts', 'FloW', 'Pretrained + attention + localization loss', 'Standard training', S3)]
rows2, rows3 = [], []
for analysis, ds, new, ref, seeds in PLAN:
    for ckpt in ('best.pt', 'last.pt'):
        for c in CORDER:
            if not all((ds, a, s, ckpt, c) in rec for a in (new, ref) for s in seeds):
                continue
            for m in M5:
                d = [pct((ds, new, s, ckpt, c), m) - pct((ds, ref, s, ckpt, c), m) for s in seeds]
                mean, sd, lo, hi, pos = summ(d); pt, ps = pvals(d)
                for s, x in zip(seeds, d):
                    rows2.append(dict(analysis=analysis if c == 'normal' and ckpt == 'best.pt' else ('checkpoint_sensitivity' if ckpt == 'last.pt' else 'condition_contrasts'), dataset=ds, seed=s,
                                      new_configuration=new, reference_configuration=ref, checkpoint=ckpt, condition=COND[c], metric=m, paired_change_pp=x, group_mean_pp=mean,
                                      group_sample_sd_pp=sd, ci95_low_pp=lo, ci95_high_pp=hi, positive_seeds=pos, n=len(seeds), p_paired_t=pt, p_sign_flip=ps,
                                      new_record_id=numbered[(ds, new, s, ckpt, c)], reference_record_id=numbered[(ds, ref, s, ckpt, c)]))
                if ckpt == 'best.pt' and c != 'normal' and m != 'mAP50':
                    n0 = [pct((ds, new, s, ckpt, 'normal'), m) - pct((ds, ref, s, ckpt, 'normal'), m) for s in seeds]
                    it = [a - b for a, b in zip(d, n0)]; mean_i, sd_i, lo_i, hi_i, pos_i = summ(it); pt_i, ps_i = pvals(it)
                    for s, x, y, z in zip(seeds, d, n0, it):
                        rows3.append(dict(dataset=ds, new_configuration=new, reference_configuration=ref, condition=COND[c], metric=m, seed=s,
                                          new_condition_percent=pct((ds, new, s, ckpt, c), m), reference_condition_percent=pct((ds, ref, s, ckpt, c), m),
                                          new_normal_percent=pct((ds, new, s, ckpt, 'normal'), m), reference_normal_percent=pct((ds, ref, s, ckpt, 'normal'), m),
                                          paired_change_condition_pp=x, paired_change_normal_pp=y, condition_interaction_pp=z, group_mean_pp=mean_i, group_sample_sd_pp=sd_i,
                                          ci95_low_pp=lo_i, ci95_high_pp=hi_i, positive_seeds=pos_i, n=len(seeds), p_paired_t=pt_i, p_sign_flip=ps_i,
                                          new_condition_record_id=numbered[(ds, new, s, ckpt, c)],
                                          reference_condition_record_id=numbered[(ds, ref, s, ckpt, c)], new_normal_record_id=numbered[(ds, new, s, ckpt, 'normal')],
                                          reference_normal_record_id=numbered[(ds, ref, s, ckpt, 'normal')]))

# real-condition subsets of the WSODD test set (standard training and fixed single view, six seeds)
SUB = {'low_luminance_lt110': ('Low luminance', 'Mean gray level below 110'), 'saturated_gt0p3pct': ('High saturation', 'Pixels with gray level of at least 250 exceed 0.3% of the image'),
       'captured_18h': ('Captured 18:00-18:59', 'File-name timestamp'), 'captured_19h': ('Captured 19:00-19:59', 'File-name timestamp'), 'rest_of_test': ('Remaining test images', 'In none of the four subsets')}
sub_rows = [r for r in csv.DictReader(open(RES / 'real_subset_ap.csv', encoding='utf-8-sig')) if r['family'] == 'strategy' and r['dataset'] == 'WSODD']
names = [n for n in SUB if any(r['subset'] == n for r in sub_rows)]
summary = {r['subset']: r for r in csv.DictReader(open(PAPER / '正文源/20261109/tables/real_condition_subsets_summary.csv', encoding='utf-8-sig'))}
label = {}
for n in names:
    hit = [k for k in SUB if n == k or n.startswith(k)]
    label[n] = SUB[hit[0]] if hit else (n, '')
assert all(label[n][0] in summary for n in names), (names, list(summary))
subs = []
for r in sorted(sub_rows, key=lambda r: (names.index(r['subset']), r['arm'], S6.index(r['seed']))):
    lab, definition = label[r['subset']]
    subs.append(dict(subset=lab, definition=definition, images=summary[lab]['images'], objects=summary[lab]['objects'], small_objects=summary[lab]['small_objects'], dataset='WSODD', split='test',
                     configuration=CONFIG[r['arm']], seed=r['seed'], **{'mAP50-95_percent': r['all'], 'APsmall_percent': r['small'], 'APmedium_percent': r['medium'], 'APlarge_percent': r['large'], 'mAP50_percent': r['mAP50']}))
write(OUT / 'S1/Table_S1_real_condition_subsets.csv', subs)
(OUT / 'S1/Real_condition_subset_definitions.json').write_text(json.dumps({'dataset': 'WSODD test set', 'note': 'Subsets were fixed before scoring; the four subsets overlap. Scores are obtained by re-scoring the saved normal-input predictions on the images of each subset.',
    'subsets': [dict(subset=label[n][0], definition=label[n][1], images=int(summary[label[n][0]]['images']), objects=int(summary[label[n][0]]['objects']), small_objects=int(summary[label[n][0]]['small_objects'])) for n in names]}, indent=2) + '\n', encoding='utf-8')
val_sub = {(r['subset'], r['arm'], r['seed']): r for r in sub_rows}
for n in names:
    for m, col in (('mAP50-95', 'all'), ('APsmall', 'small'), ('APmedium', 'medium'), ('APlarge', 'large'), ('mAP50', 'mAP50')):
        if any(val_sub[(n, a, s)][col] in ('', 'nan') for a in ('E0', 'E1') for s in S6):
            continue
        d = [float(val_sub[(n, 'E1', s)][col]) - float(val_sub[(n, 'E0', s)][col]) for s in S6]
        mean, sd, lo, hi, pos = summ(d); pt, ps = pvals(d)
        for s, x in zip(S6, d):
            rows2.append(dict(analysis='real_condition_subsets', dataset='WSODD', seed=s, new_configuration='Fixed single view', reference_configuration='Standard training', checkpoint='best.pt',
                              condition='Normal; subset: ' + label[n][0], metric=m, paired_change_pp=x, group_mean_pp=mean, group_sample_sd_pp=sd, ci95_low_pp=lo, ci95_high_pp=hi,
                              positive_seeds=pos, n=6, p_paired_t=pt, p_sign_flip=ps, new_record_id='', reference_record_id=''))
write(OUT / 'S2/Table_S2_paired_differences.csv', rows2)
write(OUT / 'S3/Table_S3_condition_interactions.csv', rows3)

# ---------------------------------------------------------------- S4 unchanged
for f in ('Table_S4_efficiency.csv', 'Table_S4_image_timings.csv', 'Efficiency_Protocol.json'):
    (OUT / 'S4').mkdir(parents=True, exist_ok=True); shutil.copyfile(OLD / 'S4' / f, OUT / 'S4' / f)

# ---------------------------------------------------------------- S5 object sizes, size bins, classes (SDS and WSODD, standard training and fixed single view)
DS = {'SeaDronesSee': 'SeaDronesSee validation', 'WSODD': 'WSODD test', 'FloW': 'FloW test'}
a_rows = []
for r in csv.DictReader(open(RES / 'gt_size_by_dataset.csv', encoding='utf-8-sig')):
    a_rows.append(dict(evaluation_set=DS[r['dataset']], category='all', objects=r['boxes'], small=r['small'], medium=r['medium'], large=r['large'], below_16sq=r['lt16'], **{'16sq_to_32sq': r['16_32'], '32sq_to_64sq': r['32_64'], '64sq_to_96sq': r['64_96']}))
for r in csv.DictReader(open(RES / 'gt_by_class.csv', encoding='utf-8-sig')):
    a_rows.append(dict(evaluation_set=DS[r['dataset']], category=r['category'], objects=r['boxes'], small=r['small'], medium=r['medium'], large=r['large'], below_16sq='', **{'16sq_to_32sq': '', '32sq_to_64sq': '', '64sq_to_96sq': ''}))
write(OUT / 'S5/Table_S5a_objects_by_class_and_size.csv', a_rows)
BINS = (('all', 'mAP50-95'), ('small', 'APsmall'), ('medium', 'APmedium'), ('large', 'APlarge'), ('lt16', 'AP below 16sq px'), ('16_32', 'AP 16sq-32sq px'), ('32_64', 'AP 32sq-64sq px'), ('64_96', 'AP 64sq-96sq px'))
b_rows = []
keep = lambda r: r['family'] == 'strategy' and r['dataset'] in ('SeaDronesSee', 'WSODD')
for r in csv.DictReader(open(RES / 'size_bin_ap.csv', encoding='utf-8-sig')):
    if keep(r):
        b_rows.append(dict(table='per-run AP (%)', dataset=r['dataset'], configuration=CONFIG[r['arm']], seed=r['seed'], **{lab: r[col] for col, lab in BINS}))
for r in csv.DictReader(open(RES / 'paired_by_size_bin.csv', encoding='utf-8-sig')):
    if keep(r) and r['metric'] != 'mAP50':
        b_rows.append(dict(table='paired change (pp), fixed single view minus standard training', dataset=r['dataset'], configuration='', seed='', size_range=dict(BINS)[r['metric']], n=r['n'],
                           mean_pp=r['mean'], sample_sd_pp=r['sd'], ci95_low_pp=r['ci_low'], ci95_high_pp=r['ci_high'], positive_seeds=r['positive'], negative_seeds=r['negative']))
write(OUT / 'S5/Table_S5b_size_bin_results.csv', b_rows, ['table', 'dataset', 'configuration', 'seed'] + [lab for _, lab in BINS] + ['size_range', 'n', 'mean_pp', 'sample_sd_pp', 'ci95_low_pp', 'ci95_high_pp', 'positive_seeds', 'negative_seeds'])
c_rows = []
for r in csv.DictReader(open(RES / 'per_class_ap.csv', encoding='utf-8-sig')):
    if keep(r):
        c_rows.append(dict(table='per-run AP (%)', dataset=r['dataset'], configuration=CONFIG[r['arm']], seed=r['seed'], category=r['category'], AP=r['AP'], APsmall=r['APsmall']))
for r in csv.DictReader(open(RES / 'paired_by_class.csv', encoding='utf-8-sig')):
    if keep(r):
        c_rows.append(dict(table='paired change (pp), fixed single view minus standard training', dataset=r['dataset'], configuration='', seed='', category=r['category'],
                           metric='AP' if r['metric'] == 'all' else 'APsmall', n=r['n'], mean_pp=r['mean'], sample_sd_pp=r['sd'], ci95_low_pp=r['ci_low'], ci95_high_pp=r['ci_high'],
                           positive_seeds=r['positive'], negative_seeds=r['negative']))
write(OUT / 'S5/Table_S5c_per_class_results.csv', c_rows, ['table', 'dataset', 'configuration', 'seed', 'category', 'AP', 'APsmall', 'metric', 'n', 'mean_pp', 'sample_sd_pp', 'ci95_low_pp', 'ci95_high_pp', 'positive_seeds', 'negative_seeds'])

audit = dict(records_full_split=len(rows1), records_by_dataset=dict(collections.Counter(r['dataset'] for r in rows1)), checkpoints=dict(collections.Counter(r['checkpoint'] for r in rows1)),
             subset_records=len(subs), s2_rows=len(rows2), s2_analyses=dict(collections.Counter(r['analysis'] for r in rows2)), s3_rows=len(rows3), s5_rows=dict(a=len(a_rows), b=len(b_rows), c=len(c_rows)),
             kept_record_numbers='R001-R078 as in the previous supplement', files=sorted(str(p.relative_to(OUT)).replace('\\', '/') for p in OUT.rglob('*') if p.is_file()))
(OUT.parent / 'supplement_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False, indent=1))
