"""CPU-only recomputation from the sealed 20260924 predictions. No inference and no training.

Outputs under OUT/results:
  stats_existing_contrasts.csv  t intervals and exact sign-flip p values for every existing contrast
  gt_size_by_dataset.csv, gt_by_class.csv  target-size distributions
  real_subsets.json             WSODD real-condition subsets, fixed before any subset is scored
  per_key/*.json                AP by size bin, by class and by subset for each checkpoint
  sealed_consistency.json       recomputed all/small AP against the sealed CSV
  size_bin_ap.csv, per_class_ap.csv, real_subset_ap.csv, paired tables
  flow_threshold_mapping.json, flow_cbam_table.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import contextlib, csv, glob, gzip, io, itertools, math, re, zipfile, collections
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
bootstrap()
RES = OUT / 'results'
ANALYSIS_DIR = RING3 / '01_当前论文/基础依赖/20260924_统一评价修订'
BIG = 1e5 ** 2
AREAS = [('all', 0, BIG), ('small', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large', 96 ** 2, BIG),
         ('lt16', 0, 16 ** 2), ('16_32', 16 ** 2, 32 ** 2), ('32_64', 32 ** 2, 64 ** 2), ('64_96', 64 ** 2, 96 ** 2)]


def tcrit(n):
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def paired_summary(values):
    n = len(values); m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else float('nan')
    hw = tcrit(n) * sd / math.sqrt(n) if n > 1 else float('nan')
    flips = [sum(s * v for s, v in zip(signs, values)) / n for signs in itertools.product((1, -1), repeat=n)]
    p = sum(abs(f) >= abs(m) - 1e-12 for f in flips) / len(flips)
    from scipy import stats
    tp = float(stats.ttest_1samp(values, 0).pvalue) if n > 1 and sd > 0 else float('nan')
    return dict(n=n, mean=m, sd=sd, ci_low=m - hw, ci_high=m + hw, excludes_zero=(m - hw > 0 or m + hw < 0), positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values), t_p=tp, signflip_p=p, min_possible_signflip_p=2 / len(flips))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def existing_stats():
    groups = collections.defaultdict(list)
    for r in csv.DictReader(open(ANALYSIS_DIR / 'paired_effects.csv', encoding='utf-8-sig')):
        groups[(r['analysis'], r['dataset'], r['condition_raw'], r['new_arm'] + ' - ' + r['reference_arm'], r['metric'])].append(float(r['paired_change_pp']))
    for r in csv.DictReader(open(ANALYSIS_DIR / 'condition_interactions_per_seed.csv', encoding='utf-8-sig')):
        if r['condition'].lower() != 'normal':
            groups[('interaction', r['dataset'], r['condition'], 'Fixed single view - Standard', r['metric'])].append(float(r['condition_interaction_pp']))
    rows = []
    for k, v in sorted(groups.items()):
        rows.append(dict(analysis=k[0], dataset=k[1], condition=k[2], contrast=k[3], metric=k[4], **paired_summary(v)))
    write_csv(RES / 'stats_existing_contrasts.csv', rows)
    print('existing contrasts', len(rows), 'excluding zero', sum(r['excludes_zero'] for r in rows), flush=True)


def size_tables():
    by_ds, by_cl = [], []
    for ds in ('SeaDronesSee', 'WSODD', 'FloW'):
        gt = read(OUT / 'data' / ds / 'ground_truth.json')
        names = {c['id']: c['name'] for c in gt['categories']}
        c = collections.Counter(); per = collections.defaultdict(collections.Counter)
        for a in gt['annotations']:
            x = a['area']
            for lbl, lo, hi in AREAS[1:]:
                if lo <= x < hi:
                    c[lbl] += 1; per[a['category_id']][lbl] += 1
            per[a['category_id']]['boxes'] += 1
        total = len(gt['annotations'])
        by_ds.append(dict(dataset=ds, images=len(gt['images']), boxes=total, **{k: c[k] for k in ('small', 'medium', 'large', 'lt16', '16_32', '32_64', '64_96')},
                          small_pct=100 * c['small'] / total, medium_pct=100 * c['medium'] / total, large_pct=100 * c['large'] / total))
        for cid, v in per.items():
            by_cl.append(dict(dataset=ds, category=names[cid], boxes=v['boxes'], small=v['small'], medium=v['medium'], large=v['large'], small_pct=100 * v['small'] / v['boxes']))
    write_csv(RES / 'gt_size_by_dataset.csv', by_ds); write_csv(RES / 'gt_by_class.csv', by_cl)
    print(by_ds, flush=True)


def real_subsets():
    path = RES / 'real_subsets.json'
    if path.exists():
        return read(path)
    import cv2, numpy as np
    records = read(OUT / 'data/WSODD/local_records.json')
    gt = read(OUT / 'data/WSODD/ground_truth.json')
    boxes = collections.Counter(a['image_id'] for a in gt['annotations'])
    small = collections.Counter(a['image_id'] for a in gt['annotations'] if a['area'] < 32 ** 2)
    stats_rows = []
    for r in records:
        im = cv2.imdecode(np.fromfile(r['local_image'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        name = Path(r['image']).name
        m = re.search(r'_(20\d{6})(\d{2})\d{2}', name) or re.search(r'WIN_(20\d{6})_(\d{2})_', name) or re.match(r'(20\d{6})(\d{2})\d{4}', name)
        stats_rows.append(dict(id=r['id'], name=name, luminance=float(im.mean()), saturated=float((im >= 250).mean()), hour=int(m.group(2)) if m else None))
    write_csv(RES / 'wsodd_test_image_stats.csv', stats_rows)
    defs = {'low_luminance_lt110': lambda s: s['luminance'] < 110, 'saturated_gt0p3pct': lambda s: s['saturated'] > 0.003, 'captured_19h': lambda s: s['hour'] == 19, 'captured_18h': lambda s: s['hour'] == 18}
    subsets = {k: [s['id'] for s in stats_rows if f(s)] for k, f in defs.items()}
    used = set().union(*map(set, subsets.values()))
    subsets['rest_of_test'] = [s['id'] for s in stats_rows if s['id'] not in used]
    out = dict(definition='Fixed before scoring. Thresholds are descriptive; subsets overlap and concentrate in few acquisition sequences.',
               subsets=subsets, counts={k: dict(images=len(v), boxes=sum(boxes[i] for i in v), small_boxes=sum(small[i] for i in v)) for k, v in subsets.items()})
    write(path, out)
    print(out['counts'], flush=True)
    return out


def score_key(task):
    key, member, dataset, want_subsets = task
    target = RES / 'per_key' / (key + '.json')
    if target.exists():
        return key
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = read(OUT / 'data' / dataset / 'ground_truth.json')
    preds = json.loads(gzip.decompress(zipfile.ZipFile(EVIDENCE_ZIP).read(member)))
    subsets = read(RES / 'real_subsets.json')['subsets'] if want_subsets else {}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = gt; coco.createIndex()
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, 'bbox'); p = ev.params
        p.imgIds = sorted(coco.imgs); p.catIds = sorted(coco.cats); p.maxDets = [1, 10, 100]
        p.areaRng = [[lo, hi] for _, lo, hi in AREAS]; p.areaRngLbl = [a[0] for a in AREAS]
        ev.evaluate()
    full = ev.evalImgs
    names = {c['id']: c['name'] for c in gt['categories']}

    def summary(ids=None):
        ev.evalImgs = full if ids is None else [e if e is not None and e['image_id'] in ids else None for e in full]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.accumulate()
        P = ev.eval['precision']; res = {}
        for a, (lbl, _, _) in enumerate(AREAS):
            v = P[:, :, :, a, 2]; v = v[v > -1]
            res[lbl] = float(v.mean()) if v.size else None
        v = P[0, :, :, 0, 2]; v = v[v > -1]; res['mAP50'] = float(v.mean()) if v.size else None
        return res, P

    overall, P = summary()
    per_class = {}
    for k, cid in enumerate(p.catIds):
        row = {}
        for a, lbl in ((0, 'all'), (1, 'small')):
            v = P[:, :, k, a, 2]; v = v[v > -1]
            row[lbl] = float(v.mean()) if v.size else None
        per_class[names[cid]] = row
    result = dict(key=key, dataset=dataset, overall=overall, per_class=per_class, subsets={})
    for name, ids in subsets.items():
        result['subsets'][name] = summary(set(ids))[0]
    write(target, result)
    return key


def score_all():
    rows = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
    z = zipfile.ZipFile(EVIDENCE_ZIP)
    by_hash = {}
    for n in z.namelist():
        if n.endswith('evaluation.json'):
            e = json.loads(z.read(n))
            by_hash[e['predictions_sha256']] = n.rsplit('/', 1)[0] + '/predictions.json.gz'
    normal = [r for r in rows if r['condition_raw'] == 'normal' and r['family'] in ('strategy', 'matched')]
    tasks = []
    for r in normal:
        member = by_hash[r['predictions_sha256']]
        tasks.append((r['evaluation_key'], member, r['dataset'], r['dataset'] == 'WSODD'))
    print('score tasks', len(tasks), flush=True)
    (RES / 'per_key').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i, key in enumerate(pool.map(score_key, tasks, chunksize=1)):
            print('scored', i + 1, len(tasks), key, flush=True)
    return normal


def assemble(normal):
    size_rows, class_rows, subset_rows, consistency = [], [], [], []
    per = {}
    for r in normal:
        d = read(RES / 'per_key' / (r['evaluation_key'] + '.json'))
        per[(r['family'], r['dataset'], r['arm_raw'], r['seed'])] = d
        meta = dict(family=r['family'], dataset=r['dataset'], arm=r['arm_raw'], seed=r['seed'])
        size_rows.append({**meta, **{k: (None if v is None else 100 * v) for k, v in d['overall'].items()}})
        for name, v in d['per_class'].items():
            class_rows.append({**meta, 'category': name, 'AP': None if v['all'] is None else 100 * v['all'], 'APsmall': None if v['small'] is None else 100 * v['small']})
        for s, v in d['subsets'].items():
            subset_rows.append({**meta, 'subset': s, **{k: (None if x is None else 100 * x) for k, x in v.items()}})
        consistency.append(dict(key=r['evaluation_key'], all_diff_pp=100 * (d['overall']['all'] - float(r['mAP50-95_fraction'])), small_diff_pp=100 * (d['overall']['small'] - float(r['APsmall_fraction'])), large_diff_pp=(100 * (d['overall']['large'] - float(r['APlarge_fraction'])) if d['overall']['large'] is not None and r['APlarge_fraction'] else None)))
    write_csv(RES / 'size_bin_ap.csv', size_rows); write_csv(RES / 'per_class_ap.csv', class_rows); write_csv(RES / 'real_subset_ap.csv', subset_rows)
    worst = max(abs(c['all_diff_pp']) for c in consistency), max(abs(c['small_diff_pp']) for c in consistency)
    write(RES / 'sealed_consistency.json', dict(checked=len(consistency), max_abs_all_diff_pp=worst[0], max_abs_small_diff_pp=worst[1], rows=consistency))
    print('consistency with sealed', worst, flush=True)
    # paired contrasts by size bin, class and real subset
    contrasts = [('strategy', 'WSODD', 'E1', 'E0'), ('strategy', 'SeaDronesSee', 'E1', 'E0'), ('strategy', 'FloW', 'E2', 'E0'), ('strategy', 'FloW', 'E3', 'E0'),
                 ('matched', 'WSODD', 'B_weights_only', 'A_baseline'), ('matched', 'WSODD', 'C_CBAM_Focal_EIoU', 'B_weights_only'), ('matched', 'WSODD', 'D_clean_duplicate', 'C_CBAM_Focal_EIoU'),
                 ('matched', 'WSODD', 'E_refsoft', 'D_clean_duplicate'), ('matched', 'WSODD', 'E_refsoft', 'A_baseline')]
    seeds = collections.defaultdict(set)
    for k in per:
        seeds[k[:3]].add(k[3])
    bins, classes, subs = [], [], []
    for fam, ds, new, ref in contrasts:
        ss = sorted(seeds[(fam, ds, new)] & seeds[(fam, ds, ref)], key=int)
        pairs = [(per[(fam, ds, new, s)], per[(fam, ds, ref, s)]) for s in ss]
        base = dict(family=fam, dataset=ds, contrast=f'{new} - {ref}')
        for lbl in [a[0] for a in AREAS] + ['mAP50']:
            v = [100 * (a['overall'][lbl] - b['overall'][lbl]) for a, b in pairs if a['overall'][lbl] is not None and b['overall'][lbl] is not None]
            if len(v) >= 2:
                bins.append({**base, 'metric': lbl, **paired_summary(v)})
        for name in pairs[0][0]['per_class']:
            for lbl in ('all', 'small'):
                v = [100 * (a['per_class'][name][lbl] - b['per_class'][name][lbl]) for a, b in pairs if a['per_class'][name][lbl] is not None and b['per_class'][name][lbl] is not None]
                if len(v) >= 2:
                    classes.append({**base, 'category': name, 'metric': lbl, **paired_summary(v)})
        for s in pairs[0][0]['subsets']:
            for lbl in ('all', 'mAP50', 'small'):
                v = [100 * (a['subsets'][s][lbl] - b['subsets'][s][lbl]) for a, b in pairs if a['subsets'][s][lbl] is not None and b['subsets'][s][lbl] is not None]
                if len(v) >= 2:
                    subs.append({**base, 'subset': s, 'metric': lbl, **paired_summary(v)})
    write_csv(RES / 'paired_by_size_bin.csv', bins); write_csv(RES / 'paired_by_class.csv', classes); write_csv(RES / 'paired_by_real_subset.csv', subs)


def flow_threshold():
    from PIL import Image
    root = RING3 / '02_当前实验/数据集/FloW_1124_335_541'
    out = {}
    for split in ('train', 'test'):
        c = collections.Counter()
        for lab in (root / 'labels' / split).glob('*.txt'):
            img = next(iter((root / 'images' / split).glob(lab.stem + '.*')), None)
            if img is None:
                continue
            with Image.open(img) as im:
                w, h = im.size
            s = 960 / max(w, h)
            for line in lab.read_text().splitlines():
                if not line.strip():
                    continue
                _, _, _, bw, bh = map(float, line.split())
                area = bw * w * bh * h
                c['boxes'] += 1
                c['coco_small_original_lt_32sq'] += area < 32 ** 2
                c['lt_1024_after_resize_only'] += area * s * s < 1024
                c['both'] += (area < 32 ** 2) and (area * s * s < 1024)
                c['lt_1024_resize_only_not_coco_small'] += (area >= 32 ** 2) and (area * s * s < 1024)
                c[f'image_{w}x{h}'] += 0
        out[split] = dict(c)
    idx = {}
    for f in glob.glob(str(BACKUP / 'manifests/003_project_records_03*.json')) + glob.glob(str(BACKUP / 'manifests/00[1-7]_*.json')):
        d = read(f)
        items = d.get('files') or d.get('entries') or d.get('members') or d
        if isinstance(items, dict):
            items = items.get('files', [])
        for it in items:
            pth = it.get('path') or it.get('archive_path') or it.get('relative_path') or it.get('name')
            if pth and re.match(r'flow_source_retrain_20260920/release_v1/runs/FloW_E3_seed\d/runtime_evidence\.json$', pth):
                idx[pth] = Path(f).name[:-5]
    e3 = {}
    for pth, pkg in sorted(idx.items()):
        z = zipfile.ZipFile(BACKUP / pkg)
        for n in (f'autodl-tmp/{pth}', pth):
            try:
                d = json.loads(z.read(n)); t = d['total_target_counts']
                e3[pth.split('/')[-2]] = dict(degraded_targets=t['degraded_targets'], degraded_small_targets=t['degraded_small_targets'], weighted_share_pct=100 * t['degraded_small_targets'] / t['degraded_targets'])
                break
            except KeyError:
                pass
    out['small_object_weighting_actual_post_augmentation'] = e3
    write(RES / 'flow_threshold_mapping.json', out)
    print(out, flush=True)


def flow_cbam():
    src = RING3 / '03_历史归档/独立补充实验/flow_source_cbam_20260920/FINAL_OFFICIAL_COCO_PER_SEED_20260921.csv'
    rows = [r for r in csv.DictReader(open(src, encoding='utf-8-sig')) if r['split'] == 'test']
    val = collections.defaultdict(dict)
    for r in rows:
        val[r['arm']][r['seed']] = r
    arms = sorted(val)
    print('FloW CBAM arms', arms, flush=True)
    base = 'E0'; out = []
    anchors = {}
    uni = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
    for arm in ('E0', 'E2', 'E3'):
        anchors[arm] = max(abs(100 * float(next(u['mAP50-95_fraction'] for u in uni if u['dataset'] == 'FloW' and u['arm_raw'] == arm and u['seed'] == s and u['family'] == 'strategy')) - 100 * float(val[arm][s]['mAP50-95'])) for s in val[arm])
    for arm in arms:
        ss = sorted(val[arm], key=int)
        for metric in ('mAP50-95', 'APsmall', 'APmedium', 'APlarge', 'mAP50'):
            abs_v = [100 * float(val[arm][s][metric]) for s in ss]
            row = dict(arm=arm, metric=metric, mean_abs=sum(abs_v) / len(abs_v), sd_abs=(sum((x - sum(abs_v) / len(abs_v)) ** 2 for x in abs_v) / (len(abs_v) - 1)) ** .5)
            if arm != base:
                row.update(paired_summary([100 * (float(val[arm][s][metric]) - float(val[base][s][metric])) for s in ss]))
            out.append(row)
    write_csv(RES / 'flow_cbam_table.csv', out)
    write(RES / 'flow_cbam_anchor_check.json', dict(max_abs_diff_pp_vs_unified=anchors))
    print('anchor differences vs unified', anchors, flush=True)


if __name__ == '__main__':
    RES.mkdir(parents=True, exist_ok=True)
    existing_stats(); size_tables(); flow_cbam(); flow_threshold(); real_subsets()
    normal = score_all()
    assemble(normal)
    print('offline analysis finished', flush=True)
