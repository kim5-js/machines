"""Verify inputs against the sealed evaluation and build the fixed job list. Writes only under OUT."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import csv, glob, re, zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
bootstrap()
OUT.mkdir(parents=True, exist_ok=True)
report = {}

# Evaluator, transform and MDG identity
import pycocotools, pycocotools.cocoeval as ce, importlib.metadata as md
report['pycocotools'] = md.version('pycocotools')
report['cocoeval_sha_matches_sealed_ignoring_crlf'] = hashlib.sha256(Path(ce.__file__).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest() == SEALED_COCOEVAL_SHA
require(report['cocoeval_sha_matches_sealed_ignoring_crlf'], 'Evaluator source differs from sealed copy')
report['transform_sha_matches_sealed'] = sha(AUTODL_CODE / 'evaluate_pretrained_sgd_robustness.py') == SEALED_TRANSFORM_SHA
report['mdg_sha_matches_sealed'] = sha(MDG_CONFIG) == SEALED_MDG_SHA
print(report)

# Sealed records and ground truth
ev = zipfile.ZipFile(EVIDENCE_ZIP)
for ds in ('WSODD', 'SeaDronesSee', 'FloW'):
    dest = OUT / 'data' / ds
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('records.json', 'ground_truth.json'):
        (dest / name).write_bytes(ev.read(f'data/{ds}/{name}'))

# Backup index with file hashes
index = {}
for f in glob.glob(str(BACKUP / 'manifests/*.json')):
    d = read(f)
    items = d.get('files') or d.get('entries') or d.get('members') or d
    if isinstance(items, dict):
        items = items.get('files', [])
    for it in items:
        p = it.get('path') or it.get('archive_path') or it.get('relative_path') or it.get('name')
        if p:
            index[p] = (Path(f).name[:-5], it.get('sha256'))

wdir = OUT / 'weights'
wdir.mkdir(exist_ok=True)
zips = {}


def extract(rel, dest, expected=None):
    pkg, listed = index[rel]
    expected = expected or listed
    if dest.exists() and sha(dest) == expected:
        return expected
    z = zips.setdefault(pkg, zipfile.ZipFile(BACKUP / pkg))
    for n in (f'autodl-tmp/{rel}', rel):
        try:
            dest.write_bytes(z.read(n))
            break
        except KeyError:
            continue
    require(sha(dest) == expected, 'Weight hash mismatch ' + rel)
    return expected


# Local image location, checked byte for byte
sds_origin = {}
for line in SDS_VAL_TSV.read_text(encoding='utf-8').splitlines():
    origin, name = line.split('\t')
    sds_origin[name] = origin
for ds in ('WSODD', 'SeaDronesSee'):
    records = read(OUT / 'data' / ds / 'records.json')
    missing, mismatch, out = [], [], []
    for r in records:
        name = Path(r['image']).name
        p = WSODD_IMAGES / name if ds == 'WSODD' else SDS_ROOT / sds_origin.get(name, 'val') / name
        if not p.exists():
            missing.append(name)
            continue
        if sha(p) != r['image_sha256']:
            mismatch.append(name)
            p = OUT / 'data' / ds / 'images_from_backup' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            extract(r['image'].split('/root/autodl-tmp/')[1], p, r['image_sha256'])
        out.append({**r, 'local_image': str(p)})
    require(len(out) == len(records), 'Unverified images ' + ds)
    report[ds] = dict(records=len(records), verified=len(out), missing=len(missing), restored_from_backup=len(mismatch),
                      examples=(missing + mismatch)[:3])
    print(ds, report[ds], flush=True)
    write(OUT / 'data' / ds / 'local_records.json', out)

# Fixed job list
rows = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
strategy = [r for r in rows if r['family'] == 'strategy' and r['dataset'] in SEEDS_ALL and r['condition_raw'] == 'normal']
require(len(strategy) == 18, 'Expected 18 strategy checkpoints')
jobs = []
for r in strategy:
    rel = r['checkpoint'].split('/root/autodl-tmp/')[-1]
    eid = r['experiment_id']
    best = wdir / f'{eid}_best.pt'
    extract(rel, best, r['checkpoint_sha256'])
    base = dict(experiment_id=eid, dataset=r['dataset'], arm=r['arm_raw'], seed=r['seed'], split=r['split'], family='strategy')
    if r['dataset'] == 'WSODD' and r['seed'] in ('42', '123', '3407'):
        for c in ('low_light_strong', 'glare_strong'):
            jobs.append({**base, 'key': f'{eid}__{c}', 'condition': c, 'weights': str(best), 'weights_sha256': r['checkpoint_sha256'], 'block': 'six_seed_condition'})
    for c in SEVERITY:
        jobs.append({**base, 'key': f'{eid}__{c}', 'condition': c, 'weights': str(best), 'weights_sha256': r['checkpoint_sha256'], 'block': 'severity'})
    if r['dataset'] == 'SeaDronesSee':
        last_rel = rel.rsplit('/', 1)[0] + '/last.pt'
        last = wdir / f'{eid}_last.pt'
        digest = extract(last_rel, last)
        jobs.append({**base, 'key': f'{eid}_last__normal', 'condition': 'normal', 'weights': str(last), 'weights_sha256': digest, 'block': 'sds_last_checkpoint'})
order = {'six_seed_condition': 0, 'sds_last_checkpoint': 1, 'severity': 2}
sev = list(SEVERITY)
jobs.sort(key=lambda j: (order[j['block']], sev.index(j['condition']) if j['condition'] in SEVERITY else -1, j['dataset'] != 'WSODD', j['experiment_id']))
write(OUT / 'jobs.json', dict(inference=INFERENCE, severity=SEVERITY, glare_geometry=GLARE_GEOMETRY, namespace_root=NAMESPACE_ROOT, jobs=jobs))
report['jobs'] = {b: sum(j['block'] == b for j in jobs) for b in order}

# Reproduction gate: existing evaluations with sealed values
gate = []
for pick in (('WSODD', 'E0', '0', 'low_light_strong'), ('SeaDronesSee', 'E1', '1', 'glare_strong'), ('WSODD', 'E1', '42', 'normal')):
    r = next(x for x in rows if (x['dataset'], x['arm_raw'], x['seed'], x['condition_raw']) == pick and x['family'] == 'strategy')
    gate.append(dict(key='GATE_' + r['evaluation_key'], sealed_key=r['evaluation_key'], experiment_id=r['experiment_id'], dataset=r['dataset'], arm=r['arm_raw'], seed=r['seed'],
                     split=r['split'], family='strategy', condition=r['condition_raw'], weights=str(wdir / f"{r['experiment_id']}_best.pt"), weights_sha256=r['checkpoint_sha256'],
                     block='gate', sealed_input_pixels_sha256=r['input_pixels_sha256'], sealed={k: float(r[k + '_fraction']) for k in ('mAP50-95', 'mAP50', 'APsmall', 'APmedium', 'APlarge', 'AR100')}))
write(OUT / 'gate_jobs.json', gate)
write(OUT / 'prepare_report.json', report)
print(report['jobs'], 'gate', [g['key'] for g in gate])
