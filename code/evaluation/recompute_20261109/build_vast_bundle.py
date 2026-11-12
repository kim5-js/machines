"""Build the 210-job list (162 + 30 matched + 18 FloW), the 5 reproduction gates and one upload tar for Vast.

Every file in the tar is listed with its SHA-256 in MANIFEST.json. Images and checkpoints are taken
from verified sources and re-hashed against the sealed 20260924 records before they are packed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import csv, glob, tarfile, zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = Path(__file__).resolve().parent
VAST = OUT / 'vast'
VAST.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig')))
by_key = {r['evaluation_key']: r for r in rows}
normal = {r['experiment_id']: r for r in rows if r['condition_raw'] == 'normal'}

# Backup index for checkpoints not yet extracted
index = {}
for f in glob.glob(str(BACKUP / 'manifests/*.json')):
    d = read(f)
    items = d.get('files') or d.get('entries') or d.get('members') or d
    if isinstance(items, dict):
        items = items.get('files', [])
    for it in items:
        p = it.get('path') or it.get('archive_path') or it.get('relative_path') or it.get('name')
        if p:
            index[p] = Path(f).name[:-5]
zips = {}


def extract(rel, dest, expected):
    if dest.exists() and sha(dest) == expected:
        return
    z = zips.setdefault(index[rel], zipfile.ZipFile(BACKUP / index[rel]))
    for n in (f'autodl-tmp/{rel}', rel):
        try:
            dest.write_bytes(z.read(n)); break
        except KeyError:
            continue
    require(sha(dest) == expected, 'Hash mismatch ' + rel)


# 1. Jobs: the fixed 162 plus matched and FloW blocks under the two existing conditions
base = read(OUT / 'jobs.json')
jobs = []
for j in base['jobs']:
    jobs.append({**j, 'weights': 'weights/' + Path(j['weights']).name})
for fam, ds, block in (('matched', 'WSODD', 'matched_condition'), ('strategy', 'FloW', 'flow_condition')):
    for eid, r in sorted(normal.items()):
        if r['family'] != fam or r['dataset'] != ds:
            continue
        dest = OUT / 'weights' / f'{eid}_best.pt'
        extract(r['checkpoint'].split('/root/autodl-tmp/')[-1], dest, r['checkpoint_sha256'])
        for c in ('low_light_strong', 'glare_strong'):
            jobs.append(dict(experiment_id=eid, dataset=ds, arm=r['arm_raw'], seed=r['seed'], split=r['split'], family=fam, key=f'{eid}__{c}', condition=c,
                             weights=f'weights/{eid}_best.pt', weights_sha256=r['checkpoint_sha256'], block=block))
require(len(jobs) == 210 and len({j['key'] for j in jobs}) == 210, 'Expected 210 unique jobs')

# 2. Gates: existing three plus one matched and one FloW checkpoint, all with sealed values
gate = []
for key in ('AD0076__low_light_strong', 'AD0072__glare_strong', 'AD0090__normal', 'AD0128__normal', 'AD0177__normal'):
    r = by_key[key]
    gate.append(dict(key='GATE_' + key, sealed_key=key, experiment_id=r['experiment_id'], dataset=r['dataset'], arm=r['arm_raw'], seed=r['seed'], split=r['split'],
                     family=r['family'], condition=r['condition_raw'], weights=f"weights/{r['experiment_id']}_best.pt", weights_sha256=r['checkpoint_sha256'], block='gate',
                     sealed_input_pixels_sha256=r['input_pixels_sha256'], sealed_predictions_sha256=r['predictions_sha256'],
                     sealed={k: float(r[k + '_fraction']) for k in ('mAP50-95', 'mAP50', 'APsmall', 'APmedium', 'APlarge', 'AR100')}))
common_cfg = dict(severity=base['severity'], glare_geometry=base['glare_geometry'], namespace_root=base['namespace_root'], inference=base['inference'])
write(VAST / 'jobs_vast.json', dict(**common_cfg, jobs=jobs))
write(VAST / 'gate_vast.json', dict(**common_cfg, jobs=gate))

# 3. Requirements from the original mdc-yolo26 environment (torch is installed separately)
env = next(e for e in read(BACKUP / 'environment_versions.json')['environments'] if e['name'] == 'mdc-yolo26')
skip = ('torch', 'torchvision', 'triton', 'nvidia-', 'pip', 'setuptools', 'wheel', 'ultralytics')
req = [f"{p['name']}=={p['version']}" for p in env['details']['packages'] if not p['name'].lower().startswith(skip) or p['name'].lower() == 'ultralytics-thop']
req.append('pycocotools==2.0.10')
(VAST / 'requirements_vast.txt').write_text('\n'.join(sorted(set(req))) + '\n', encoding='utf-8')

# 4. Files to pack
members = {}
for ds in ('WSODD', 'SeaDronesSee', 'FloW'):
    members[f'data/{ds}/records.json'] = OUT / 'data' / ds / 'records.json'
    members[f'data/{ds}/ground_truth.json'] = OUT / 'data' / ds / 'ground_truth.json'
    if ds == 'FloW':
        recs = [{**r, 'local_image': str(RING3 / '02_当前实验/数据集/FloW_1124_335_541/images/test' / Path(r['image']).name)} for r in read(OUT / 'data/FloW/records.json')]
    else:
        recs = read(OUT / 'data' / ds / 'local_records.json')
    for r in recs:
        members[f"images/{ds}/{Path(r['image']).name}"] = (Path(r['local_image']), r['image_sha256'])
for j in jobs + gate:
    members[j['weights']] = (OUT / j['weights'], j['weights_sha256'])
code_files = {'code/recompute_vast.py': HERE / 'recompute_vast.py',
              'code/setup/evaluate_pretrained_sgd_robustness.py': AUTODL_CODE / 'evaluate_pretrained_sgd_robustness.py',
              'code/setup/e1_mdg_config.json': MDG_CONFIG,
              'jobs_vast.json': VAST / 'jobs_vast.json', 'gate_vast.json': VAST / 'gate_vast.json', 'requirements_vast.txt': VAST / 'requirements_vast.txt'}
members.update(code_files)
for src_root, arc in ((FLOW_BUNDLE / 'vendor', 'code/flow_vendor'), (FLOW_BUNDLE / 'historical_code', 'code/flow_historical_code'), (MATCHED_RELEASE / 'vendor', 'code/matched_vendor')):
    for p in src_root.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            members[f'{arc}/{p.relative_to(src_root).as_posix()}'] = p

manifest = {}
for arc, src in members.items():
    path, expected = (src if isinstance(src, tuple) else (src, None))
    digest = sha(path)
    require(expected is None or digest == expected, 'Source hash mismatch ' + arc)
    manifest[arc] = dict(sha256=digest, bytes=path.stat().st_size)
write(VAST / 'MANIFEST.json', manifest)
require(manifest['code/setup/evaluate_pretrained_sgd_robustness.py']['sha256'] == SEALED_TRANSFORM_SHA, 'Transform drift')
require(manifest['code/setup/e1_mdg_config.json']['sha256'] == SEALED_MDG_SHA, 'MDG drift')
tar_path = VAST / 'recompute_vast_bundle.tar'
with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
    tar.add(VAST / 'MANIFEST.json', arcname='MANIFEST.json')
    for arc, src in sorted(members.items()):
        tar.add(src[0] if isinstance(src, tuple) else src, arcname=arc, recursive=False)
summary = dict(jobs=len(jobs), gate=len(gate), blocks={b: sum(j['block'] == b for j in jobs) for b in sorted({j['block'] for j in jobs})}, files=len(manifest),
               payload_gb=round(sum(m['bytes'] for m in manifest.values()) / 1e9, 3), tar=str(tar_path), tar_sha256=sha(tar_path), tar_gb=round(tar_path.stat().st_size / 1e9, 3))
write(VAST / 'bundle_summary.json', summary)
print(json.dumps(summary, indent=1))
