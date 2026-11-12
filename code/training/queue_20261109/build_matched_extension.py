"""Build the WSODD matched-control extension for Vast. Writes only under OUT.

New jobs (fixed before any result): the five original configurations for seeds 2, 123, 3407, plus two
single-factor configurations for seeds 0, 1, 42, 2, 123, 3407:
  cbam_only  = bundle recipe and CBAM architecture with the CIoU loss (differs from bundle only in the loss)
  focal_only = weights_only recipe with the Focal-EIoU-variant loss (differs from weights_only only in the loss)
Code: the archived refsoft release (Ultralytics 8.3.182). Only train.py changes, adding the two arms; the
five original arms keep exactly their original recipe, model and loss. The 397-image copy selection stays
the original seed-42 selection. Data and overlays come from the AutoDL backup with their recorded hashes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'recompute_20261109'))
from common import *
import glob, hashlib, re, shutil, tarfile, zipfile, datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = Path(__file__).resolve().parent
OUTDIR = RING3 / '02_当前实验/原始证据/train_20261109/matched'
CODE_SRC = MATCHED_RELEASE
CODE_NEW = OUTDIR / 'code'
ORIG_ROOT = 'refsoft_reproduction_20260910/release_v1'
NEW_ROOT = 'refsoft_extension_20261109/release_v1'
SEEDS_NEW = [2, 123, 3407]
SEEDS_ALL = [0, 1, 42, 2, 123, 3407]
ORIGINAL_ARMS = ['baseline', 'weights_only', 'bundle', 'duplicate', 'refsoft']

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
zips = {}


def from_backup(rel):
    pkg = index[rel][0]
    z = zips.setdefault(pkg, zipfile.ZipFile(BACKUP / pkg))
    for n in (f'autodl-tmp/{rel}', rel):
        try:
            return z.read(n)
        except KeyError:
            continue
    raise KeyError(rel)


# 1. Code: archived release plus the two-arm patch in train.py
if CODE_NEW.exists():
    shutil.rmtree(CODE_NEW)
shutil.copytree(CODE_SRC, CODE_NEW, ignore=shutil.ignore_patterns('__pycache__'))
t = (CODE_NEW / 'train.py').read_text(encoding='utf-8')
patches = [
    ("    vanilla = arm in {'baseline', 'weights_only'}\n    recipe = dict(recipes['baseline' if vanilla else 'wsodd_refsoft'])\n",
     "    # Extension 20261109: focal_only uses the plain architecture; cbam_only uses the CBAM architecture.\n"
     "    vanilla = arm in {'baseline', 'weights_only', 'focal_only'}\n"
     "    focal = arm in {'focal_only', 'bundle', 'duplicate', 'refsoft'}\n"
     "    recipe = dict(recipes['baseline' if vanilla else 'wsodd_refsoft'])\n"),
    ("    if arm == 'weights_only':\n        recipe.update(box=4.0, dfl=1.3)\n",
     "    if arm in {'weights_only', 'focal_only'}:\n        recipe.update(box=4.0, dfl=1.3)\n"
     "    if arm == 'focal_only':\n"
     "        recipe.update({k: recipes['wsodd_refsoft'][k] for k in ('bbox_loss', 'focal_eiou_alpha', 'focal_eiou_gamma', 'focal_eiou_eps')})\n"
     "    if arm == 'cbam_only':\n        recipe.update(bbox_loss='ciou')\n"),
    ("        assert crit.bbox_loss.loss_type == ('ciou' if vanilla else 'focal_eiou')\n",
     "        assert crit.bbox_loss.loss_type == ('focal_eiou' if focal else 'ciou')\n"),
]
for old, new in patches:
    require(t.count(old) == 1, 'Patch anchor not found: ' + old[:60])
    t = t.replace(old, new)
(CODE_NEW / 'train.py').write_text(t, encoding='utf-8', newline='\n')
manifest = read(CODE_SRC / 'release_manifest.json')
for item in manifest['files']:
    item['sha256'] = sha(CODE_NEW / item['path'])
    if item['path'] == 'train.py':
        item['source'] = 'refsoft release train.py with the 20261109 two-arm extension'
manifest['schema'] = manifest.get('schema', '') + '+extension_20261109'
write(CODE_NEW / 'release_manifest.json', manifest)
release_sha = sha(CODE_NEW / 'release_manifest.json')
changed = [i['path'] for i, o in zip(manifest['files'], read(CODE_SRC / 'release_manifest.json')['files']) if i['sha256'] != o['sha256']]
require(changed == ['train.py'], 'Unexpected code changes: ' + repr(changed))

# 2. Protocol for the new root
orig = json.loads(from_backup(f'{ORIG_ROOT}/protocol.json'))
jobs = [dict(key=f'WSODD_{arm}_seed{s}', dataset='WSODD', arm=arm, seed=s) for s in SEEDS_NEW for arm in ORIGINAL_ARMS]
jobs += [dict(key=f'WSODD_{arm}_seed{s}', dataset='WSODD', arm=arm, seed=s) for s in SEEDS_ALL for arm in ('cbam_only', 'focal_only')]
locked = {k: v for k, v in orig['locked_files'].items() if '/WSODD/' in k}
proto = dict(orig, schema='ring3.refsoft_extension_20261109.v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             root=f'/root/autodl-tmp/{NEW_ROOT}', release_sha256=release_sha, datasets={'WSODD': orig['datasets']['WSODD']},
             arms=ORIGINAL_ARMS + ['cbam_only', 'focal_only'], seeds=SEEDS_ALL, jobs=jobs, maximum_full_runs=len(jobs), locked_files=locked,
             extension_of=dict(protocol=f'/root/autodl-tmp/{ORIG_ROOT}/protocol.json', protocol_sha256=hashlib.sha256(from_backup(f'{ORIG_ROOT}/protocol.json')).hexdigest(),
                               original_seeds=orig['seeds'], new_arms={'cbam_only': 'bundle with CIoU', 'focal_only': 'weights_only with Focal-EIoU variant'}))
require(len(jobs) == 27, 'Expected 27 jobs')
OUTDIR.mkdir(parents=True, exist_ok=True)
write(OUTDIR / 'protocol.json', proto)

# Initial-state hashes of the original runs, to check that new same-seed runs start from identical tensors
init = {}
for arm in ORIGINAL_ARMS:
    for s in orig['seeds']:
        rel = f'{ORIG_ROOT}/runs/WSODD_{arm}_seed{s}/runtime.json'
        alt = f'refsoft_reproduction_20260910/release_v1/runs_recovery_v1/WSODD_{arm}_seed{s}/runtime.json'
        for r in (rel, alt):
            if r in index:
                init.setdefault(f'{arm}_seed{s}', json.loads(from_backup(r))['initial_state_sha256'])
write(OUTDIR / 'original_initial_states.json', init)

# 3. Files to place under /root/autodl-tmp on Vast
members = {}
inventory_rel = orig['datasets']['WSODD']['inventory'].split('/root/autodl-tmp/')[1]
inventory = json.loads(from_backup(inventory_rel))
for row in inventory:
    if row['split'] in ('train', 'val'):
        for kind in ('image', 'label'):
            members[row[kind].split('/root/autodl-tmp/')[1]] = row[kind + '_sha256']
for path, digest in locked.items():
    members[path.split('/root/autodl-tmp/')[1]] = digest
for rel in list(index):
    if rel.startswith(f'{ORIG_ROOT}/manifests/WSODD/') or rel.startswith(f'{ORIG_ROOT}/overlays/WSODD/') or rel.startswith(f'{ORIG_ROOT}/smoke_inputs/WSODD/'):
        if index[rel][1]:
            members.setdefault(rel, index[rel][1])
for rel in ('ring3_setup/autodl/datasets/wsodd_sequence_disjoint_v1.yaml', 'weights/yolov8s.pt', f'{ORIG_ROOT}/protocol.json'):
    members[rel] = index[rel][1]
tar_path = OUTDIR / 'matched_train_bundle.tar'
count = 0
with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
    for rel, digest in sorted(members.items()):
        data = from_backup(rel)
        require(hashlib.sha256(data).hexdigest() == digest, 'Backup hash mismatch ' + rel)
        info = tarfile.TarInfo('autodl-tmp/' + rel); info.size = len(data); info.mode = 0o644
        import io
        tar.addfile(info, io.BytesIO(data)); count += 1
    tar.add(CODE_NEW, arcname=f'autodl-tmp/{NEW_ROOT}/code', filter=lambda ti: None if '__pycache__' in ti.name else ti)
    tar.add(OUTDIR / 'protocol.json', arcname=f'autodl-tmp/{NEW_ROOT}/protocol.json')
    tar.add(OUTDIR / 'original_initial_states.json', arcname=f'autodl-tmp/{NEW_ROOT}/original_initial_states.json')
summary = dict(jobs=len(jobs), release_sha256=release_sha, data_files=count, original_initial_states=len(init),
               tar=str(tar_path), tar_gb=round(tar_path.stat().st_size / 1e9, 3), tar_sha256=sha(tar_path))
write(OUTDIR / 'bundle_summary.json', summary)
print(json.dumps(summary, indent=1))
