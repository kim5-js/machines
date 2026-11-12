"""Mirror the archived flow_source_retrain_20260920 release (without its run outputs) for Vast, plus the
three extra pretrained weights its gates check, the extension wrapper and the original E0/E2/E3
initialization evidence for seeds 0-2. Every file is taken from the AutoDL backup and hash-checked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'recompute_20261109'))
from common import *
import glob, hashlib, io, tarfile, zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = Path(__file__).resolve().parent
OUTDIR = RING3 / '02_当前实验/原始证据/train_20261109/flow'
OUTDIR.mkdir(parents=True, exist_ok=True)
R = 'flow_source_retrain_20260920/release_v1/'
SKIP = ('runs/', 'smoke_runs/', 'exports/', 'evaluations/', 'logs/')
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


members = [rel for rel, (pkg, digest) in index.items() if rel.startswith(R) and digest and not rel[len(R):].startswith(SKIP)]
members += ['weights/yolo11s.pt', 'weights/yolo26s.pt', 'weights/rtdetr-l.pt']
evidence = {}
for arm in ('E0', 'E2', 'E3'):
    for s in (0, 1, 2):
        rel = f'{R}runs/FloW_{arm}_seed{s}/runtime_evidence.json'
        evidence[f'FloW_{arm}_seed{s}'] = json.loads(from_backup(rel))['initialization']['state_sha256']
write(OUTDIR / 'original_initial_states.json', evidence)
tar_path = OUTDIR / 'flow_train_bundle.tar'
with tarfile.open(tar_path, 'w', format=tarfile.PAX_FORMAT) as tar:
    for rel in sorted(members):
        data = from_backup(rel)
        require(hashlib.sha256(data).hexdigest() == index[rel][1], 'Backup hash mismatch ' + rel)
        info = tarfile.TarInfo('autodl-tmp/' + rel); info.size = len(data); info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    tar.add(HERE / 'flow_ext_train.py', arcname='autodl-tmp/flow_extension_20261109/flow_ext_train.py')
    tar.add(OUTDIR / 'original_initial_states.json', arcname='autodl-tmp/flow_extension_20261109/original_initial_states.json')
summary = dict(files=len(members), initial_states=len(evidence), tar=str(tar_path), tar_gb=round(tar_path.stat().st_size / 1e9, 3), tar_sha256=sha(tar_path))
write(OUTDIR / 'bundle_summary.json', summary)
print(json.dumps(summary, indent=1))
