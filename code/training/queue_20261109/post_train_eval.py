"""Runs on Vast after the training queue finishes: evaluate every completed new checkpoint with the unified
evaluator of recompute_20261109 (same COCOeval, inference contract and transforms as the sealed 20260924
evaluation), on the WSODD or FloW test set, under normal, the existing low-light and glare conditions and the
two strong low-light levels. Then pack the results for download. Training is not touched.
Usage: python post_train_eval.py [--dry-run]
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

TR = Path('/workspace/train_20261109')
BUNDLE = Path('/workspace/recompute_20261109/bundle')
PY = '/workspace/recompute_20261109/venv/bin/python'
A = Path('/root/autodl-tmp')
CONDITIONS = ['normal', 'low_light_strong', 'glare_strong', 'low_light_L4', 'low_light_L5']
SOURCES = [  # (family for the evaluator, dataset, run directory)
    ('matched', 'WSODD', A / 'refsoft_extension_20261109/release_v1/runs'),
    ('strategy', 'FloW', A / 'flow_source_retrain_20260920/release_v1/runs'),
    ('strategy', 'WSODD', A / 'pretrained_copies_20261109/runs'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


dry = '--dry-run' in sys.argv
log = (TR / 'post_train_eval.log').open('a')
def say(*a):
    print(*a, file=log, flush=True); print(*a, flush=True)


while not dry:
    s = json.loads((TR / 'queue.status.json').read_text())
    if s.get('state') == 'finished':
        break
    time.sleep(600)
say('queue finished' if not dry else 'dry run', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
base = json.loads((BUNDLE / 'jobs_vast.json').read_text())
jobs = []
for family, dataset, root in SOURCES:
    for done in sorted(root.glob('*/completion.json')):
        run = done.parent
        best = run / 'weights' / 'best.pt'
        if not best.exists():
            continue
        digest = sha(best)
        for cond in CONDITIONS:
            jobs.append(dict(key=f'NEW_{run.name}__{cond}', experiment_id=run.name, dataset=dataset, family=family, arm=run.name,
                             seed=run.name.rsplit('seed', 1)[-1], split='test', condition=cond, weights=str(best), weights_sha256=digest, block='post_train'))
say('evaluation jobs', len(jobs), 'checkpoints', len(jobs) // len(CONDITIONS))
cfg = {k: v for k, v in base.items() if k != 'jobs'}
(BUNDLE / 'jobs_post_train.json').write_text(json.dumps(dict(cfg, jobs=jobs), indent=1))
if dry:
    sys.exit(0)
subprocess.run([PY, '-B', str(BUNDLE / 'code/recompute_vast.py'), 'run', '--jobs', str(BUNDLE / 'jobs_post_train.json'), '--per-gpu', '8', '--threads', '4'],
               cwd=BUNDLE, stdout=log, stderr=subprocess.STDOUT)
keys = [f"output/evaluations/{j['key']}" for j in jobs if (BUNDLE / 'output/evaluations' / j['key'] / 'evaluation.json').exists()]
say('completed evaluations', len(keys))
# Training records (no weights) for every run, plus the evaluations
records = []
for _, _, root in SOURCES:
    for run in sorted(root.glob('*')):
        for name in ('args.yaml', 'results.csv', 'completion.json', 'runtime.json', 'runtime_evidence.json', 'e2c_evidence.json', 'epoch_evidence.jsonl', 'job.json'):
            if (run / name).exists():
                records.append(str(run / name))
(TR / 'post_train_records.txt').write_text('\n'.join(records))
subprocess.run(['tar', '-cf', str(TR / 'results_post_train.tar'), '-C', str(BUNDLE), *keys, 'output/status_jobs_post_train.json', 'jobs_post_train.json'], check=True)
subprocess.run(['tar', '-rf', str(TR / 'results_post_train.tar'), '-P', '-T', str(TR / 'post_train_records.txt'), str(TR / 'queue.status.json'), str(TR / 'queue.json')], check=True)
(TR / 'results_post_train.tar.sha256').write_text(sha(TR / 'results_post_train.tar') + '  results_post_train.tar\n')
say('POST_TRAIN_RESULTS_READY', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
