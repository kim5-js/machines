"""Translate the fixed local job lists to AutoDL paths. Job keys, conditions and severities are unchanged."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import csv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rows = {r['experiment_id']: r for r in csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig'))}
pred_sha = {r['evaluation_key']: r['predictions_sha256'] for r in csv.DictReader(open(UNIFIED_CSV, encoding='utf-8-sig'))}
dest = OUT / 'remote'
dest.mkdir(exist_ok=True)


def remote(job):
    best = rows[job['experiment_id']]['checkpoint']
    path = best if job['weights'].endswith('_best.pt') else best.rsplit('/', 1)[0] + '/last.pt'
    return {**job, 'weights': path}


data = read(OUT / 'jobs.json')
data['jobs'] = [remote(j) for j in data['jobs']]
write(dest / 'remote_jobs.json', data)
gate = []
for g in read(OUT / 'gate_jobs.json'):
    g = remote(g); g['sealed_predictions_sha256'] = pred_sha[g['sealed_key']]; gate.append(g)
write(dest / 'remote_gate_jobs.json', dict(severity=data['severity'], glare_geometry=data['glare_geometry'], jobs=gate))
print(len(data['jobs']), 'jobs;', len(gate), 'gate jobs;', sorted({j['weights'].rsplit('/', 1)[1] for j in data['jobs']}))
