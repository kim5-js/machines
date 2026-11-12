"""48 extra jobs added after the 210-job results: matched (15) and FloW (9) checkpoints under the two
strongest low-light levels. The levels themselves were fixed in common.py before any result was seen;
extending them to these two blocks was decided after seeing the strategy results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
base = read(OUT / 'vast/jobs_vast.json')
src = [j for j in base['jobs'] if j['block'] in ('matched_condition', 'flow_condition') and j['condition'] == 'low_light_strong']
jobs = []
for j in src:
    for c in ('low_light_L4', 'low_light_L5'):
        jobs.append({**j, 'key': f"{j['experiment_id']}__{c}", 'condition': c, 'block': j['block'].replace('_condition', '_strong_low_light')})
require(len(jobs) == 48 and len({j['key'] for j in jobs}) == 48, 'Expected 48 jobs')
write(OUT / 'vast/jobs_extra.json', {k: v for k, v in base.items() if k != 'jobs'} | {'jobs': jobs})
print(len(jobs), sorted({j['block'] for j in jobs}))
