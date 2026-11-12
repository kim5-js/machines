"""Run jobs one at a time in fresh processes. Resumable: completed evaluations are skipped."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import argparse, subprocess, time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('--jobs', required=True); p.add_argument('--block', action='append'); p.add_argument('--status', default='status.json')
a = p.parse_args()
data = read(a.jobs)
jobs = data['jobs'] if isinstance(data, dict) else data
if a.block:
    jobs = [j for j in jobs if j['block'] in a.block]
logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
status = OUT / a.status
done = failed = 0
started = time.monotonic()
for i, j in enumerate(jobs):
    if (OUT / 'evaluations' / j['key'] / 'evaluation.json').exists():
        done += 1
        continue
    write(status, dict(state='running', position=i, total=len(jobs), done=done, failed=failed, key=j['key'], elapsed_hours=round((time.monotonic() - started) / 3600, 3), utc=datetime.now(timezone.utc).isoformat()))
    for attempt in (1, 2):
        with (logs / (j['key'] + '.log')).open('a', encoding='utf-8') as log:
            rc = subprocess.run([sys.executable, '-I', '-B', '-u', str(HERE / 'evaluate_local.py'), '--jobs', a.jobs, '--key', j['key']], stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            done += 1
            break
        time.sleep(20)
    else:
        failed += 1
        if failed >= 3:
            write(status, dict(state='stopped_after_failures', done=done, failed=failed, key=j['key'], utc=datetime.now(timezone.utc).isoformat()))
            sys.exit(1)
write(status, dict(state='finished', total=len(jobs), done=done, failed=failed, elapsed_hours=round((time.monotonic() - started) / 3600, 3), utc=datetime.now(timezone.utc).isoformat()))
print(json.dumps(dict(total=len(jobs), done=done, failed=failed)))
