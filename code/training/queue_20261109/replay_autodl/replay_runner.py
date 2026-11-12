"""Replay runner (server side): start each job exactly as listed and stop it once three epochs are recorded.

The jobs are existing training jobs launched with their original commands and the full 150-epoch schedule;
only the first three rows of results.csv are used. Nothing here changes a recipe or an argument.
Usage: python replay_runner.py jobs.json [concurrency]
Status: /root/autodl-tmp/replay_20261111/status.json; logs in logs/<name>.log.
"""
import json, os, shutil, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp/replay_20261111')
EPOCHS = 3
LIMIT = 45 * 60  # seconds per job
UNSET = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')  # the September launches set none of these


def now():
    return datetime.now(timezone.utc).isoformat()


def rows(rundir):
    p = Path(rundir) / 'results.csv'
    if not p.exists():
        return 0
    return max(0, sum(1 for line in p.read_text().splitlines() if line.strip()) - 1)


def save(status):
    tmp = BASE / 'status.json.tmp'
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(BASE / 'status.json')


jobs = json.loads((BASE / sys.argv[1]).read_text())
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2
(BASE / 'logs').mkdir(exist_ok=True)
status = {j['name']: dict(state='waiting') for j in jobs}
todo, running = list(jobs), {}
while todo or running:
    while todo and len(running) < limit:
        j = todo.pop(0)
        if Path(j['rundir']).exists():
            status[j['name']] = dict(state='refused', reason='run directory already exists: ' + j['rundir']); save(status); continue
        env = {k: v for k, v in os.environ.items() if k not in UNSET}
        env.update(j.get('env', {}))
        log = (BASE / 'logs' / (j['name'] + '.log')).open('a')
        log.write(f"=== start {now()} cmd={json.dumps(j['cmd'])} extra_env={json.dumps(j.get('env', {}))}\n"); log.flush()
        proc = subprocess.Popen(j['cmd'], cwd=j['cwd'], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        running[j['name']] = (j, proc, log, time.monotonic())
        status[j['name']] = dict(state='running', started_utc=now(), pid=proc.pid); save(status)
    for name in list(running):
        j, proc, log, t0 = running[name]
        n = rows(j['rundir']); code = proc.poll(); elapsed = time.monotonic() - t0
        if code is None and n < EPOCHS and elapsed < LIMIT:
            status[name].update(rows=n, elapsed_seconds=round(elapsed)); continue
        if code is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=15); break
                except subprocess.TimeoutExpired:
                    pass
        log.write(f"=== end {now()} rows={n} returncode={proc.poll()}\n"); log.close(); del running[name]
        state = 'stopped_after_three_epochs' if n >= EPOCHS else ('exited' if code is not None else 'timeout')
        status[name] = dict(status[name], state=state, rows=n, elapsed_seconds=round(elapsed), returncode=code, ended_utc=now())
        if j.get('move_to') and Path(j['rundir']).exists():  # keep an original tree as it was: move the replay run out of it
            Path(j['move_to']).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(j['rundir'], j['move_to']); status[name]['moved_to'] = j['move_to']
    save(status)
    time.sleep(2)
status['finished_utc'] = now(); save(status)
