"""Priority training queue on the Vast instance: several jobs per GPU, one process per job.

Usage: python train_queue_vast.py --queue queue.json --per-gpu 3
queue.json: {"jobs": [{"key", "cmd": [...], "cwd", "done": path, "rundir": path}, ...]} in priority order.
A job is complete when its "done" file exists. A failed job's run directory is moved to failed_runs/
with a timestamp and retried once; nothing is overwritten. New jobs appended to queue.json are picked up.
Status: <queue>.status.json next to the queue file.
"""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


p = argparse.ArgumentParser(); p.add_argument('--queue', required=True); p.add_argument('--per-gpu', type=int, default=3); p.add_argument('--gpus', type=int, default=2)
a = p.parse_args()
qpath = Path(a.queue).resolve(); status = qpath.with_suffix('.status.json'); logs = qpath.parent / 'queue_logs'; logs.mkdir(exist_ok=True)
slots = [g for g in range(a.gpus) for _ in range(a.per_gpu)]
running, attempts, failed, started = {}, {}, {}, time.monotonic()
while True:
    jobs = json.loads(qpath.read_text())['jobs']
    if (qpath.parent / 'STOP').exists() and not running:
        break
    todo = [j for j in jobs if not Path(j['done']).exists() and j['key'] not in running and failed.get(j['key'], 0) < 2]
    free = list(slots)
    for v in running.values(): free.remove(v[2])
    while todo and free and not (qpath.parent / 'STOP').exists():
        j = todo.pop(0); gpu = free.pop(0)
        rundir = Path(j['rundir'])
        if rundir.exists():
            dest = qpath.parent / 'failed_runs' / f"{j['key']}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            dest.parent.mkdir(exist_ok=True); shutil.move(str(rundir), str(dest))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        log = (logs / (j['key'] + '.log')).open('a')
        log.write(f"\n=== start {now()} gpu={gpu} attempt={failed.get(j['key'], 0) + 1}\n"); log.flush()
        running[j['key']] = (subprocess.Popen(j['cmd'], cwd=j.get('cwd'), stdout=log, stderr=subprocess.STDOUT, env=env), log, gpu, time.monotonic())
    for key in list(running):
        proc, log, gpu, t0 = running[key]
        if proc.poll() is None: continue
        log.close(); del running[key]
        job = next(j for j in jobs if j['key'] == key)
        if proc.returncode != 0 or not Path(job['done']).exists():
            failed[key] = failed.get(key, 0) + 1
    done = [j['key'] for j in jobs if Path(j['done']).exists()]
    write(status, dict(utc=now(), total=len(jobs), done=len(done), running={k: dict(gpu=v[2], hours=round((time.monotonic() - v[3]) / 3600, 2)) for k, v in running.items()},
                       failed={k: v for k, v in failed.items()}, elapsed_hours=round((time.monotonic() - started) / 3600, 2)))
    if not running and not [j for j in jobs if not Path(j['done']).exists() and failed.get(j['key'], 0) < 2]:
        break
    time.sleep(30)
write(status, dict(json.loads(status.read_text()), state='finished', utc=now()))
