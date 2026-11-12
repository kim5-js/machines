"""Dispatch the prepared queue once; never interrupt another experiment."""
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'adapter'))
import common as c

with (ROOT/'dispatch.lock').open('a+') as lock:
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'launch_record.json').exists():
        print((ROOT/'launch_record.json').read_text(),flush=True)
        raise SystemExit(0)
    req=c.request();c.verify_release(req)
    c.require(not (ROOT/'state.json').exists(),'Existing queue state requires review')
    command=[sys.executable,'-B','-u',str(ROOT/'adapter/supervise.py')]
    with (ROOT/'supervisor.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,RING3_LEGACY_RELEASE_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'status':'dispatched','utc':c.now(),'pid':child.pid,'process':c.process_record(child.pid),'command':command,'request_sha256':c.REQUEST_SHA,'order':'four baseline trainings and evaluations first; then nine ablation trainings and evaluations','other_jobs_modified':False}
    c.write(ROOT/'launch_record.json',record)
    time.sleep(2)
    c.require(child.poll() is None,'Supervisor exited; retained log must be inspected')
    print((ROOT/'launch_record.json').read_text(),flush=True)
