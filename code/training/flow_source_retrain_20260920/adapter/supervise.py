"""Serial queue: four complete baseline experiments, then nine ablations."""
from pathlib import Path
import fcntl
import os
import signal
import sys
import traceback
import common as c
import training_helpers as h


def train_if_needed(req,key,phase,adapter):
    job=c.resolve_job(req,key,phase);run=Path(job['run_dir'])
    if (run/'completion.json').exists():
        c.verify_completion(req,job,adapter)
    else:
        c.require(not run.exists(),'Incomplete run retained; inspect before recovery: '+str(run))
        c.wait_for_resources(req)
        h.run_child(job,'train_baseline' if job['family']=='baseline' else 'train')
        c.verify_completion(req,job,adapter)
    return job


def evaluate(req,job,adapter):
    c.verify_completion(req,job,adapter)
    outputs=[c.ROOT/'evaluations'/job['key']/split/'evaluation.json' for split in ['val','test']]
    if all(p.exists() for p in outputs):
        for p in outputs:
            result=c.read(p)
            c.require(result['status']=='completed' and result['request_sha256']==c.REQUEST_SHA,'Existing evaluation provenance mismatch')
        return
    c.require(not any(p.exists() for p in outputs),'Partial evaluation retained; inspect before recovery')
    c.wait_for_resources(req)
    h.run_child(job,'evaluate')


def main():
    req=c.request();adapter=c.verify_release(req)
    with (c.ROOT/'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.make_smoke_data(req,adapter)
        for model in req['baseline_order']:
            key=f'baseline_{model}_seed0'
            train_if_needed(req,key,'smoke',adapter)
            job=train_if_needed(req,key,'full',adapter)
            evaluate(req,job,adapter)
            import finalize
            finalize.summarize(req,complete=False)
        c.write(c.ROOT/'baseline_gate.json',{'status':'passed','utc':c.now(),'request_sha256':c.REQUEST_SHA,'training_jobs':4,'evaluation_splits_per_job':['val','test'],'smoke_jobs':4})
        smoke=[]
        for arm in req['matrix']['arms']:
            job=train_if_needed(req,f'FloW_{arm}_seed0','smoke',adapter)
            smoke.append({'job':job})
        h.write_smoke_gate(req,adapter,smoke);h.verify_smoke_gate(req,adapter)
        for seed in req['matrix']['seeds']:
            for arm in req['matrix']['arms']:
                job=train_if_needed(req,f'FloW_{arm}_seed{seed}','full',adapter)
                evaluate(req,job,adapter)
                finalize.summarize(req,complete=False)
        finalize.summarize(req,complete=True)
        c.write(c.ROOT/'state.json',{'status':'completed','updated_utc':c.now(),'supervisor_pid':os.getpid(),'formal_training_jobs':13,'smoke_training_jobs':7,'completed_val_test_evaluations':26,'ablation_historical_secondary_evaluations':18,'request_sha256':c.REQUEST_SHA,'other_jobs_modified':False})


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    try:main()
    except BaseException:
        c.write(c.ROOT/'failures/supervisor.json',{'utc':c.now(),'traceback':traceback.format_exc()})
        c.write(c.ROOT/'state.json',{'status':'failed','updated_utc':c.now(),'supervisor_pid':os.getpid(),'traceback':traceback.format_exc(),'recovery':'Retain all evidence; no automatic scientific recipe changes or overwrites'})
        raise
