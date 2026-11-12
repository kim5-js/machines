"""Five mechanism-verified smokes, then 15 paired factor trainings and evaluations."""
from pathlib import Path
import fcntl,os,signal,sys,traceback
import common as c
import training_helpers as h
import finalize

def train(req,key,phase,adapter):
    job=c.resolve_job(req,key,phase);run=Path(job['run_dir'])
    if (run/'completion.json').exists():c.verify_completion(req,job,adapter)
    else:
        c.require(not run.exists(),'Incomplete run preserved; inspect before any recovery: '+str(run))
        c.wait_for_resources(req);h.run_child(job,'train');c.verify_completion(req,job,adapter)
    return job

def evaluate(req,job,adapter):
    c.verify_completion(req,job,adapter)
    outputs=[c.ROOT/'evaluations'/job['key']/s/'evaluation.json' for s in ['val','test']]
    if all(p.exists() for p in outputs):
        for p in outputs:
            r=c.read(p)
            c.require(r['status']=='completed' and r['request_sha256']==c.REQUEST_SHA,'Existing evaluation mismatch')
        return
    c.require(not any(p.parent.exists() for p in outputs),'Partial evaluation retained; inspect before recovery')
    c.wait_for_resources(req);h.run_child(job,'evaluate')
    c.require(all(p.exists() for p in outputs),'Evaluation did not complete both splits')

def main():
    req=c.request();adapter=c.verify_release(req)
    with (c.ROOT/'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.make_smoke_data(req,adapter)
        smoke=[{'job':train(req,f'FloW_{arm}_seed0','smoke',adapter)} for arm in req['matrix']['arms']]
        h.write_smoke_gate(req,adapter,smoke);h.verify_smoke_gate(req,adapter)
        for seed in req['matrix']['seeds']:
            for arm in req['matrix']['arms']:
                job=train(req,f'FloW_{arm}_seed{seed}','full',adapter)
                evaluate(req,job,adapter)
                finalize.summarize(req,complete=False)
        finalize.summarize(req,complete=True)
        c.write(c.ROOT/'state.json',{'status':'completed','updated_utc':c.now(),'supervisor_pid':os.getpid(),
            'formal_training_jobs':15,'smoke_training_jobs':5,'completed_val_test_evaluations':30,
            'historical_secondary_evaluations':30,'existing_anchors_reused':9,
            'request_sha256':c.REQUEST_SHA,'other_jobs_modified':False})

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    try:main()
    except BaseException:
        r={'utc':c.now(),'traceback':traceback.format_exc()}
        c.write(c.ROOT/'failures/supervisor.json',r)
        c.write(c.ROOT/'state.json',{'status':'failed','updated_utc':c.now(),'supervisor_pid':os.getpid(),
            **r,'recovery':'Preserve all evidence; no automatic scientific changes, overwrites or duplicate launch'})
        raise
