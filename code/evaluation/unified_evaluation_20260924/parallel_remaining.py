"""Drain this task's serial runner, verify repeats, and score remaining jobs.

The frozen evaluator file and request are never edited. Only this task's exact
runner process is stopped; any active evaluator is allowed to finish normally.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime,gzip,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path('/root/autodl-tmp/manuscript_unified_eval_20260924')
PY='/root/autodl-tmp/envs/mdc-yolo26/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,data):
    temp=p.with_suffix(p.suffix+'.parallel_tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)
def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    except FileNotFoundError:return []
def phase(name,**kw):write(ROOT/'parallel_progress.json',{'phase':name,'utc':now(),**kw})
def completed():return {p.parent.name:read(p) for p in (ROOT/'evaluations').glob('*/evaluation.json')}
def worker(j,workroot=ROOT):
    logdir=workroot/'logs';logdir.mkdir(exist_ok=True)
    with (logdir/(j['key']+'.log')).open('w') as log:
        p=subprocess.run([PY,'-B','-u',str(workroot/'unified_eval.py'),'evaluate','--key',j['key']],stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,(j['key'],p.returncode)
    r=read(workroot/'evaluations'/j['key']/'evaluation.json');assert r['status']=='completed'
    return r

def main():
    gate=ROOT/'parallel_handoff.json';assert not gate.exists(),'Parallel handoff already exists'
    req=read(ROOT/'request.json');prepared=read(ROOT/'prepared.json');started=read(ROOT/'run_started.json')
    pid=started['pid'];cmd=command(pid)
    assert str(ROOT/'unified_eval.py') in cmd and cmd[-1]=='run',(pid,cmd)
    assert read(ROOT/'status.json')['status']=='running'
    os.kill(pid,signal.SIGSTOP)
    info={'utc':now(),'serial_pid':pid,'serial_command':cmd,'frozen_code_sha256':prepared['code_sha256'],
          'request_sha256':prepared['request_sha256'],'reason':'GPU mostly idle between preprocessing and scoring; independent checkpoint processes, no protocol change.'}
    try:
        time.sleep(.2)
        children_file=Path(f'/proc/{pid}/task/{pid}/children')
        children=[int(x) for x in children_file.read_text().split()]
        info['drained_children']=children
        phase('draining_current_evaluation',children=children,completed=len(completed()),total=78)
        for child in children:
            cc=command(child)
            if not cc:
                state=Path(f'/proc/{child}/stat').read_text().split(') ',1)[1].split()[0]
                assert state=='Z' and read(ROOT/'status.json')['key'] in completed()
                continue
            assert str(ROOT/'unified_eval.py') in cc and 'evaluate' in cc,(child,cc)
            key=cc[cc.index('--key')+1]
            deadline=time.monotonic()+360
            while time.monotonic()<deadline:
                record=ROOT/'evaluations'/key/'evaluation.json'
                stat=Path(f'/proc/{child}/stat')
                state=stat.read_text().split(') ',1)[1].split()[0] if stat.exists() else 'gone'
                if record.exists() and read(record)['status']=='completed' and state in ['Z','gone']:break
                if state in ['Z','gone']:raise RuntimeError('Current evaluation failed '+key)
                time.sleep(1)
            else:raise TimeoutError('Timed out waiting for current evaluation '+key)
        done=completed()
        for folder in (ROOT/'evaluations').iterdir():assert folder.name in done,('Partial evaluation',folder)
        info['serial_completed']=sorted(done);info['serial_status']=read(ROOT/'status.json')
        write(gate,info)
        # Parent has no live evaluation left. Terminate only this exact task runner.
        assert command(pid)==cmd
        os.kill(pid,signal.SIGTERM)
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    except BaseException:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        raise
    phase('checking_parallel_repeats',completed=len(done),total=78)
    repeats=[]
    for dataset in ['FloW','WSODD']:
        ref=next(r for r in done.values() if r['job']['family']=='strategy' and r['job']['dataset']==dataset
                 and r['job']['arm']=='E0' and r['job']['seed']==0 and r['job']['condition']=='normal')
        aux=ROOT/'parallel_repeats'/dataset;aux.mkdir(parents=True)
        for name in ['unified_eval.py','request.json','prepared.json']:shutil.copy2(ROOT/name,aux/name)
        (aux/'data').symlink_to(ROOT/'data',target_is_directory=True)
        repeats.append((ref,aux))
    comparisons=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(worker,ref['job'],aux):(ref,aux) for ref,aux in repeats}
        for f in as_completed(futures):
            ref,aux=futures[f];repeat=f.result()
            assert repeat['metrics']==ref['metrics'],('Parallel metric mismatch',ref['job']['key'])
            assert repeat['per_class_AP']==ref['per_class_AP']
            assert repeat['input_pixels_sha256']==ref['input_pixels_sha256']
            with (gzip.open(aux/'evaluations'/ref['job']['key']/'predictions.json.gz','rb') as a,
                  gzip.open(ROOT/'evaluations'/ref['job']['key']/'predictions.json.gz','rb') as b):
                assert a.read()==b.read(),('Parallel predictions mismatch',ref['job']['key'])
            comparisons.append({'key':ref['job']['key'],'all_metrics_exact':True,'decoded_prediction_bytes_exact':True,
                                'input_pixels_exact':True,'repeated_record':str(aux/'evaluations'/ref['job']['key']/'evaluation.json')})
    write(ROOT/'parallel_repeat_gate.json',{'utc':now(),'status':'passed','workers':2,'checks':comparisons,
        'note':'Reproducibility checks are outside the 78 manuscript evaluation records.'})
    jobs=sorted(req['jobs'],key=lambda j:({'strategy':0,'matched':1,'detector':2}[j['family']],j['dataset'],j['experiment_id'],j['condition']))
    remaining=[j for j in jobs if j['key'] not in done]
    write(ROOT/'status.json',{'status':'running','completed':len(done),'total':78,'workers':2,'utc':now()})
    phase('parallel_evaluation',remaining=len(remaining),completed=len(done),total=78)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(worker,j):j for j in remaining}
        for f in as_completed(futures):
            r=f.result();done[r['job']['key']]=r
            write(ROOT/'status.json',{'status':'running','completed':len(done),'total':78,'workers':2,
                  'last_completed_key':r['job']['key'],'utc':now()})
    assert len(done)==78
    write(ROOT/'status.json',{'status':'completed','completed':78,'total':78,'workers':2,'utc':now()})
    phase('completed',completed=78,total=78)

if __name__=='__main__':
    try:main()
    except BaseException:
        phase('failed',traceback=traceback.format_exc());print(traceback.format_exc(),flush=True);raise
