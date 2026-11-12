"""Read-only compact progress and completion snapshot."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'adapter'))
import common as c
req=c.request();adapter=c.verify_release(req)
state=c.read(ROOT/'state.json') if (ROOT/'state.json').exists() else {}
out={'utc':c.now(),'root':str(ROOT),'request_sha256':c.REQUEST_SHA,'adapter_manifest_sha256':adapter,'preflight_sha256':c.sha(ROOT/'preflight.json'),'state':state,'completed_training':[],'completed_evaluations':[],'failures':[]}
for p in sorted((ROOT/'runs').glob('*/completion.json')):
    r=c.read(p)
    out['completed_training'].append({'job':r['job']['key'],'epochs':r['epochs'],'checkpoint_selection':r['checkpoint_selection'],'completion_sha256':c.sha(p)})
for p in sorted((ROOT/'evaluations').glob('*/*/evaluation.json')):
    r=c.read(p);out['completed_evaluations'].append({'job':r['job']['key'],'split':r['evaluation_split'],'official_metrics':r['official_coco']['metrics'],'evaluation_sha256':c.sha(p)})
for p in sorted((ROOT/'failures').glob('*.json')):out['failures'].append({'file':p.name,'details':c.read(p)})
if state.get('run_directory'):
    run=Path(state['run_directory'])
    for name in ['progress.json','runtime_evidence.json']:
        p=run/name
        if p.exists():
            r=c.read(p)
            if name=='runtime_evidence.json':r={k:r[k] for k in ['runtime','initialization','loaded_data','actual_args','finite_gradient_probe','validated_historical_runtime','checkpoint_selection'] if k in r}
            out[name]=r
for key in ['supervisor_pid','child_pid']:
    if state.get(key):
        try:out[key]=c.process_record(state[key])
        except FileNotFoundError:out[key]=None
if state.get('log') and Path(state['log']).exists():out['log_tail']=Path(state['log']).read_bytes()[-2800:].decode(errors='replace')
out['resources']=c.resource_snapshot(req)
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
