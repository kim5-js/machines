"""Evaluation only. Frozen checkpoints and datasets are read-only inputs."""
from pathlib import Path
import argparse, contextlib, gzip, hashlib, importlib, importlib.metadata, io, json
import math, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
FLOW = Path('/root/autodl-tmp/flow_source_retrain_20260920/release_v1')
MATCHED = Path('/root/autodl-tmp/refsoft_reproduction_20260910/release_v1')
PYTHON = '/root/autodl-tmp/envs/mdc-yolo26/bin/python'

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def request(): return read(ROOT/'request.json')
def bootstrap(family='standard'):
    sys.dont_write_bytecode=True
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'settings'/family)
    os.environ['MPLCONFIGDIR']=str(ROOT/'mplconfig')
    paths=[str(FLOW/'vendor'),str(FLOW/'historical_code'),'/root/autodl-tmp/ring3_setup/autodl']
    if family=='matched': paths.insert(0,str(MATCHED/'code/vendor'))
    sys.path[:0]=paths
    require(importlib.metadata.version('pycocotools')=='2.0.10','Wrong COCO API')

def score(gt_dict, predictions, out=None):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=gt_dict;gt.createIndex()
        if predictions: dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={'images':gt_dict['images'],'categories':gt_dict['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=sorted(gt.imgs);ev.params.catIds=sorted(gt.cats)
        ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    names=['mAP50-95','mAP50','mAP75','APsmall','APmedium','APlarge','AR1','AR10','AR100','ARsmall','ARmedium','ARlarge']
    metrics={k:None if v<0 else float(v) for k,v in zip(names,ev.stats)}
    per_class={}
    for k,cat in enumerate(ev.params.catIds):
        v=ev.eval['precision'][:,:,k,0,-1];v=v[v>=0]
        per_class[gt.cats[cat]['name']]=float(v.mean()) if len(v) else None
    if out:
        (out/'coco_eval.log').write_text(log.getvalue())
        np.savez_compressed(out/'coco_arrays.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
    return metrics,per_class

def self_test():
    gt={'info':{},'images':[{'id':1,'width':200,'height':200}],
        'categories':[{'id':1,'name':'target'},{'id':2,'name':'absent'}],
        'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[5,5,20,20],'area':400,'iscrowd':0}]}
    perfect,classes=score(gt,[{'image_id':1,'category_id':1,'bbox':[5,5,20,20],'score':0.9}])
    empty,_=score(gt,[])
    wrong,_=score(gt,[{'image_id':1,'category_id':2,'bbox':[5,5,20,20],'score':0.9}])
    require(perfect['mAP50-95']>0.99999 and perfect['APsmall']>0.99999,'Perfect-match failure')
    require(perfect['APmedium'] is None and classes['absent'] is None,'Missing-area/class failure')
    require(empty['mAP50-95']==0 and wrong['mAP50-95']==0,'Empty/wrong-class failure')
    return {'status':'passed','checks':['perfect detection','empty detections','wrong category','absent class','absent area']}

def labels_for(image):
    parts=list(image.parts);idx=max(i for i,p in enumerate(parts) if p=='images');parts[idx]='labels'
    return Path(*parts).with_suffix('.txt')

def prepare():
    bootstrap()
    import yaml
    from PIL import Image
    req=request()
    require(not (ROOT/'prepared.json').exists(),'Prepared evidence already exists')
    require(sha(req['transform_source'])==req['transform_source_sha256'],'Transform code drift')
    require(sha(req['mdg_config'])==req['mdg_config_sha256'],'MDG parameter drift')
    checkpoints={}
    for j in req['jobs']:
        if j['experiment_id'] in checkpoints: continue
        p=Path(j['checkpoint'])
        if not p.exists() or sha(p)!=j['checkpoint_sha256']:
            candidates=[]
            if j['family']=='matched':
                candidates=list((MATCHED/'runs_recovery_v1').glob(p.parent.parent.name+'/weights/best.pt'))
            candidates=[x for x in candidates if sha(x)==j['checkpoint_sha256']]
            require(len(candidates)==1,'Cannot resolve frozen checkpoint '+j['key'])
            p=candidates[0]
        checkpoints[j['experiment_id']]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    datasets={}
    for name,spec in req['dataset_specs'].items():
        gt={'info':{},'images':[],'categories':[],'annotations':[]}
        records=[]
        if name=='FloW':
            original=[json.loads(line) for line in Path(spec['records']).read_text().splitlines()]
            names={0:'bottle'}
            images=[Path(r['image']) for r in original]
        else:
            config=yaml.safe_load(Path(spec['yaml']).read_text())
            names={int(k):v for k,v in config['names'].items()}
            listing=Path(config['path'])/config[spec['split']]
            require(listing.is_dir(),'Expected image directory')
            images=sorted(p for p in listing.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp'])
        require(len(images)==spec['images'] and len(names)==spec['classes'],'Dataset count mismatch '+name)
        require(len(set(map(str,images)))==len(images),'Duplicate images')
        gt['categories']=[{'id':k+1,'name':names[k]} for k in sorted(names)]
        for ident,p in enumerate(images,1):
            label=labels_for(p);require(label.is_file(),'Missing label '+str(label))
            with Image.open(p) as im: width,height=im.size
            record={'id':ident,'image':str(p),'label':str(label),'width':width,'height':height,
                    'image_sha256':sha(p),'label_sha256':sha(label)}
            records.append(record)
            gt['images'].append({'id':ident,'file_name':str(p),'width':width,'height':height})
            for line in label.read_text().splitlines():
                if not line.strip():continue
                f=list(map(float,line.split()));require(len(f)==5,'Malformed label')
                cl,xc,yc,w,h=f
                require(cl==int(cl) and int(cl) in names and all(math.isfinite(x) for x in f) and w>0 and h>0,'Invalid label')
                box=[(xc-w/2)*width,(yc-h/2)*height,w*width,h*height]
                gt['annotations'].append({'id':len(gt['annotations'])+1,'image_id':ident,'category_id':int(cl)+1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
        if name=='FloW':
            require(all(r['image_sha256']==o['image_sha256'] and r['label_sha256']==o['label_sha256'] for r,o in zip(records,original)),'FloW inputs changed')
            archived=read(spec['ground_truth'])
            require(len(gt['annotations'])==len(archived['annotations']),'FloW annotation count changed')
            require(all(a['image_id']==b['image_id'] and a['category_id']==b['category_id'] and
                        max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))<1e-5 and abs(a['area']-b['area'])<0.01
                        for a,b in zip(gt['annotations'],archived['annotations'])),'FloW labels differ from authoritative pixel boxes')
            require(all(a['id']==b['id'] and a['width']==b['width'] and a['height']==b['height'] and Path(a['file_name']).name==Path(b['file_name']).name
                        for a,b in zip(gt['images'],archived['images'])),'FloW image identity mismatch')
            # The original COCO GT retains exact source pixel boxes; YOLO decimal serialization has sub-micropixel error.
            gt=archived
        dest=ROOT/'data'/name;dest.mkdir(parents=True,exist_ok=True)
        for filename,value in [('records.json',records),('ground_truth.json',gt)]:
            if (dest/filename).exists():require(read(dest/filename)==value,'Prepared data changed '+str(dest/filename))
            else:write(dest/filename,value)
        datasets[name]={'images':len(images),'annotations':len(gt['annotations']),'records_sha256':sha(dest/'records.json'),
                        'ground_truth_sha256':sha(dest/'ground_truth.json'),'names':names,'split':spec['split']}
    bootstrap_test=self_test()
    prepared={'status':'prepared','utc':now(),'request_sha256':sha(ROOT/'request.json'),
              'code_sha256':sha(__file__),'checkpoints':checkpoints,'datasets':datasets,'self_test':bootstrap_test}
    write(ROOT/'prepared.json',prepared)
    print(json.dumps({'status':'prepared','checkpoints':len(checkpoints),'datasets':datasets,'self_test':bootstrap_test}))

def transforms(req,dataset):
    import evaluate_pretrained_sgd_robustness as robust
    require(sha(robust.__file__)==req['transform_source_sha256'],'Wrong imported transform')
    conditions,params=robust.build_conditions({'controlled_degradation':{
        'source_config':req['mdg_config'],'severities':['strong'],
        'random_seed_namespace':req['condition_namespace_root']+'/'+dataset}})
    return conditions,params

def evaluate(key,smoke=False):
    req=request();job=next(j for j in req['jobs'] if j['key']==key)
    bootstrap(job['family'])
    import cv2,numpy as np,torch,ultralytics
    from ultralytics import YOLO,RTDETR
    import pycocotools.cocoeval as coco_module
    prepared=read(ROOT/'prepared.json')
    require(prepared['request_sha256']==sha(ROOT/'request.json') and prepared['code_sha256']==sha(__file__),'Frozen evaluation code/request changed')
    ck=prepared['checkpoints'][job['experiment_id']]
    require(sha(ck['path'])==job['checkpoint_sha256'],'Checkpoint changed')
    data=ROOT/'data'/job['dataset']
    require(sha(data/'records.json')==prepared['datasets'][job['dataset']]['records_sha256'],'Image manifest changed')
    require(sha(data/'ground_truth.json')==prepared['datasets'][job['dataset']]['ground_truth_sha256'],'Ground truth changed')
    records=read(data/'records.json')
    if smoke: records=records[:4]
    out=ROOT/('smoke' if smoke else 'evaluations')/key
    require(not out.exists(),'Evaluation output exists '+key);out.mkdir(parents=True)
    model=(RTDETR if job['arm']=='rtdetr_l' else YOLO)(ck['path'])
    names={int(k):v for k,v in prepared['datasets'][job['dataset']]['names'].items()}
    require(model.names==names,'Model classes differ '+repr(model.names))
    torch.manual_seed(0);np.random.seed(0);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    probe={'forward_calls':0,'maximum_forward_batch':0,'input_dtypes':[],'input_shapes':[]}
    def before_forward(module,args):
        tensor=args[0]
        require(tensor.ndim==4 and tensor.shape[0]==1 and tensor.dtype==torch.float32,'Batch/precision mismatch')
        require(tuple(tensor.shape[2:])==(960,960),'Non-square preprocessing')
        probe['forward_calls']+=1;probe['maximum_forward_batch']=max(probe['maximum_forward_batch'],int(tensor.shape[0]))
        for field,value in [('input_dtypes',str(tensor.dtype)),('input_shapes',list(tensor.shape))]:
            if value not in probe[field]:probe[field].append(value)
    def on_start(predictor):
        require(predictor.dataset.bs==1 and not predictor.model.fp16 and predictor.args.rect is False,'Predictor contract mismatch')
        if not getattr(predictor,'_unified_guard',False):
            predictor.model.register_forward_pre_hook(before_forward);predictor._unified_guard=True
    model.add_callback('on_predict_start',on_start)
    conditions,parameters=transforms(req,job['dataset'])
    transform=conditions[job['condition']]
    input_hasher=hashlib.sha256()
    def load_input(rec):
        require(sha(rec['image'])==rec['image_sha256'] and sha(rec['label'])==rec['label_sha256'],'Dataset bytes changed')
        im=cv2.imread(rec['image'],cv2.IMREAD_COLOR)
        require(im is not None and im.shape[:2]==(rec['height'],rec['width']),'Image decode mismatch')
        return transform(im,Path(rec['image']))
    started=time.monotonic();predictions=[];coverage=[]
    # Bounded ordered prefetch preserves exactly one image per model call.
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={i:pool.submit(load_input,records[i]) for i in range(min(4,len(records)))}
        for index,rec in enumerate(records):
            arr=pending.pop(index).result()
            nxt=index+4
            if nxt<len(records):pending[nxt]=pool.submit(load_input,records[nxt])
            input_hasher.update(str(rec['id']).encode());input_hasher.update(arr.tobytes())
            results=list(model.predict(source=[arr],stream=True,verbose=False,save=False,**req['inference']))
            require(len(results)==1,'Wrong result count')
            result=results[0];require(tuple(result.orig_shape)==(rec['height'],rec['width']),'Output coordinate mismatch')
            boxes=result.boxes.data.detach().cpu().numpy()
            require(boxes.ndim==2 and boxes.shape[1]==6 and np.isfinite(boxes).all(),'Nonfinite predictions')
            for x1,y1,x2,y2,confidence,cl in boxes.tolist():
                require(int(cl)==cl and int(cl) in names and 0<=confidence<=1 and x2>=x1 and y2>=y1,'Invalid detection')
                predictions.append({'image_id':rec['id'],'category_id':int(cl)+1,'bbox':[x1,y1,x2-x1,y2-y1],'score':confidence})
            coverage.append(rec['id'])
            if (index+1)%100==0 or index+1==len(records):
                write(out/'progress.json',{'status':'inference','key':key,'images':index+1,'total':len(records),'elapsed_seconds':time.monotonic()-started,'utc':now()})
    require(coverage==[r['id'] for r in records] and probe['forward_calls']==len(records),'Coverage/batch probe failure')
    with gzip.open(out/'predictions.json.gz','wt',encoding='utf-8') as f:json.dump(predictions,f,allow_nan=False)
    gt=read(data/'ground_truth.json')
    if smoke:
        gt['images']=[r for r in gt['images'] if r['id'] in coverage]
        gt['annotations']=[r for r in gt['annotations'] if r['image_id'] in coverage]
    metrics,per_class=score(gt,predictions,out)
    require(all(v is None or math.isfinite(v) and 0<=v<=1 for v in metrics.values()),'Invalid metrics')
    require(sha(ck['path'])==job['checkpoint_sha256'],'Checkpoint changed during evaluation')
    report={'status':'smoke_completed' if smoke else 'completed','utc':now(),'job':job,'checkpoint':ck,
            'request_sha256':prepared['request_sha256'],'code_sha256':prepared['code_sha256'],
            'ground_truth_sha256':prepared['datasets'][job['dataset']]['ground_truth_sha256'],
            'image_manifest_sha256':prepared['datasets'][job['dataset']]['records_sha256'],
            'input_pixels_sha256':input_hasher.hexdigest(),'images':len(records),'ground_truth_boxes':len(gt['annotations']),
            'predictions':len(predictions),'predictions_sha256':sha(out/'predictions.json.gz'),
            'arrays_sha256':sha(out/'coco_arrays.npz'),'metrics':metrics,'per_class_AP':per_class,
            'inference':req['inference'],'actual_forward_probe':probe,'evaluator':req['evaluator'],
            'evaluator_code_sha256':sha(coco_module.__file__),'condition_parameters':parameters[job['condition']],
            'runtime':{'ultralytics':ultralytics.__version__,'ultralytics_file':ultralytics.__file__,
                       'torch':torch.__version__,'pycocotools':importlib.metadata.version('pycocotools'),
                       'gpu':torch.cuda.get_device_name(0)},'elapsed_seconds':time.monotonic()-started}
    write(out/'evaluation.json',report)
    print(json.dumps({'key':key,'status':report['status'],'metrics':metrics,'elapsed_seconds':report['elapsed_seconds']}),flush=True)

def smoke_all():
    req=request()
    keys=[]
    for family,arm,cond in [('matched','C_CBAM_Focal_EIoU','normal'),('detector','rtdetr_l','normal'),
                            ('detector','yolo26s','normal'),('strategy','E2','normal'),
                            ('strategy','E1','low_light_strong'),('strategy','E0','glare_strong')]:
        key=next(j['key'] for j in req['jobs'] if j['family']==family and j['arm']==arm and j['condition']==cond)
        subprocess.run([PYTHON,'-B','-u',str(__file__),'evaluate','--key',key,'--smoke'],check=True)
        keys.append(key)
    write(ROOT/'smoke_gate.json',{'status':'passed','keys':keys,'utc':now(),'code_sha256':sha(__file__)})

def run_all():
    req=request();require(read(ROOT/'smoke_gate.json')['status']=='passed','Smoke gate missing')
    require(read(ROOT/'smoke_gate.json')['code_sha256']==sha(__file__),'Smoke code mismatch')
    require(not (ROOT/'run_started.json').exists(),'Evaluation queue already started')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
    require(int(gpu.splitlines()[0])<3000,'GPU is occupied by another job')
    write(ROOT/'run_started.json',{'utc':now(),'pid':os.getpid(),'evaluations':len(req['jobs'])})
    (ROOT/'logs').mkdir(exist_ok=True)
    # Score the changed strategy/condition evidence first, then attribution and reference checkpoints.
    jobs=sorted(req['jobs'],key=lambda j:({'strategy':0,'matched':1,'detector':2}[j['family']],j['dataset'],j['experiment_id'],j['condition']))
    for index,j in enumerate(jobs):
        write(ROOT/'status.json',{'status':'running','completed':index,'total':len(jobs),'key':j['key'],'utc':now()})
        with (ROOT/'logs'/(j['key']+'.log')).open('w') as log:
            proc=subprocess.run([PYTHON,'-B','-u',str(__file__),'evaluate','--key',j['key']],stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            write(ROOT/'status.json',{'status':'failed','completed':index,'key':j['key'],'returncode':proc.returncode,'utc':now()})
            raise RuntimeError('Evaluation failed '+j['key'])
    write(ROOT/'status.json',{'status':'completed','completed':len(jobs),'total':len(jobs),'utc':now()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate','smoke','run']);p.add_argument('--key');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='evaluate':evaluate(a.key,a.smoke)
        elif a.mode=='smoke':smoke_all()
        else:run_all()
    except BaseException:
        print(traceback.format_exc(),flush=True);raise
