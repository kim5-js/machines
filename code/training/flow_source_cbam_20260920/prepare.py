"""Copy verified existing inputs into a new release; never touch prior experiments."""
from pathlib import Path
import concurrent.futures
import hashlib,json,shutil,sys

BUNDLE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    manifest=read(BUNDLE/'bundle_manifest.json')
    for e in manifest['files']:
        p=(BUNDLE/e['path']).resolve()
        assert p.is_relative_to(BUNDLE) and sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],e['path']
    req=read(BUNDLE/'request.json');assert sha(BUNDLE/'request.json')==manifest['request_sha256']
    ref=Path(req['source_reference']['root'])
    assert sha(ref/'request.json')==req['source_reference']['request_sha256']
    assert sha(ref/'preflight.json')==req['source_reference']['preflight_sha256']
    assert sha(ref/'state.json')==req['source_reference']['state_sha256'] and read(ref/'state.json')['status']=='completed'
    original=read(ref/'request.json')
    for k in ['training_reference_args','mdg_config','runtime','pretrained_weight']:
        assert req[k]==original[k],k
    assert req['datasets'][0]['splits']==original['datasets'][0]['splits']
    assert req['datasets'][0]['source_manifest_sha256']==original['datasets'][0]['source_manifest_sha256']
    old=Path(req['execution']['predecessor_root'])
    assert sha(old/'request.json')==req['execution']['predecessor_request_sha256']
    assert read(old/'state.json')['status']=='completed' and read(old/'summary.json')['completed_jobs']==27
    root=Path(req['execution']['remote_root']);assert not root.exists(),'Existing release preserved'
    root.mkdir(parents=True)
    for e in manifest['files']:
        dest=root/e['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BUNDLE/e['path'],dest)
    shutil.copyfile(BUNDLE/'bundle_manifest.json',root/'execution_bundle_manifest.json')
    code=[{'path':e['path'],'sha256':e['sha256']} for e in manifest['files'] if e['path'].startswith('adapter/') or e['path'] in ('prepare.py','launch.py','probe.py','collect.py')]
    write(root/'adapter_manifest.json',{'files':code,'bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json')})
    assets=[]
    def copy_existing(rel,dest_rel,expected,size=None):
        src=ref/rel;dest=root/dest_rel
        assert sha(src)==expected,rel
        if size is not None:assert src.stat().st_size==size
        if dest.exists():assert sha(dest)==expected
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        assert sha(dest)==expected
        assets.append({'path':dest_rel,'sha256':expected})
    for e in read(root/'staging_manifest.json')['files']:
        copy_existing(e['path'],e['path'],e['sha256'],e['bytes'])
    for e in req['source_runtime_assets']:
        copy_existing(e['path'],e['path'],e['sha256'])
    for rel,e in req['source_files'].items():copy_existing(rel,rel,e['sha256'],e['bytes'])
    for a in req['anchors']:
        assert sha(ref/'runs'/a['key']/'weights/best.pt')==a['checkpoint_sha256']
        for rel,e in a['files'].items():copy_existing(rel,'anchors/'+rel,e['sha256'],e['bytes'])
    records=read(root/'data_records.json');assert len(records)==2000
    for field in ['source_group','raw_folder','original_parent','original_filename_key']:
        groups={}
        for r in records:
            if r[field]:groups.setdefault(r[field],set()).add(r['split'])
        assert all(len(v)==1 for v in groups.values()),field
    details={}
    source_proof=read(ref/'preflight.json')
    for split in ['train','val','test']:
        p=root/'verified_data/FloW'/(split+'.jsonl')
        assert sha(p)==source_proof['datasets']['FloW'][split]['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        wanted=req['datasets'][0]['splits'][split]
        assert len(rows)==wanted['images'] and sum(r['boxes'] for r in rows)==wanted['boxes']
        def verify(r):
            assert sha(r['image'])==r['image_sha256'] and sha(r['label'])==r['label_sha256'],r['image']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,rows))
        details[split]={**wanted,'sha256':sha(p)}
        print(json.dumps({'verified_split':split,**wanted}),flush=True)
    sys.path.insert(0,str(root/'adapter'))
    import common as c
    write(root/'preflight.json',{'status':'passed','utc':c.now(),'request_sha256':c.REQUEST_SHA,
        'bundle_manifest_sha256':sha(root/'execution_bundle_manifest.json'),'datasets':{'FloW':details},
        'runtime_assets':list({e['path']:e for e in assets}.values()),
        'reference_preflight_sha256':sha(ref/'preflight.json'),'all_2000_images_and_labels_rehashed':True,
        'pixel_dimensions_and_gt_inherited_from_hash_verified_reference_preflight':True,
        'anchor_jobs_verified':len(req['anchors']),'predecessor_27_complete':True,
        'scientific_training_args_identical_to_source_E_series':True,'other_experiments_modified':False})
    c.verify_release(req);versions=c.bootstrap(req)
    import importlib.metadata
    assert importlib.metadata.version('pycocotools')=='2.0.10'
    import coco_functions
    write(root/'evaluator_self_test.json',coco_functions.self_test())
    write(root/'runtime_check.json',versions)
    print(json.dumps({'status':'prepared','root':str(root),'preflight_sha256':sha(root/'preflight.json'),'runtime':versions}),flush=True)

if __name__=='__main__':main()
