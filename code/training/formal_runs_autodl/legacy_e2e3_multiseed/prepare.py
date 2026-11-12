"""Verify frozen historical inputs and prepare the authorized follow-up release."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, io, json, shutil, traceback
import yaml
from PIL import Image
BUNDLE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/legacy_e2e3_multiseed_20260911')
FIRST = Path('/root/autodl-tmp/refsoft_reproduction_20260910/release_v1')
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(p)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def members(value,base):
    result=[]
    for item in value if isinstance(value,list) else [value]:
        p=Path(item); p=p if p.is_absolute() else base/p
        if p.is_dir():
            result.extend(x.resolve() for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
        else:
            for line in p.read_text().splitlines():
                if not line.strip(): continue
                q=Path(line.strip()); result.append((q if q.is_absolute() else p.parent/q).resolve())
    return sorted(result)
def main():
    if not __debug__: raise RuntimeError('Run preparation without Python optimization so all audit assertions execute')
    bundle=read(BUNDLE/'bundle_manifest.json')
    assert bundle['request_sha256']=='ae66cd4e7e435be3777d860460a22455a7a9a75adc6d30c06a054cb323ed8c1e'
    for entry in bundle['files']:
        path=(BUNDLE/entry['path']).resolve()
        assert path.is_relative_to(BUNDLE), 'Bundle path escaped the upload directory'
        assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'],entry['path']
    assert read(FIRST/'state.json')['status']=='completed'
    assert read(FIRST/'summary.json')['status']=='completed'
    assert sha(FIRST/'finalization_20260913/audit.json')=='1881c21e4f20c5e1513ccb0838cb37490e419d7309243a899af460d4ed2cbe7e'
    assert read(FIRST/'state.json')['completed_runs']==30
    assert read(FIRST/'summary.json')['evaluated_runs']==30
    assert read(FIRST/'finalization_20260913/audit.json')['status']=='passed'
    proof=read(BUNDLE/'predecessor_proof.json')
    assert sha(FIRST/'final_evidence_20260913.zip')==proof['archive_sha256']
    assert proof['first_stage_complete'] and proof['status']=='passed' and proof['verified_files']==799
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: cmd=(p/'cmdline').read_bytes()
        except OSError: continue
        assert str(FIRST).encode() not in cmd, ('predecessor process remains',p.name)
    stage=BASE/'staged_v1'
    assert sha(stage/'request.json')=='ae66cd4e7e435be3777d860460a22455a7a9a75adc6d30c06a054cb323ed8c1e'
    assert sha(stage/'staging_manifest.json')=='0c47c11d7c9539a7f08ec687bac0e556d18b4008e76369dec13a888714dcc948'
    manifest=read(stage/'staging_manifest.json')
    for entry in manifest['files']: assert sha(stage/entry['path'])==entry['sha256'],entry['path']
    root=BASE/'release_v2'
    assert not root.exists(), 'Release already exists; inspect it rather than overwrite'
    shutil.copytree(stage,root)
    shutil.copy2(BUNDLE/'prepare.py',root/'prepare.py')
    shutil.copytree(BUNDLE/'adapter',root/'adapter')
    shutil.copy2(BUNDLE/'bundle_manifest.json',root/'execution_bundle_manifest.json')
    code=[{'path':e['path'],'sha256':e['sha256']} for e in bundle['files']
          if e['path'].startswith('adapter/') or e['path']=='prepare.py']
    write(root/'adapter_manifest.json',{'files':code,'bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json')})
    assets=[
        (Path('/root/autodl-tmp/ring3_setup/autodl/yolo26n.pt'),'yolo26n.pt',
         '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'),
        (FIRST/'settings/Arial.ttf','settings/Arial.ttf',
         '525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9')]
    runtime_assets=[]
    for source,relative,expected_sha in assets:
        assert sha(source)==expected_sha, str(source)
        destination=root/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        assert sha(destination)==expected_sha
        runtime_assets.append({'path':relative,'sha256':expected_sha})

    write(root/'predecessor_gate.json',proof)
    req=read(root/'request.json')
    assert sha(Path(req['pretrained_weight']['path']))==req['pretrained_weight']['sha256']
    output={'status':'running','started_utc':now(),'request_sha256':sha(root/'request.json'),'staging_manifest_sha256':sha(stage/'staging_manifest.json'),'datasets':{},'runtime_assets':runtime_assets,'bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json')}
    write(root/'preflight.json',output)
    try:
        for ds in req['datasets']:
            yp=Path(ds['yaml']); assert sha(yp)==ds['yaml_sha256']
            data=yaml.safe_load(yp.read_text()); base=Path(data['path'])
            names=data['names']; names=dict(enumerate(names)) if isinstance(names,list) else names
            assert {str(k):v for k,v in names.items()}==ds['names']
            details={}
            for split in ['train','val']:
                mf=root/'dataset_manifests'/ds['key']/(split+'.jsonl')
                assert sha(mf)==ds['splits'][split]['image_membership_size_label_content_sha256']
                rows=[json.loads(line) for line in mf.read_text().splitlines()]
                assert len(rows)==ds['splits'][split]['images']
                paths=[(base/r[0]).resolve() for r in rows]
                assert members(data[split],base)==sorted(paths),(ds['key'],split,'membership mismatch')
                def inspect(row):
                    imrel,size,labrel,labhash=row
                    image=base/imrel; label=base/labrel
                    blob=image.read_bytes()
                    assert len(blob)==size and sha(label)==labhash,str(image)
                    with Image.open(io.BytesIO(blob)) as im:
                        im.load(); width,height=im.size
                    return {'image':str(image.resolve()),'label':str(label.resolve()),'image_sha256':hashlib.sha256(blob).hexdigest(),'label_sha256':labhash,'width':width,'height':height}
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: verified=list(pool.map(inspect,rows))
                dest=root/'verified_data'/ds['key']/(split+'.jsonl'); dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in verified))
                details[split]={'images':len(verified),'pixel_manifest':str(dest),'sha256':sha(dest)}
                print(json.dumps({'dataset':ds['key'],'split':split,'verified_images':len(verified),'utc':now()}),flush=True)
            output['datasets'][ds['key']]=details; write(root/'preflight.json',output)
        output.update(status='passed',completed_utc=now(),historical_staging_files=len(manifest['files']),pretrained_weight_sha256=req['pretrained_weight']['sha256'])
        write(root/'preflight.json',output)
    except BaseException:
        output.update(status='failed',traceback=traceback.format_exc(),failed_utc=now());write(root/'preflight.json',output);raise
    print(json.dumps({'status':'passed','root':str(root),'preflight_sha256':sha(root/'preflight.json')}),flush=True)
if __name__=='__main__': main()