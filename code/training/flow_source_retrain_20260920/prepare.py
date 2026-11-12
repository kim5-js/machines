"""Validate uploads, preserve old experiments, and prepare a new release."""
from pathlib import Path
import concurrent.futures
import csv
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
import yaml

BUNDLE=Path(__file__).resolve().parent


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):value.update(part)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert __debug__
    manifest=read(BUNDLE/'bundle_manifest.json')
    for e in manifest['files']:
        p=(BUNDLE/e['path']).resolve()
        assert p.is_relative_to(BUNDLE) and sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],e['path']
    req=read(BUNDLE/'request.json')
    assert sha(BUNDLE/'request.json')==manifest['request_sha256']
    old=Path(req['execution']['historical_training_root'])
    assert sha(old/'request.json')==req['snapshot']['historical_request_sha256']
    reference=read(old/'request.json')
    for key in ['training_reference_args','arms','mdg_config','runtime','scientific_implementation','pretrained_weight']:
        assert req[key]==reference[key],key
    for e in read(BUNDLE/'staging_manifest.json')['files']:
        assert sha(old/e['path'])==e['sha256'],e['path']
    assert sha(old/'adapter/train.py')==sha(BUNDLE/'adapter/train.py')
    for spec in req['baselines'].values():
        assert sha(Path(spec['historical_run'])/'args.yaml')==spec['historical_args_sha256']
        assert sha(spec['pretrained_weight']['path'])==spec['pretrained_weight']['sha256']
    archive=read(BUNDLE/'data_archive.json'); archive_path=BUNDLE.parent/archive['name']
    assert archive_path.stat().st_size==archive['bytes'] and sha(archive_path)==archive['sha256']
    root=Path(req['execution']['remote_root'])
    assert not root.exists(),'Existing release is preserved; no automatic overwrite'
    root.mkdir(parents=True)
    for e in manifest['files']:
        dest=root/e['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BUNDLE/e['path'],dest)
    shutil.copyfile(BUNDLE/'bundle_manifest.json',root/'execution_bundle_manifest.json')
    code=[{'path':e['path'],'sha256':e['sha256']} for e in manifest['files'] if e['path'].startswith('adapter/') or e['path'] in ('prepare.py','launch.py','probe.py','collect.py')]
    write(root/'adapter_manifest.json',{'files':code,'bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json')})
    target=root/'data/FloW_IMG_YOLO_2000_source_disjoint'
    source_files=read(root/'source_files.json')['files']
    with zipfile.ZipFile(archive_path) as z:
        members=z.namelist()
        assert len(set(members))==len(members)==len(source_files)
        assert set(members)=={e['path'] for e in source_files}
        assert all((target/name).resolve().is_relative_to(target) for name in members)
        z.extractall(target)
    for e in source_files:assert sha(target/e['path'])==e['sha256'],e['path']
    proof=read(target/'audit/independent_validation.json')
    assert proof['status']=='passed' and proof['manifest_sha256']==sha(target/'audit/manifest.csv')
    assets=[]
    for e in req['external_runtime_assets']:
        src=Path(req['external_runtime_source'])/e['path'];dest=root/'vendor'/e['path']
        assert sha(src)==e['sha256']
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        assets.append({'path':dest.relative_to(root).as_posix(),'sha256':sha(dest)})
    for relative,expected in [('yolo26n.pt','9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'),('settings/Arial.ttf','525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9')]:
        src=old/relative;dest=root/relative
        assert sha(src)==expected
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        assets.append({'path':relative,'sha256':sha(dest)})
    sys.path.insert(0,str(root/'adapter'))
    import common as c
    records=read(root/'data_records.json')
    assert len(records)==2000
    for field in ['source_group','raw_folder','original_parent','original_filename_key']:
        groups={}
        for row in records:
            if row[field]:groups.setdefault(row[field],set()).add(row['split'])
        assert all(len(v)==1 for v in groups.values()),field
    details={}
    for split in ['train','val','test']:
        rows=sorted([r for r in records if r['split']==split],key=lambda r:r['image'])
        assert {p.relative_to(target).as_posix() for p in (target/'images'/split).glob('*.jpg')}=={r['image'] for r in rows}
        def inspect(r):
            image,label=target/r['image'],target/r['label']
            assert sha(image)==r['image_sha256'] and sha(label)==r['label_sha256']
            with Image.open(image) as im:
                rgb=im.convert('RGB');assert rgb.size==(r['width'],r['height'])
                h=hashlib.sha256(f'RGB:{rgb.width}x{rgb.height}:'.encode('ascii'));h.update(rgb.tobytes())
                assert h.hexdigest()==r['pixel_sha256']
            lines=[line.split() for line in label.read_text().splitlines() if line.strip()]
            assert len(lines)==r['box_count']>0
            assert len({tuple(line) for line in lines})==len(lines)
            return {'image':str(image),'label':str(label),'image_sha256':r['image_sha256'],'label_sha256':r['label_sha256'],'width':r['width'],'height':r['height'],'boxes':r['box_count']}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(inspect,rows))
        dest=root/'verified_data/FloW'/(split+'.jsonl');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in verified),encoding='utf-8')
        counts={'images':len(rows),'boxes':sum(r['box_count'] for r in rows),'empty_labels':0}
        assert counts==req['datasets'][0]['splits'][split]
        details[split]={**counts,'sha256':sha(dest)}
        if split!='train':
            gt={'info':{'description':'Official FloW source-grouped '+split},'images':[],'annotations':[],'categories':[{'id':1,'name':'bottle'}]}
            for image_id,(r,verified_row) in enumerate(zip(rows,verified),1):
                gt['images'].append({'id':image_id,'file_name':Path(r['image']).name,'width':r['width'],'height':r['height']})
                for obj in ET.parse(target/r['archived_xml']).getroot().findall('object'):
                    x1,y1,x2,y2=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
                    gt['annotations'].append({'id':len(gt['annotations'])+1,'image_id':image_id,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
            assert len(gt['annotations'])==counts['boxes']
            gt_path=root/'evaluation_data'/(split+'_ground_truth.json');write(gt_path,gt)
            eval_config=yaml.safe_load((root/'dataset.yaml').read_text());eval_config['val']='images/'+split
            yp=root/'evaluation_data'/(split+'.yaml');yp.write_text(yaml.safe_dump(eval_config,sort_keys=False),encoding='utf-8')
            for p in [gt_path,yp]:assets.append({'path':p.relative_to(root).as_posix(),'sha256':sha(p)})
        print(json.dumps({'split':split,**counts}),flush=True)
    times={s:[float(r['timestamp_seconds']) for r in records if r['split']==s and r['time_evidence']=='fractional_unix_seconds_in_original_filename'] for s in ['train','val','test']}
    assert max(times['train'])<min(times['val']) and max(times['val'])<min(times['test'])
    assert all(r['split']=='train' for r in records if r['time_evidence']!='fractional_unix_seconds_in_original_filename')
    result={'status':'passed','utc':c.now(),'request_sha256':c.REQUEST_SHA,'bundle_manifest_sha256':sha(root/'execution_bundle_manifest.json'),'datasets':{'FloW':details},'runtime_assets':assets,'archive_sha256':archive['sha256'],'all_2000_image_hashes_pixels_labels_verified':True,'observed_source_groups_disjoint':True,'supported_capture_times_ordered':True,'unknown_source_limitations_preserved':True,'historical_scientific_files_and_baseline_args_verified':True,'training_observer_byte_identical':True,'other_jobs_modified':False}
    write(root/'preflight.json',result)
    c.verify_release(req)
    versions=c.bootstrap(req)
    import importlib.metadata
    assert importlib.metadata.version('pycocotools')=='2.0.10'
    import coco_functions
    write(root/'evaluator_self_test.json',coco_functions.self_test())
    write(root/'runtime_check.json',versions)
    print(json.dumps({'status':'passed','root':str(root),'preflight_sha256':sha(root/'preflight.json'),'runtime':versions}),flush=True)


if __name__=='__main__':main()
