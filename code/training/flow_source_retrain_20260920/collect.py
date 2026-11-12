"""Collect auditable result files; retain formal best/last checkpoints optionally."""
from pathlib import Path
import argparse
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'adapter'))
import common as c
p=argparse.ArgumentParser();p.add_argument('--include-weights',action='store_true');args=p.parse_args()
req=c.request();c.verify_release(req)
files=[]
for item in sorted(ROOT.rglob('*')):
    if not item.is_file():continue
    rel=item.relative_to(ROOT)
    if rel.parts[0] in ('data','exports','settings') or '__pycache__' in rel.parts:continue
    if item.suffix in ('.lock','.cache'):continue
    if item.suffix=='.pt' and not (args.include_weights and rel.parts[0]=='runs' and item.name in ('best.pt','last.pt')):continue
    files.append(item)
manifest={'utc':c.now(),'request_sha256':c.REQUEST_SHA,'included_formal_best_last_weights':args.include_weights,'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':c.sha(p),'bytes':p.stat().st_size} for p in files]}
directory=ROOT/'exports';directory.mkdir(exist_ok=True)
stamp=c.now().replace(':','').replace('+','_')
dest=directory/('evidence_'+stamp+'.zip')
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_STORED) as z:
    for item in files:z.write(item,item.relative_to(ROOT).as_posix())
    z.writestr('recovery_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':c.sha(dest),'files':len(files)}),flush=True)
