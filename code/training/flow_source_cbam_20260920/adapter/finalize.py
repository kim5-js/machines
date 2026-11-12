"""Summarize new factor runs with hash-verified, same-protocol source E anchors."""
from pathlib import Path
import csv,math,statistics
import common as c
METRICS=['mAP50','mAP50-95','APsmall','APmedium','APlarge']

def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(req,complete=False):
    adapter=c.verify_release(req);official=[];secondary=[];native=[];evidence=[]
    specs=[(c.resolve_job(req,j['key'],'full'),c.ROOT,False) for j in c.job_list(req)]
    for a in req['anchors']:
        base=c.ROOT/'anchors';done=c.read(base/'runs'/a['key']/'completion.json')
        specs.append((done['job'],base,True))
    for job,base,anchor in specs:
        run=base/'runs'/job['key'];p=run/'completion.json'
        if not p.exists():continue
        done=c.read(p) if anchor else c.verify_completion(req,job,adapter)
        values=c.metric_rows(run/'results.csv')
        chosen=next(r for r in values if int(float(r['epoch']))==done['checkpoint_selection']['epoch'])
        common={'job':job['key'],'arm':job['arm'],'seed':job['seed'],'reused_anchor':anchor,'best_epoch':done['checkpoint_selection']['epoch']}
        native.append({**common,'split':'val','evaluator':'native_training_checkpoint_epoch',
            'mAP50':float(chosen['metrics/mAP50(B)']),'mAP50-95':float(chosen['metrics/mAP50-95(B)']),
            'precision':float(chosen['metrics/precision(B)']),'recall':float(chosen['metrics/recall(B)'])})
        for split in ['val','test']:
            ep=base/'evaluations'/job['key']/split/'evaluation.json'
            if not ep.exists():continue
            r=c.read(ep);out=ep.parent
            expected=req['source_reference']['request_sha256'] if anchor else c.REQUEST_SHA
            c.require(r['status']=='completed' and r['request_sha256']==expected and r['checkpoint_sha256']==done['artifacts']['weights/best.pt'],'Evaluation provenance mismatch')
            coco=r['official_coco'];hist=r['historical_secondary']
            c.require(c.sha(out/'predictions.json.gz')==coco['predictions_sha256'] and c.sha(out/'coco_arrays.npz')==coco['arrays_sha256'],'Prediction evidence changed')
            c.require(c.sha(out/'historical_area_evaluation.json')==hist['raw_sha256'],'Historical raw evaluation changed')
            common_row={**common,'split':split,'images':r['evaluation_images'],'boxes':r['evaluation_boxes']}
            official.append({**common_row,'evaluator':'official_pycocotools_COCOeval_2.0.10',**{m:coco['metrics'][m] for m in METRICS}})
            secondary.append({**common_row,'evaluator':'frozen_project_COCO_style_area_evaluator',**{m:hist['metrics'][m] for m in METRICS}})
            evidence.append({'path':ep.relative_to(c.ROOT).as_posix(),'sha256':c.sha(ep),'reused_anchor':anchor})
    count=len([r for r in native if not r['reused_anchor']])
    eval_count=len([r for r in official if not r['reused_anchor']])
    if complete:c.require(count==15 and eval_count==30 and len(native)==24 and len(official)==len(secondary)==48,'Incomplete final matrix')
    out=c.ROOT/'results';out.mkdir(exist_ok=True)
    csv_write(out/'official_coco_per_run.csv',official);csv_write(out/'historical_secondary_per_run.csv',secondary)
    csv_write(out/'native_val_checkpoint_metrics.csv',native)
    means=[];pairs=[]
    for source in [official,secondary]:
        for split in ['val','test']:
            rows=[r for r in source if r['split']==split];lookup={(r['arm'],r['seed']):r for r in rows}
            for arm in ['E0','E2','E3']+req['matrix']['arms']:
                for metric in METRICS:
                    v=[r[metric] for r in rows if r['arm']==arm and r[metric] is not None]
                    c.require(all(math.isfinite(x) for x in v),'Nonfinite score')
                    means.append({'evaluator':source[0]['evaluator'],'split':split,'arm':arm,'metric':metric,'n':len(v),
                        'mean_percent':100*statistics.mean(v) if v else None,'sample_sd_pp':100*statistics.stdev(v) if len(v)>1 else None})
            for a,b in req['evaluation']['comparisons']:
                for metric in METRICS:
                    delta={s:100*(lookup[a,s][metric]-lookup[b,s][metric]) for s in req['matrix']['seeds']
                           if (a,s) in lookup and (b,s) in lookup and lookup[a,s][metric] is not None and lookup[b,s][metric] is not None}
                    v=list(delta.values())
                    pairs.append({'evaluator':source[0]['evaluator'],'split':split,'comparison':a+'-'+b,'metric':metric,
                        'seed0_pp':delta.get(0),'seed1_pp':delta.get(1),'seed2_pp':delta.get(2),'n':len(v),
                        'mean_pp':statistics.mean(v) if v else None,'sample_sd_pp':statistics.stdev(v) if len(v)>1 else None,
                        'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v)})
    csv_write(out/'mean_sample_sd.csv',means);csv_write(out/'paired_differences.csv',pairs)
    c.write(out/'summary.json',{'status':'completed' if complete else 'partial','updated_utc':c.now(),'request_sha256':c.REQUEST_SHA,
        'new_formal_trainings_completed':count,'new_official_val_test_evaluations_completed':eval_count,
        'new_historical_evaluations_completed':len([r for r in secondary if not r['reused_anchor']]),
        'reused_anchor_jobs':9,'evidence':evidence,'scientific_notes':req['scientific_notes']})
    def fmt(x):return 'NA' if x is None else f'{100*x:.3f}'
    lines=['# FloW 最新来源划分 CBAM/Focal 实验','',
        f'新增正式训练 {count}/15，新增 val/test 评价 {eval_count}/30；复用同划分 E0/E2/E3 共 9 组。',
        '', '1124 train / 335 val / 541 test。全部本次因素组与 E 系列采用相同 SGD 配方；best.pt 只按 val 选择。',
        '下面是固定 test 的官方 pycocotools COCOeval，分数为 %。三 seed 均值与样本 SD、配对差值另见 CSV。',
        '', '| 组别 | seed | mAP50 | mAP50–95 | APsmall | APmedium | APlarge |','|---|---:|---:|---:|---:|---:|---:|']
    for r in official:
        if r['split']=='test':lines.append('| '+r['arm']+' | '+str(r['seed'])+' | '+' | '.join(fmt(r[m]) for m in METRICS)+' |')
    lines+=['','val 官方结果、历史项目 COCO-style、native 验证分别保存在各自 CSV；缺失值不作零。未根据 test 调参。',
        '来源隔离由现有可追溯元数据支持；未知原始视频 ID 的局限保留。三 seed 不足以支持过强稳定性结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':summarize(c.request(),complete=True)
