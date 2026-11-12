"""Summarize completed evidence without pooling protocols or choosing by test."""
from pathlib import Path
import csv
import statistics
import common as c

METRICS=['mAP50','mAP50-95','APsmall','APmedium','APlarge']


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(req,complete=False):
    adapter=c.verify_release(req)
    rows=[];secondary=[];native=[];evidence=[]
    for spec in c.job_list(req):
        job=c.resolve_job(req,spec['key'],'full');run=Path(job['run_dir'])
        if not (run/'completion.json').exists():continue
        done=c.verify_completion(req,job,adapter)
        values=c.metric_rows(run/'results.csv')
        chosen=next(r for r in values if int(float(r['epoch']))==done['checkpoint_selection']['epoch'])
        native.append({'job':job['key'],'family':job['family'],'model':job['model_key'],'arm':job['arm'],'seed':job['seed'],'evaluation_split':'val','evaluator':'native_training_checkpoint_epoch','best_epoch':done['checkpoint_selection']['epoch'],'mAP50':float(chosen['metrics/mAP50(B)']),'mAP50-95':float(chosen['metrics/mAP50-95(B)']),'precision':float(chosen['metrics/precision(B)']),'recall':float(chosen['metrics/recall(B)'])})
        for split in ['val','test']:
            p=c.ROOT/'evaluations'/job['key']/split/'evaluation.json'
            if not p.exists():continue
            result=c.read(p)
            c.require(result['status']=='completed' and result['request_sha256']==c.REQUEST_SHA and result['checkpoint_sha256']==done['artifacts']['weights/best.pt'],'Evaluation provenance mismatch')
            official=result['official_coco'];out=p.parent
            c.require(c.sha(out/'predictions.json.gz')==official['predictions_sha256'] and c.sha(out/'coco_arrays.npz')==official['arrays_sha256'],'Evaluation artifacts changed')
            common={'job':job['key'],'family':job['family'],'model':job['model_key'],'arm':job['arm'],'seed':job['seed'],'evaluation_split':split,'best_epoch':done['checkpoint_selection']['epoch'],'images':result['evaluation_images'],'boxes':result['evaluation_boxes']}
            rows.append({**common,'evaluator':'official_pycocotools_COCOeval_2.0.10',**{m:official['metrics'][m] for m in METRICS}})
            if result['historical_secondary']:
                c.require(c.sha(out/'historical_area_evaluation.json')==result['historical_secondary']['raw_sha256'],'Historical result changed')
                secondary.append({**common,'evaluator':'frozen_project_COCO_style_area_evaluator',**result['historical_secondary']['metrics']})
            evidence.append({'path':p.relative_to(c.ROOT).as_posix(),'sha256':c.sha(p)})
    if complete:c.require(len(native)==13 and len(rows)==26 and len(secondary)==18,'Incomplete final experiment matrix')
    out=c.ROOT/'results';out.mkdir(exist_ok=True)
    csv_write(out/'official_coco_per_run.csv',rows);csv_write(out/'historical_secondary_per_run.csv',secondary);csv_write(out/'native_val_checkpoint_metrics.csv',native)
    means=[];pairs=[]
    for source in [rows,secondary]:
        for split in ['val','test']:
            selected=[r for r in source if r['family']=='ablation' and r['evaluation_split']==split]
            if not selected:continue
            evaluator=selected[0]['evaluator']
            for arm in ['E0','E2','E3']:
                for metric in METRICS:
                    group=[r[metric] for r in selected if r['arm']==arm and r[metric] is not None]
                    means.append({'evaluator':evaluator,'split':split,'arm':arm,'metric':metric,'n':len(group),'mean_percent':100*statistics.mean(group) if group else None,'sample_sd_percent':100*statistics.stdev(group) if len(group)>1 else None})
            lookup={(r['arm'],r['seed']):r for r in selected}
            for a,b in [('E2','E0'),('E3','E0'),('E3','E2')]:
                for metric in METRICS:
                    delta={s:100*(lookup[a,s][metric]-lookup[b,s][metric]) for s in [0,1,2] if (a,s) in lookup and (b,s) in lookup and lookup[a,s][metric] is not None and lookup[b,s][metric] is not None}
                    vv=list(delta.values())
                    pairs.append({'evaluator':evaluator,'split':split,'comparison':a+'-'+b,'metric':metric,'seed0_pp':delta.get(0),'seed1_pp':delta.get(1),'seed2_pp':delta.get(2),'n':len(vv),'mean_pp':statistics.mean(vv) if vv else None,'sample_sd_pp':statistics.stdev(vv) if len(vv)>1 else None,'positive':sum(v>0 for v in vv),'negative':sum(v<0 for v in vv)})
    csv_write(out/'ablation_mean_sample_sd.csv',means);csv_write(out/'ablation_paired_differences.csv',pairs)
    c.write(out/'summary.json',{'status':'completed' if complete else 'partial','updated_utc':c.now(),'request_sha256':c.REQUEST_SHA,'formal_trainings_completed':len(native),'official_val_test_evaluations_completed':len(rows),'historical_secondary_completed':len(secondary),'evidence':evidence,'scientific_notes':req['scientific_notes']})
    def fmt(v):return 'NA' if v is None else f'{100*v:.3f}'
    text='# FloW 来源分组复训结果\n\n'+('全部 13 组正式训练和独立 val/test 评价完成。' if complete else '仍在运行；下表只列已完成并核验的结果。')+'\n\n训练/验证/测试为 1124/335/541 张；best.pt 仅由 val 选择。四基模采用历史 AdamW 设置，E0/E2/E3 采用历史 SGD 设置，两套协议分开解释。表中为官方 pycocotools COCOeval，单位 %。\n\n| 模型/分支 | seed | 子集 | mAP50 | mAP50–95 | APsmall | APmedium | APlarge |\n|---|---:|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text+='| '+(' / '.join([r['model'],r['arm']]) if r['family']=='ablation' else r['model'])+' | '+str(r['seed'])+' | '+r['evaluation_split']+' | '+' | '.join(fmt(r[m]) for m in METRICS)+' |\n'
    text+='\nE 系列的旧项目 COCO-style 复核结果在 historical_secondary_per_run.csv；native 验证值在 native_val_checkpoint_metrics.csv。三种评价口径不混用。缺少某尺寸目标时 AP 保留为空，三种子统计采用样本标准差和同 seed 配对差值。\n\n来源隔离依据现有可追溯元数据；未知原始视频 ID 的局限保留。新旧分数差还受样本量、场景组成、原图编码和补齐标注影响，不解释为时间泄露的因果效应。\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':summarize(c.request(),complete=True)
