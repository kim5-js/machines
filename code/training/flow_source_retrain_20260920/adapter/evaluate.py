"""Evaluate val-selected best.pt on distinct val/test, without tuning."""
import argparse
import gzip
import importlib.metadata
import math
from pathlib import Path
import traceback
import common as c


def official_evaluation(req,job,split,records,out,checkpoint):
    import numpy as np
    import torch
    from ultralytics import YOLO,RTDETR
    from coco_functions import coco_evaluate,self_test
    c.require(importlib.metadata.version('pycocotools')=='2.0.10','COCO API version changed')
    test=self_test()
    model=(RTDETR if job['model_key']=='rtdetr_l' else YOLO)(str(checkpoint))
    c.require({str(k):v for k,v in model.names.items()}=={'0':'bottle'},'Evaluation model class mismatch')
    torch.manual_seed(0);np.random.seed(0)
    probe={'forward_calls':0,'maximum_forward_batch':0,'input_dtypes':[],'input_shapes':[]}
    def before_forward(module,args):
        tensor=args[0]
        c.require(tensor.ndim==4 and tensor.shape[0]==1 and tensor.dtype==torch.float32,'Official evaluation must use actual batch 1 FP32')
        probe['forward_calls']+=1;probe['maximum_forward_batch']=max(probe['maximum_forward_batch'],int(tensor.shape[0]))
        for field,value in [('input_dtypes',str(tensor.dtype)),('input_shapes',list(tensor.shape))]:
            if value not in probe[field]:probe[field].append(value)
    def on_predict_start(predictor):
        c.require(predictor.dataset.bs==1 and not predictor.model.fp16 and predictor.args.quantize is None,'Inference precision/batch changed')
        if not getattr(predictor,'_flow_guard',False):
            predictor.model.register_forward_pre_hook(before_forward);predictor._flow_guard=True
    model.add_callback('on_predict_start',on_predict_start)
    predictions=[];processed=[]
    for image_id,r in enumerate(records,1):
        results=model.predict(source=[r['image']],stream=True,verbose=False,save=False,**req['evaluation']['official_coco']['inference'])
        n=0
        for result in results:
            n+=1
            c.require(Path(result.path).resolve()==Path(r['image']).resolve() and tuple(result.orig_shape)==(r['height'],r['width']),'Wrong prediction image or dimensions')
            values=result.boxes.data.detach().cpu().numpy()
            c.require(values.shape[1]==6 and np.isfinite(values).all(),'Nonfinite prediction')
            for x1,y1,x2,y2,score,cls in values.tolist():
                c.require(cls==0 and 0<=score<=1 and x1<=x2 and y1<=y2,'Invalid prediction')
                predictions.append({'image_id':image_id,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
            processed.append(image_id)
        c.require(n==1,'Incomplete inference coverage')
        if image_id%100==0 or image_id==len(records):
            c.write(out/'progress.json',{'status':'official_coco_inference','job':job['key'],'split':split,'images':image_id,'target_images':len(records),'updated_utc':c.now()})
    c.require(processed==list(range(1,len(records)+1)) and probe['forward_calls']==len(records),'Inference coverage mismatch')
    import json
    with gzip.open(out/'predictions.json.gz','wt',encoding='utf-8') as f:json.dump(predictions,f,allow_nan=False)
    gt=c.ROOT/'evaluation_data'/(split+'_ground_truth.json')
    metrics,per_class=coco_evaluate(gt,predictions,out)
    c.require(all(v is None or math.isfinite(v) for v in metrics.values()),'Invalid COCO metrics')
    from recover_wsodd_legacy_coco_areas import release_gpu
    release_gpu(model)
    return {'evaluator':req['evaluation']['official_coco']['evaluator'],'inference':req['evaluation']['official_coco']['inference'],'metrics':metrics,'per_class_AP':per_class,'actual_batch_probe':probe,'images':len(records),'predictions':len(predictions),'ground_truth_sha256':c.sha(gt),'predictions_sha256':c.sha(out/'predictions.json.gz'),'arrays_sha256':c.sha(out/'coco_arrays.npz'),'self_test':test,'preprocessing':'imgsz=960 rect=False FP32 batch=1; native architecture predictor and postprocessing'}


def historical_secondary(req,job,split,records,out,done):
    import torch
    from ultralytics import YOLO
    import evaluate_e3_coco_areas as evaluator
    import recover_wsodd_legacy_coco_areas as recovery
    import recover_wsodd_legacy_coco_areas_v2 as chunked
    contract=req['evaluation']['ablation_legacy_secondary']
    c.require(c.sha(evaluator.__file__)==contract['primary_evaluator_sha256'],'Historical evaluator drift')
    c.require(c.sha(chunked.__file__)==contract['prediction_chunk_adapter_sha256'],'Historical chunk adapter drift')
    yp=c.ROOT/'evaluation_data'/(split+'.yaml')
    images,data=evaluator.resolve_validation_images(yp)
    c.require(len(images)==len(records) and {str(p.resolve()) for p in images}=={str(Path(r['image']).resolve()) for r in records},'Historical evaluation members mismatch')
    recovery.EXPECTED_VALIDATION_IMAGES=len(records);recovery.EXPECTED_CLASSES=1
    original_predict=YOLO.predict
    observed={'finite_images':0,'finite_boxes':0}
    def guarded_predict(self,*args,**kwargs):
        for result in original_predict(self,*args,**kwargs):
            if result.boxes is not None:
                c.require(bool(torch.isfinite(result.boxes.data).all()) and bool((result.boxes.cls==0).all()),'Invalid historical prediction')
                observed['finite_boxes']+=len(result.boxes)
            observed['finite_images']+=1
            yield result
    YOLO.predict=guarded_predict
    try:
        raw=chunked.evaluate_one_chunked(run_key=job['key'],run_dir=Path(job['run_dir']),data_yaml=yp,output_path=out/'historical_area_evaluation.json',imgsz=960,batch=8,device=0)
    finally:YOLO.predict=original_predict
    c.require(observed['finite_images']==len(records),'Incomplete historical prediction coverage')
    c.require(raw['artifacts']['best_weights_sha256']==done['artifacts']['weights/best.pt'],'Historical evaluator used another checkpoint')
    area=raw['area_evaluation']['metrics']
    metrics={'mAP50':area['all']['AP50'],'mAP50-95':area['all']['AP50_95'],'APsmall':area['small']['AP50_95'],'APmedium':area['medium']['AP50_95'],'APlarge':area['large']['AP50_95']}
    c.require(all(v is None or math.isfinite(v) and 0<=v<=1 for v in metrics.values()),'Invalid historical metrics')
    return {'evaluator':'frozen_project_COCO_style_area_evaluator','metrics':metrics,'area_details':area,'prediction_evidence':observed,'raw_sha256':c.sha(out/'historical_area_evaluation.json'),'note':'Unchanged historical evaluator and chunked prediction. Only area_evaluation.metrics are used; legacy raw WSODD text is not dataset evidence. Separate from official COCOeval and native training metrics.'}


def main(key,phase):
    req=c.request();c.require(phase=='full','Final evaluations use full runs')
    job=c.resolve_job(req,key,phase);adapter=c.verify_release(req);done=c.verify_completion(req,job,adapter)
    c.bootstrap(req)
    for split in ['val','test']:
        records=c.verify_eval_data(req,job,split)
        out=c.ROOT/'evaluations'/key/split
        c.require(not out.exists(),'Existing evaluation evidence must be preserved')
        out.mkdir(parents=True)
        checkpoint=Path(job['run_dir'])/'weights/best.pt'
        official=official_evaluation(req,job,split,records,out,checkpoint)
        historical=historical_secondary(req,job,split,records,out,done) if job['family']=='ablation' else None
        c.verify_completion(req,job,adapter);c.verify_release(req);c.verify_eval_data(req,job,split)
        result={'status':'completed','completed_utc':c.now(),'job':job,'evaluation_split':split,'dedicated_test':split=='test','observed_source_group_disjoint':True,'unknown_video_provenance_guaranteed':False,'request_sha256':c.REQUEST_SHA,'adapter_manifest_sha256':adapter,'completion_sha256':c.sha(Path(job['run_dir'])/'completion.json'),'checkpoint_sha256':done['artifacts']['weights/best.pt'],'checkpoint_selection':done['checkpoint_selection'],'evaluation_images':len(records),'evaluation_boxes':sum(r['boxes'] for r in records),'manifest_sha256':c.sha(c.ROOT/'verified_data/FloW'/(split+'.jsonl')),'official_coco':official,'historical_secondary':historical,'test_used_for_model_selection':False}
        c.write(out/'evaluation.json',result)
        print('EVALUATION_COMPLETED '+key+' '+split,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--phase',required=True);args=p.parse_args()
    try:main(args.job,args.phase)
    except BaseException:
        c.write(c.ROOT/'failures'/f'{args.phase}_{args.job}_evaluation.json',{'utc':c.now(),'traceback':traceback.format_exc()})
        raise
