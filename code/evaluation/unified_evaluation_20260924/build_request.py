from pathlib import Path, PurePosixPath
import csv, hashlib, json

BASE = Path(__file__).resolve().parent
SOURCE = Path('D:/ring3/02_当前实验/已验证交付/AutoDL_论文可用结果_20260920')
def rows(rel):
    return list(csv.DictReader((SOURCE / rel).open(encoding='utf-8-sig')))
train = {r['experiment_id']: r for r in rows('03_完整目录/正式与阶段训练明细.csv')}
main = rows('01_主结果/主结果_逐次评价.csv')
sup = rows('02_独立补充/独立补充_逐次评价.csv')
cross = rows('02_独立补充/E0_E1多数据集_逐次评价.csv')
cross = [r for r in cross if r['dataset'] in ['SeaDronesSee', 'WSODD']]
extra = [r for r in sup if r['dataset']=='WSODD' and r['seed'] in ['42','123','3407']
         and r['condition']=='normal' and r['arm'] in ['E0_pretrained_sgd','E1_deterministic_pretrained_sgd']]
hashes = {}
for r in main + sup:
    if r.get('checkpoint_sha256'):
        hashes.setdefault(r['experiment_id'], set()).add(r['checkpoint_sha256'])
jobs = []
for r in main + cross + extra:
    eid = r['experiment_id']
    expected = r.get('checkpoint_sha256')
    if not expected:
        assert len(hashes[eid])==1, (eid, hashes[eid])
        expected = next(iter(hashes[eid]))
    family = ('detector' if r['comparison_group'].startswith('DETECTOR') else
              'matched' if r['comparison_group'].startswith('REFSOFT') else 'strategy')
    arm = r['arm']
    if family=='strategy':
        arm = 'E1' if arm.startswith('E1') else 'E0' if arm.startswith('E0') else arm
    run_dir = '/root/autodl-tmp/' + str(PurePosixPath(train[eid]['source_csv']).parent)
    jobs.append({'key':eid+'__'+r['condition'], 'experiment_id':eid, 'family':family,
                 'dataset':r['dataset'], 'arm':arm, 'seed':int(r['seed']),
                 'condition':r['condition'], 'split':r['evaluation_split'],
                 'checkpoint':run_dir+'/weights/best.pt', 'checkpoint_sha256':expected,
                 'historical_metrics':{k:float(r[k]) if r.get(k) else None for k in ['mAP50-95','mAP50','APsmall','APmedium','APlarge']},
                 'historical_evaluator':r['evaluator'], 'historical_source':r['source_file'],
                 'historical_evaluation_id':r['evaluation_id']})
assert len(jobs)==78 and len({j['key'] for j in jobs})==78
assert len({j['experiment_id'] for j in jobs})==54
request = {'schema':'ring3.manuscript_unified_reevaluation.v1',
           'root':'/root/autodl-tmp/manuscript_unified_eval_20260924',
           'purpose':'Reevaluate the 54 frozen manuscript checkpoints in 78 evaluations using one official COCO scoring implementation and one inference contract; no training or checkpoint reselection.',
           'inference':{'imgsz':960,'batch':1,'half':False,'rect':False,'conf':0.001,'iou':0.7,'max_det':300,'device':0},
           'evaluator':{'implementation':'pycocotools.COCOeval','version':'2.0.10','iou_type':'bbox','maxDets':[1,10,100],'recall_points':101,'iou_thresholds':'0.50:0.05:0.95','area_coordinates':'original image pixels'},
           'dataset_specs':{
               'SeaDronesSee':{'yaml':'/root/autodl-tmp/ring3_setup/autodl/datasets/seadronessee_source_disjoint_v1.yaml','split':'val','images':1555,'classes':5},
               'WSODD':{'yaml':'/root/autodl-tmp/ring3_setup/autodl/datasets/wsodd_sequence_disjoint_v1.yaml','split':'test','images':2406,'classes':14},
               'FloW':{'records':'/root/autodl-tmp/flow_source_retrain_20260920/release_v1/verified_data/FloW/test.jsonl','ground_truth':'/root/autodl-tmp/flow_source_retrain_20260920/release_v1/evaluation_data/test_ground_truth.json','split':'test','images':541,'classes':1}},
           'transform_source':'/root/autodl-tmp/ring3_setup/autodl/evaluate_pretrained_sgd_robustness.py',
           'transform_source_sha256':'cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f',
           'mdg_config':'/root/autodl-tmp/ring3_setup/autodl/e1_mdg_config.json',
           'mdg_config_sha256':'41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a',
           'condition_namespace_root':'ring3/cross-dataset/e0-e1/controlled-strong/v1',
           'jobs':jobs}
(BASE/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'evaluations':len(jobs),'checkpoints':54,'request_sha256':hashlib.sha256((BASE/'request.json').read_bytes()).hexdigest()}))
