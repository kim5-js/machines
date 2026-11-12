"""Auditable full-precision rows underlying the revised manuscript."""
from pathlib import Path
import csv,json,math,statistics
BASE=Path(__file__).resolve().parent
OUT=Path('D:/ring3/01_当前论文/基础依赖/20260924_统一评价修订')
source=json.loads((BASE/'evaluations_latest.json').read_text(encoding='utf-8'))
data=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
assert data['complete'] and len(source['evaluations'])==78
records={r['job']['key']:r for r in source['evaluations']}
def csv_file(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
evaluations=[]
for key,r in sorted(records.items()):
    j=r['job'];row={'evaluation_key':key,'experiment_id':j['experiment_id'],
        'historical_evaluation_id':j['historical_evaluation_id'],'record_type':'full_split_frozen_checkpoint_reevaluation',
        'protocol':'unified_official_COCOeval_20260924','family':j['family'],'dataset':j['dataset'],
        'split':j['split'],'arm_raw':j['arm'],'seed':j['seed'],'condition_raw':j['condition'],
        'condition_label':data['display_names']['conditions'][j['condition']],
        'checkpoint':r['checkpoint']['path'],'checkpoint_sha256':r['checkpoint']['sha256'],
        'images':r['images'],'boxes':r['ground_truth_boxes'],
        'ground_truth_sha256':r['ground_truth_sha256'],'input_pixels_sha256':r['input_pixels_sha256'],
        'predictions_sha256':r['predictions_sha256'],'evaluator_code_sha256':r['evaluator_code_sha256'],
        'evaluator':'pycocotools 2.0.10 COCOeval','imgsz':960,'actual_batch':1,'half':False,'rect':False}
    row.update({k+'_fraction':v for k,v in r['metrics'].items()})
    evaluations.append(row)
csv_file('unified_evaluations.csv',evaluations)
paired=[]
for name in ['stability_contrasts','matched_contrasts','condition_contrasts']:
    for c in data[name]:
        for metric in ['mAP50-95','APsmall','mAP50']:
            s=c['metrics'][metric]
            for seed,value,keys in zip(c['seeds'],s['values'],s['evaluation_keys']):
                paired.append({'analysis':name,'dataset':c['dataset'],'condition_raw':c['condition'],
                    'new_arm':c['new'],'reference_arm':c['old'],'seed':seed,'metric':metric,
                    'paired_change_pp':value,'group_mean_pp':s['mean'],'group_sample_sd_pp':s['sample_sd'],
                    'n':s['n'],'new_evaluation_key':keys[0],'reference_evaluation_key':keys[1]})
csv_file('paired_effects.csv',paired)
interactions=[];checks=[]
for c in data['condition_interactions']:
    for metric in ['mAP50-95','APsmall']:
        values=[];s=c['metrics'][metric]
        for seed,value,keys in zip(c['seeds'],s['values'],s['evaluation_keys']):
            (aug_c,base_c),(aug_n,base_n)=keys
            a,b,an,bn=[records[k]['metrics'][metric] for k in [aug_c,base_c,aug_n,base_n]]
            did=100*((a-an)-(b-bn))
            assert math.isclose(did,value,abs_tol=1e-12)
            values.append(did)
            interactions.append({'dataset':c['dataset'],'condition':data['display_names']['conditions'][c['condition']],
                'metric':metric,'seed':seed,'aug_condition_percent':100*a,'standard_condition_percent':100*b,
                'aug_normal_percent':100*an,'standard_normal_percent':100*bn,
                'paired_change_condition_pp':100*(a-b),'paired_change_normal_pp':100*(an-bn),
                'condition_interaction_pp':did,'group_mean_pp':s['mean'],'group_sample_sd_pp':s['sample_sd'],
                'n':3,'aug_condition_key':aug_c,'standard_condition_key':base_c,
                'aug_normal_key':aug_n,'standard_normal_key':base_n})
        assert math.isclose(statistics.mean(values),s['mean'],abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),s['sample_sd'],abs_tol=1e-12)
        checks.append({'dataset':c['dataset'],'condition':c['condition'],'metric':metric,
            'equivalent_condition_change_formula_passed':True,'sample_sd_of_paired_interactions_passed':True})
csv_file('condition_interactions_per_seed.csv',interactions)
(OUT/'qa/interaction_verification.json').write_text(json.dumps({'rows':len(interactions),'checks':checks,
 'formula':'100*((Aug_c-Aug_normal)-(Standard_c-Standard_normal))',
 'agreement_tolerance_pp':1e-12},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Exported 78 evaluation records,',len(paired),'paired metric rows and',len(interactions),'interaction rows.')
