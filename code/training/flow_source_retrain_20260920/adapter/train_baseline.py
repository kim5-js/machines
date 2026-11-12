"""Train one historical baseline with unchanged arguments and observation guards."""
import argparse
import hashlib
import math
from pathlib import Path
import traceback
import common as c


def main(key,phase):
    req=c.request();job=c.resolve_job(req,key,phase);adapter=c.verify_release(req)
    c.require(job['family']=='baseline','Wrong baseline job')
    run=Path(job['run_dir']);c.require(not run.exists(),'Existing run must be preserved')
    c.verify_data(req,job);versions=c.bootstrap(req)
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils.torch_utils import unwrap_model
    import ultralytics.utils.checks as checks
    checks.check_pip_update_available=lambda:None
    spec=req['baselines'][job['model_key']]
    c.require(c.sha(spec['pretrained_weight']['path'])==spec['pretrained_weight']['sha256'],'Pretrained file drift')
    model=YOLO(spec['pretrained_weight']['path'])
    base=model._smart_load('trainer')
    expected=c.expected_args(req,job)
    evidence={'job':job,'runtime':versions,'request_sha256':c.REQUEST_SHA,'adapter_manifest_sha256':adapter,
              'pretrained_weight':spec['pretrained_weight'],'source_trainer':base.__module__+'.'+base.__name__,
              'historical_args_sha256':spec['historical_args_sha256'],'preflight_sha256':c.sha(c.ROOT/'preflight.json'),
              'finite_training_batches':0,'consecutive_nonfinite_val_losses':0}

    def finite(value,label):
        if isinstance(value,dict):
            for k,v in value.items():finite(v,label+'/'+k)
        elif isinstance(value,torch.Tensor):c.require(bool(torch.isfinite(value.detach()).all()),'Nonfinite '+label)
        elif value is not None:c.require(math.isfinite(float(value)),'Nonfinite '+label)

    def tensor_hash(network):
        h=hashlib.sha256()
        for name,value in sorted(network.state_dict().items()):
            t=value.detach().cpu().contiguous()
            h.update(name.encode());h.update(str((str(t.dtype),list(t.shape))).encode());h.update(t.numpy().tobytes())
        return h.hexdigest()

    class ObservedTrainer(base):
        def get_model(self,cfg=None,weights=None,verbose=True):
            c.require(weights is not None,'Fresh generic pretrained initialization is required')
            network=super().get_model(cfg=cfg,weights=weights,verbose=verbose)
            original=weights['model'] if isinstance(weights,dict) else weights
            source,target=original.state_dict(),network.state_dict()
            head_prefix='model.'+str(len(network.model)-1)+'.'
            keys=[k for k in target if k in source and not k.startswith(head_prefix) and source[k].shape==target[k].shape]
            c.require(len(keys)>100,'Too few transferred backbone/neck tensors')
            for k in keys:c.require(torch.equal(source[k].float(),target[k].float()),'Pretrained transfer mismatch: '+k)
            evidence['initialization']={'state_sha256':tensor_hash(network),'verified_backbone_neck_tensors':len(keys),'fresh_pretrained':True}
            c.write(run/'runtime_evidence.json',evidence)
            return network

        def save_model(self):
            for label,network in [('model',unwrap_model(self.model)),('ema',self.ema.ema)]:
                for name,tensor in network.state_dict().items():finite(tensor,label+'/'+name)
            result=super().save_model()
            c.require(result is not False,'Native checkpoint saving failed')
            if self.best_fitness==self.fitness and self.best.exists():
                evidence['checkpoint_selection']={'rule':'native val mAP50-95 best.pt; no test selection','epoch':int(self.epoch)+1,'fitness':float(self.fitness)}
                c.write(run/'runtime_evidence.json',evidence)
            return result

    def verify_loader(loader,split):
        records=c.data_records(req,job,split)
        expected_files={str(Path(r['image']).resolve()):r for r in records}
        actual=[str(Path(p).resolve()) for p in loader.dataset.im_files]
        c.require(len(actual)==len(expected_files) and set(actual)==set(expected_files),'Wrong loaded '+split+' membership')
        for label in loader.dataset.labels:
            r=expected_files[str(Path(label['im_file']).resolve())]
            original=np.asarray([[float(v) for v in line.split()] for line in Path(r['label']).read_text().splitlines() if line.strip()],dtype=np.float32).reshape(-1,5)
            cached=np.concatenate((label['cls'],label['bboxes']),axis=1).astype(np.float32)
            c.require(np.array_equal(np.unique(original,axis=0),np.unique(cached,axis=0)),'Cached labels changed')
        return len(actual)

    def on_start(trainer):
        c.verify_args(vars(trainer.args),expected,DEFAULT_CFG_DICT)
        c.verify_args(yaml.safe_load((run/'args.yaml').read_text()),expected,DEFAULT_CFG_DICT)
        c.require(bool(trainer.amp)==expected['amp'],'Historical precision setting changed')
        c.require(type(trainer.optimizer).__name__=='AdamW','Historical optimizer changed')
        c.require({str(k):v for k,v in trainer.model.names.items()}=={'0':'bottle'},'Wrong classes')
        evidence['loaded_data']={'train':verify_loader(trainer.train_loader,'train'),'val':verify_loader(trainer.test_loader,'val')}
        evidence['actual_args']=vars(trainer.args).copy()
        evidence['started_utc']=c.now();c.write(run/'runtime_evidence.json',evidence)

    def on_batch(trainer):
        finite(trainer.loss,'training loss');finite(trainer.loss_items,'training loss components')
        evidence['finite_training_batches']+=1

    def on_epoch(trainer):
        rows=c.metric_rows(run/'results.csv')
        if not rows:return
        epoch=int(float(rows[-1]['epoch']))
        if epoch!=int(trainer.epoch)+1:return
        c.verify_args(vars(trainer.args),expected,DEFAULT_CFG_DICT)
        row=rows[-1]
        finite({k:float(v) for k,v in row.items() if k.startswith(('train/','metrics/','lr/'))},'epoch metrics')
        bad=any(not math.isfinite(float(v)) for k,v in row.items() if k.startswith('val/'))
        evidence['consecutive_nonfinite_val_losses']=evidence['consecutive_nonfinite_val_losses']+1 if bad else 0
        c.require(evidence['consecutive_nonfinite_val_losses']<3,'Three consecutive nonfinite native validation losses; stop and retain evidence')
        progress={'status':'training','job':job,'epoch':epoch,'target_epochs':job['epochs'],'updated_utc':c.now(),'epoch_seconds':float(trainer.epoch_time),'row':row,'native_fitness':float(trainer.fitness),'native_best_fitness':float(trainer.best_fitness)}
        c.write(run/'progress.json',progress)
        evidence['last_epoch']=progress;c.write(run/'runtime_evidence.json',evidence)
        print('BASELINE_PROGRESS '+job['key']+' '+str(epoch),flush=True)

    model.add_callback('on_train_start',on_start)
    model.add_callback('on_train_batch_end',on_batch)
    model.add_callback('on_fit_epoch_end',on_epoch)
    model.train(trainer=ObservedTrainer,**{k:v for k,v in expected.items() if k!='save_dir'})
    rows=c.metric_rows(run/'results.csv')
    c.require([int(float(r['epoch'])) for r in rows]==list(range(1,job['epochs']+1)),'Baseline did not complete its frozen epoch count')
    finite({k:float(v) for k,v in rows[-1].items()},'final baseline CSV')
    c.require('checkpoint_selection' in evidence and evidence['finite_training_batches']>0,'Missing actual training evidence')
    evidence['final_state_sha256']=tensor_hash(unwrap_model(model.trainer.model))
    c.require(evidence['final_state_sha256']!=evidence['initialization']['state_sha256'],'Model parameters never changed')
    for name in ['best.pt','last.pt']:
        cp=torch.load(run/'weights'/name,map_location='cpu',weights_only=False)
        network=cp.get('ema');network=cp.get('model') if network is None else network
        c.require(network is not None,'Checkpoint model missing')
        for k,v in network.state_dict().items():finite(v,name+'/'+k)
    c.verify_data(req,job);c.verify_release(req)
    evidence['completed_utc']=c.now();c.write(run/'runtime_evidence.json',evidence)
    artifacts=['args.yaml','results.csv','runtime_evidence.json','weights/best.pt','weights/last.pt']
    c.write(run/'completion.json',{'status':'completed','job':job,'epochs':len(rows),'completed_utc':c.now(),'request_sha256':c.REQUEST_SHA,'adapter_manifest_sha256':adapter,'preflight_sha256':c.sha(c.ROOT/'preflight.json'),'data_yaml_sha256':c.sha(job['data_yaml']),'checkpoint_selection':evidence['checkpoint_selection'],'native_patience_early_stop':False,'artifacts':{p:c.sha(run/p) for p in artifacts}})
    print('BASELINE_COMPLETED '+job['key'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--phase',choices=['smoke','full'],required=True);args=parser.parse_args()
    try:main(args.job,args.phase)
    except BaseException:
        c.write(c.ROOT/'failures'/f'{args.phase}_{args.job}_baseline_training.json',{'utc':c.now(),'traceback':traceback.format_exc()})
        raise
