"""Run an archived recipe, with evidence callbacks that do not change optimization."""
from support import RECOVERY_CODE, resolve_run, verify_recovery, validate_csv, assert_replay_prefix
from observer import ValidationAudit
from common import CODE, now, sha, write, load, protocol
from pathlib import Path
import argparse
import csv
import hashlib
import math
import random
import shutil
import time
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import ultralytics
from ultralytics.utils.torch_utils import strip_optimizer


def weights_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(root, job, smoke=False):
    p = protocol(root)
    verify_recovery(root)
    arm, dataset, seed = job['arm'], job['dataset'], job['seed']
    assert not smoke, 'Recovery is only for the frozen full queue'
    directory = resolve_run(root, job['key'])
    if directory.exists():
        raise RuntimeError(f'Existing run directory; refusing silent restart: {directory}')
    recipes = load(CODE / 'historical/recipes.json')
    vanilla = arm in {'baseline', 'weights_only'}
    recipe = dict(recipes['baseline' if vanilla else 'wsodd_refsoft'])
    recipe.pop('save_dir', None)
    recipe.pop('model', None)
    if arm == 'weights_only':
        recipe.update(box=4.0, dfl=1.3)
    mode = arm if arm in {'duplicate', 'refsoft'} else 'none'
    data = p['datasets'][dataset]['overlay_yamls'][mode]
    if smoke:
        data = str(root / 'smoke_inputs' / dataset / mode / 'data.yaml')
        recipe.update(epochs=2, close_mosaic=0, patience=100)
    recipe.update(data=data, seed=seed, device=0, project=str(directory.parent),
                  name=directory.name, exist_ok=False, plots=False, verbose=False)
    assert recipe['pretrained'] is False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = YOLO('yolov8s.yaml' if vanilla else str(CODE / 'historical/yolov8s_cbam.yaml'))
    started = time.monotonic()
    evidence = {'optimizer_step_calls': 0, 'optimizer_parameter_changed': False,
                'normal_train_end': False, 'best_map_epoch': None, 'best_map': -1.0,
                'epoch_seconds': [], 'epoch_seen': set()}
    timer = [started]
    audit = ValidationAudit()
    overflow_epochs = []
    replay_verified = [None]

    def on_start(trainer):
        audit.trainer = trainer
        if Path(trainer.save_dir).resolve() != directory.resolve():
            raise RuntimeError('Unexpected output path')
        actual = yaml.safe_load((directory / 'args.yaml').read_text())
        for k, v in recipe.items():
            if k == 'device':
                assert str(actual[k]) == str(v)
            else:
                assert actual.get(k) == v, (k, actual.get(k), v)
        net = trainer.model
        crit = net.init_criterion()
        indices = [i for i, m in enumerate(net.model) if type(m).__name__ == 'C2f_CBAM']
        assert indices == ([] if vanilla else [4, 6]), indices
        assert crit.bbox_loss.loss_type == ('ciou' if vanilla else 'focal_eiou')
        assert net.yaml['scale'] == 's'
        initial_hash = weights_hash(net)
        runtime = {'ultralytics_version': ultralytics.__version__, 'ultralytics_file': ultralytics.__file__,
                   'pretrained': actual['pretrained'], 'scale': net.yaml['scale'],
                   'parameters': sum(x.numel() for x in net.parameters()), 'cbam_indices': indices,
                   'criterion': type(crit).__name__, 'bbox_loss': crit.bbox_loss.loss_type,
                   'box': actual['box'], 'dfl': actual['dfl'], 'initial_state_sha256': initial_hash,
                   'train_images': len(trainer.train_loader.dataset), 'batches_per_epoch': len(trainer.train_loader),
                   'optimizer': type(trainer.optimizer).__name__, 'amp': bool(trainer.amp),
                   'protocol_sha256': sha(root / 'protocol.json'), 'job': job, 'smoke': smoke}
        assert runtime['ultralytics_version'] == '8.3.182'
        assert str(CODE / 'vendor') in runtime['ultralytics_file']
        assert runtime['optimizer'] == 'SGD' and runtime['amp']
        runtime['monitoring_revision'] = 'recovery_v1'
        runtime['recovery_manifest_sha256'] = sha(RECOVERY_CODE / 'manifest.json')
        write(directory / 'runtime.json', runtime)
        write(directory / 'job.json', job)
        param = next(x for x in net.parameters() if x.requires_grad)
        original_step = trainer.optimizer_step

        def observed_step():
            before = param.detach().clone() if not evidence['optimizer_parameter_changed'] else None
            original_step()
            evidence['optimizer_step_calls'] += 1
            if before is not None:
                assert torch.isfinite(param).all()
                evidence['optimizer_parameter_changed'] = not torch.equal(before, param.detach())
        trainer.optimizer_step = observed_step

    def epoch_start(trainer):
        timer[0] = time.monotonic()

    def batch_end(trainer):
        if not torch.isfinite(trainer.tloss).all():
            raise FloatingPointError('Non-finite train loss')
        if (root / 'STOP').exists():
            raise InterruptedError('User STOP sentinel')

    def fit_end(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch in evidence['epoch_seen']:
            return
        evidence['epoch_seen'].add(epoch)
        metrics = {k: float(v) for k, v in trainer.metrics.items()}
        observation = audit.check(trainer, metrics)
        if observation['raw_nonfinite_metrics']:
            overflow_epochs.append(epoch)
        replay_verified[0] = assert_replay_prefix(root, job, directory, epoch)
        score = metrics['metrics/mAP50-95(B)']
        if score > evidence['best_map']:
            evidence['best_map'], evidence['best_map_epoch'] = score, epoch
            shutil.copyfile(trainer.last, directory / 'weights/best_map.pt')
        evidence['epoch_seconds'].append(time.monotonic() - timer[0])
        row = {'status': 'training', 'utc': now(), 'key': job['key'], 'epoch': epoch,
               'epochs': trainer.epochs, **observation, 'elapsed_seconds': time.monotonic()-started,
               'recent_seconds_per_epoch': float(np.mean(evidence['epoch_seconds'][-5:])),
               'optimizer_step_calls': evidence['optimizer_step_calls'],
               'optimizer_parameter_changed': evidence['optimizer_parameter_changed']}
        write(directory / 'progress.json', row)
        with (directory / 'epoch_evidence.jsonl').open('a') as f:
            import json
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def train_end(trainer):
        evidence['normal_train_end'] = True
        assert evidence['optimizer_parameter_changed'] and evidence['optimizer_step_calls'] > 0

    for name, fn in [('on_train_start', on_start), ('on_train_epoch_start', epoch_start),
                     ('on_train_batch_end', batch_end), ('on_val_start', audit.start), ('on_val_end', audit.end),
                     ('on_fit_epoch_end', fit_end), ('on_train_end', train_end)]:
        model.add_callback(name, fn)
    model.train(**recipe)
    assert evidence['normal_train_end']
    rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader((directory / 'results.csv').open())]
    assert [int(r['epoch']) for r in rows] == list(range(1, len(rows)+1))
    validate_csv(rows, set(overflow_epochs))
    checkpoints = {}
    for name in ['best.pt', 'best_map.pt']:
        cp = directory / 'weights' / name
        if name == 'best_map.pt':
            strip_optimizer(cp)
        checkpoints[name] = {'path': str(cp), 'sha256': sha(cp)}
    completion = {'status': 'completed', 'utc': now(), 'job': job, 'smoke': smoke,
                  'protocol_sha256': sha(root / 'protocol.json'), 'runtime': load(directory / 'runtime.json'),
                  'epochs_completed': len(rows), 'maximum_epochs': recipe['epochs'],
                  'early_stopped': len(rows) < recipe['epochs'], 'elapsed_seconds': time.monotonic()-started,
                  'optimizer_step_calls': evidence['optimizer_step_calls'],
                  'optimizer_parameter_changed': evidence['optimizer_parameter_changed'],
                  'best_map_epoch': evidence['best_map_epoch'], 'peak_val_map': evidence['best_map'],
                  'checkpoints': checkpoints, 'results_sha256': sha(directory / 'results.csv')}
    completion.update(monitoring_revision='recovery_v1',
                      recovery_manifest_sha256=sha(RECOVERY_CODE / 'manifest.json'),
                      validation_overflow_epochs=overflow_epochs,
                      failed_prefix_reproduced=replay_verified[0])
    write(directory / 'completion.json', completion)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--job', required=True)
    a.add_argument('--smoke', action='store_true')
    args = a.parse_args()
    p = protocol(args.root)
    job = next(x for x in p['jobs'] if x['key'] == args.job)
    train(args.root, job, args.smoke)
