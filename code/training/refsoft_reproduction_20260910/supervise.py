"""Bounded serial queue: real-data smoke, archived runs, paired fixed-split reports."""
from common import CODE, now, sha, write, load, protocol, state, event
from pathlib import Path
import argparse
import csv
import fcntl
import importlib.util
import math
import os
import shutil
import statistics
import subprocess
import sys
import traceback


def child(root, script, job, smoke=False):
    cmd = [sys.executable, '-u', str(CODE / script), '--root', str(root), '--job', job['key']]
    if smoke:
        cmd.append('--smoke')
    log = root / 'logs' / (('smoke_' if smoke else '') + job['key'] + '_' + script + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    state(root, 'smoke' if smoke else 'training' if script == 'train.py' else 'evaluating', current_run=job['key'], log=str(log))
    with log.open('a') as output:
        r = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'{script} failed ({r.returncode}): {log}')


def source_checks(root):
    p = protocol(root)
    for d in p['datasets'].values():
        for row in load(d['inventory']):
            for kind in ['image', 'label']:
                path = Path(row[kind])
                if sha(path) != row[kind + '_sha256']:
                    raise RuntimeError('Source changed: ' + str(path))
    return True


def numerical_checks(root):
    import torch
    from ultralytics.utils.loss import BboxLoss, FocalEIoULoss
    from custom_losses.focal_eiou import FocalEIoULoss as ReferenceFocal
    spec = importlib.util.spec_from_file_location('ultralytics.utils.historical_baseline_loss', CODE / 'historical/baseline_loss.py')
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    target = torch.tensor([[0., 0., 3., 4.], [4., 3., 8., 8.], [10., 10., 14., 15.]], dtype=torch.float64)
    a = torch.tensor([[.2, .1, 3.1, 4.4], [5., 4., 9., 8.], [1., 2., 2., 4.]], dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    x = FocalEIoULoss(reduction='none')(a, target)
    y = ReferenceFocal(reduction='none')(b, target)
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    x.sum().backward(); y.sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert torch.isfinite(a.grad).all()
    torch.manual_seed(711)
    dist = torch.randn(1, 3, 64, requires_grad=True)
    boxes = a.detach().float().unsqueeze(0).requires_grad_()
    anchors = torch.tensor([[1., 1.], [6., 5.], [11., 11.]])
    scores = torch.tensor([[[.8], [.5], [.2]]])
    fg = torch.tensor([[True, True, True]])
    args = (dist, boxes, anchors, target.float().unsqueeze(0), scores, scores.sum(), fg)
    new_result = BboxLoss(16, loss_type='ciou')(*args)
    old_result = old.BboxLoss(16)(*args)
    for x, y in zip(new_result, old_result, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    ng = torch.autograd.grad(sum(new_result), (dist, boxes), retain_graph=True)
    og = torch.autograd.grad(sum(old_result), (dist, boxes))
    for x, y in zip(ng, og, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        assert torch.isfinite(x).all()
    result = {'status': 'passed', 'historical_focal_forward_and_gradient_exact': True,
              'historical_ciou_and_dfl_forward_and_gradient_exact': True, 'utc': now()}
    write(root / 'numerical_checks.json', result)
    return result


def smoke(root):
    p = protocol(root)
    source_checks(root)
    numerical_checks(root)
    jobs = [j for j in p['jobs'] if j['seed'] == 42 and
            (j['dataset'] == 'WSODD' or j['arm'] == 'refsoft')]
    runs = []
    for job in jobs:
        completion = root / 'smoke_runs' / job['key'] / 'completion.json'
        if not completion.exists():
            child(root, 'train.py', job, True)
        done = load(completion)
        assert done['status'] == 'completed' and done['smoke'] and done['epochs_completed'] == 2
        assert done['optimizer_parameter_changed']
        child(root, 'evaluate.py', job, True)
        runs.append(done)
    ws = {r['job']['arm']: r['runtime'] for r in runs if r['job']['dataset'] == 'WSODD'}
    assert len({ws[x]['initial_state_sha256'] for x in ['baseline', 'weights_only']}) == 1
    assert len({ws[x]['initial_state_sha256'] for x in ['bundle', 'duplicate', 'refsoft']}) == 1
    assert ws['duplicate']['train_images'] == ws['refsoft']['train_images']
    write(root / 'smoke_completed.json', {'status': 'passed', 'utc': now(), 'runs': [r['job']['key'] for r in runs],
            'protocol_sha256': sha(root / 'protocol.json'), 'numerical_checks': load(root / 'numerical_checks.json'),
            'paired_initial_states_identical': True, 'real_training_and_evaluation_completed': True})
    state(root, 'ready', smoke_runs=len(runs), maximum_full_runs=p['maximum_full_runs'])


def report(root):
    p = protocol(root)
    reports = {}
    raw = []
    for job in p['jobs']:
        path = root / 'evaluation/test' / (job['key'] + '__best.json')
        if not path.exists():
            continue
        item = load(path)
        reports[job['key']] = item['summary']
        done = load(root / 'runs' / job['key'] / 'completion.json')
        secondary = load(root / 'evaluation/test' / (job['key'] + '__best_map.json'))['summary']
        raw.append({**job, **item['summary'], 'peak_val_map': done['peak_val_map'],
                    'epochs': done['epochs_completed'], 'optimizer_steps': done['optimizer_step_calls'],
                    'secondary_test_mAP50-95': secondary['mAP50-95']})
    pairs = [('loss_weights', 'weights_only', 'baseline'), ('cbam_focal_bundle', 'bundle', 'weights_only'),
             ('duplicate_package', 'duplicate', 'bundle'), ('reflection_only', 'refsoft', 'duplicate'),
             ('augmentation_total', 'refsoft', 'bundle'), ('full_package', 'refsoft', 'baseline')]
    effects = []
    for dataset in p['datasets']:
        for effect, higher, lower in pairs:
            values = []
            for seed in p['seeds']:
                a, b = f'{dataset}_{higher}_seed{seed}', f'{dataset}_{lower}_seed{seed}'
                if a in reports and b in reports:
                    values.append({'seed': seed, **{m: None if reports[a][m] is None or reports[b][m] is None else
                        100*(reports[a][m]-reports[b][m]) for m in ['mAP50-95', 'mAP50', 'APsmall']}})
            if values:
                stats = {}
                for m in ['mAP50-95', 'mAP50', 'APsmall']:
                    v = [x[m] for x in values if x[m] is not None]
                    stats[m] = {'mean_pp': statistics.mean(v) if v else None,
                                'sample_sd_pp': statistics.stdev(v) if len(v)>1 else None,
                                'positive_seeds': sum(x>0 for x in v), 'n': len(v)}
                effects.append({'dataset': dataset, 'effect': effect, 'per_seed_pp': values, 'stats': stats})
    complete = len(raw) == len(p['jobs'])
    write(root / 'summary.json', {'status': 'completed' if complete else 'partial', 'utc': now(),
                                 'evaluated_runs': len(raw), 'expected_runs': len(p['jobs']),
                                 'effects': effects, 'raw': raw})
    if raw:
        with (root / 'paired_metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
    lines = ['# 旧 ref_soft 复验', '', f'已完成固定 test 评价 {len(raw)}/{len(p["jobs"])}。', '',
             '单位为百分点；不足三个 seed 的结果仅为中间记录。reflection_only = refsoft − duplicate。', '',
             '| 数据集 | 差值项 | 配对 seed 数 | mAP50-95 均值 | 样本标准差 | 正向 seed |',
             '|---|---|---:|---:|---:|---:|']
    for e in effects:
        s = e['stats']['mAP50-95']
        sd = f'{s["sample_sd_pp"]:.4f}' if s['sample_sd_pp'] is not None else 'NA'
        lines.append(f'| {e["dataset"]} | {e["effect"]} | {s["n"]} | {s["mean_pp"]:+.4f} | {sd} | {s["positive_seeds"]} |')
    lines += ['', '原 val 峰值、固定 test、APsmall 和 best_map 检查点敏感性详见 paired_metrics.csv / summary.json。',
              'loss_weights 与 cbam_focal_bundle 分开；duplicate_package 包含重采样、额外步数与重编码，不能单独解释为训练时长。',
              '当前协议未重复旧数据划分，不能据此唯一判定旧增长由数据划分或环境造成。']
    (root / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return complete


def queue(root):
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = protocol(root)
    smoke_done = load(root / 'smoke_completed.json')
    assert smoke_done['status'] == 'passed' and smoke_done['protocol_sha256'] == sha(root / 'protocol.json')
    source_checks(root)
    event(root, 'queue_started', pid=os.getpid(), maximum_full_runs=p['maximum_full_runs'])
    for index, job in enumerate(p['jobs']):
        if (root / 'STOP').exists():
            raise InterruptedError('User STOP sentinel')
        if shutil.disk_usage(root).free < 8*1024**3:
            raise RuntimeError('Less than 8 GiB free disk; queue stopped')
        # No silent resumes or failed-run replacement. Completed work has frozen fingerprints.
        completion = root / 'runs' / job['key'] / 'completion.json'
        if not completion.exists():
            child(root, 'train.py', job)
        done = load(completion)
        assert done['status'] == 'completed' and not done['smoke']
        assert done['protocol_sha256'] == sha(root / 'protocol.json')
        # Matched models must start from identical tensors for a paired seed/dataset.
        family = ['baseline', 'weights_only'] if job['arm'] in ['baseline', 'weights_only'] else ['bundle', 'duplicate', 'refsoft']
        hashes = []
        for arm in family:
            runtime = root / 'runs' / f'{job["dataset"]}_{arm}_seed{job["seed"]}' / 'runtime.json'
            if runtime.exists():
                hashes.append(load(runtime)['initial_state_sha256'])
        assert len(set(hashes)) == 1, ('Unmatched initialization', job)
        child(root, 'evaluate.py', job)
        event(root, 'job_completed', key=job['key'], completed=index+1, total=len(p['jobs']))
        report(root)
    assert report(root)
    state(root, 'completed', completed_runs=len(p['jobs']))
    event(root, 'queue_completed', runs=len(p['jobs']))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=['smoke', 'queue', 'report'])
    a.add_argument('--root', type=Path, required=True)
    args = a.parse_args()
    try:
        {'smoke': smoke, 'queue': queue, 'report': report}[args.action](args.root)
    except Exception as exc:
        state(args.root, 'failed', action=args.action, error=repr(exc), traceback=traceback.format_exc())
        event(args.root, 'failed', action=args.action, error=repr(exc))
        raise
