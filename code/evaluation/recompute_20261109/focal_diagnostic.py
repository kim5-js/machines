"""Diagnostic for the localization-loss variant in Equation (1). Produces no model and no paper result.

Part A: autograd unit test on synthetic boxes, against the unclipped form and the published IoU^0.5 form.
Part B: how often positives fall into the clipped region on real WSODD training batches, using the
matched-block vendor code and recipe on CPU, so the GPU evaluation queue is not disturbed:
  - from random initialization through the first iterations (optimizer steps are discarded on exit)
  - with last.pt of the attention + localization configuration at learning rate zero
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import collections, os, shutil, traceback

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
bootstrap('matched')
import torch
import ultralytics
import ultralytics.utils.loss as L
from ultralytics import YOLO

DIAG = OUT / 'focal_diag'
DIAG.mkdir(parents=True, exist_ok=True)
require(ultralytics.__version__ == '8.3.182', 'Wrong vendor ' + ultralytics.__version__)
torch.set_num_threads(4)


def part_a():
    loss = L.FocalEIoULoss(alpha=0.25, gamma=2.0, eps=1e-7, reduction='none')
    rows = []
    gt = torch.tensor([[-10., -10., 10., 10.]], dtype=torch.float64)
    for ratio in (1.0, 1.2, 1.5, 2, 3, 3.5, 3.7, 4, 5, 8, 12):
        half = 10.0 * ratio
        pred = torch.tensor([[-half, -half, half, half]], dtype=torch.float64, requires_grad=True)
        val = loss(pred, gt); val.sum().backward()
        g = pred.grad.norm().item()
        p2 = pred.detach().clone().requires_grad_(True)
        iou = L._iou_xyxy(p2, gt, 1e-7); e = L._efficient_iou_xyxy(p2, gt, 1e-7)
        unclipped = ((0.25 + 0.75 * (1 - iou) ** 2) * (1 - e)).sum(); unclipped.backward()
        p3 = pred.detach().clone().requires_grad_(True)
        i3 = L._iou_xyxy(p3, gt, 1e-7); e3 = L._efficient_iou_xyxy(p3, gt, 1e-7)
        published = (i3 ** 0.5 * (1 - e3)).sum(); published.backward()
        rows.append(dict(case='concentric', pred_to_gt_side=ratio, iou=iou.item(), eiou=e.item(), clipped=e.item() < -1, weight_implemented=(0.25 + 0.75 * (1 - iou.item()) ** 2), weight_published_gamma05=iou.item() ** 0.5, grad_implemented=g, grad_unclipped=p2.grad.norm().item(), grad_published=p3.grad.norm().item()))
    for dx, w in ((30.0, 20.0), (60.0, 20.0), (60.0, 80.0), (200.0, 80.0), (200.0, 400.0)):
        pred = torch.tensor([[dx - w / 2, -w / 2, dx + w / 2, w / 2]], dtype=torch.float64, requires_grad=True)
        val = loss(pred, gt); val.sum().backward()
        p2 = pred.detach().clone().requires_grad_(True)
        iou = L._iou_xyxy(p2, gt, 1e-7); e = L._efficient_iou_xyxy(p2, gt, 1e-7)
        ((0.25 + 0.75 * (1 - iou) ** 2) * (1 - e)).sum().backward()
        rows.append(dict(case=f'offset dx={dx} side={w}', iou=iou.item(), eiou=e.item(), clipped=e.item() < -1, weight_implemented=(0.25 + 0.75 * (1 - iou.item()) ** 2), grad_implemented=pred.grad.norm().item(), grad_unclipped=p2.grad.norm().item()))
    write(DIAG / 'unit_test.json', rows)
    for r in rows:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


class Stop(Exception):
    pass


stats = []
current = {'phase': None}
orig = L.FocalEIoULoss.forward


def wrapped(self, pred, target):
    with torch.no_grad():
        p = pred.view(-1, 4); t = target.view(-1, 4)
        iou = L._iou_xyxy(p, t, self.eps); e = L._efficient_iou_xyxy(p, t, self.eps)
        w = self.alpha + (1 - self.alpha) * (1 - iou).clamp(min=0) ** self.gamma
        base = (1 - e.clamp(-1, 1)).clamp(min=0)
        total = float((w * base).sum()) or 1.0
        clipped = e < -1
        stats.append(dict(phase=current['phase'], positives=int(p.shape[0]), clipped=int(clipped.sum()), no_overlap=int((iou <= 0).sum()), no_overlap_and_clipped=int(((iou <= 0) & clipped).sum()), loss_share_clipped=float((w * base)[clipped].sum()) / total, mean_iou=float(iou.mean()) if p.shape[0] else None, mean_weight=float(w.mean()) if p.shape[0] else None, mean_published_weight=float((iou.clamp(min=0) ** 0.5).mean()) if p.shape[0] else None))
    return orig(self, pred, target)


def training_data():
    y = DIAG / 'data.yaml'
    if y.exists():
        return y
    src = RING3 / '02_当前实验/数据集原始包/WSODD USV_dataset/WSODD USV_dataset'
    names = [Path(x).name for x in (AUTODL_CODE / 'manifests/wsodd_sequence_disjoint_v1/train.txt').read_text(encoding='utf-8').split()]
    img = DIAG / 'data/images/train'; lab = DIAG / 'data/labels/train'
    img.mkdir(parents=True, exist_ok=True); lab.mkdir(parents=True, exist_ok=True)
    for n in names:
        stem = Path(n).stem
        label = next(p for p in (src / 'train/labels' / (stem + '.txt'), src / 'val/labels' / (stem + '.txt')) if p.exists())
        if not (img / n).exists():
            os.link(src / 'image' / n, img / n)
        if not (lab / (stem + '.txt')).exists():
            os.link(label, lab / (stem + '.txt'))
    gt = read(OUT / 'data/WSODD/ground_truth.json')
    cls = {c['id'] - 1: c['name'] for c in gt['categories']}
    text = 'path: ' + str(DIAG / 'data').replace('\\', '/') + '\ntrain: images/train\nval: images/train\nnames:\n' + ''.join(f'  {k}: {v}\n' for k, v in cls.items())
    y.write_text(text, encoding='utf-8')
    return y


def run(phase, model_source, iterations, seed, **extra):
    current['phase'] = phase
    recipe = dict(read(MATCHED_RELEASE / 'historical/recipes.json')['wsodd_refsoft'])
    for k in ('save_dir', 'model'):
        recipe.pop(k, None)
    recipe.update(data=str(training_data()), seed=seed, device='cpu', workers=0, amp=False, save=False, val=False, plots=False, verbose=False, exist_ok=True, project=str(DIAG / 'runs'), name=phase, cache=False)
    recipe.update(extra)
    model = YOLO(model_source)
    count = {'n': 0}

    def stop(trainer):
        count['n'] += 1
        if count['n'] % 25 == 0:
            write(DIAG / f'progress_{phase}.json', dict(iterations=count['n'], target=iterations))
            write(DIAG / 'stats.json', stats)
        if count['n'] >= iterations:
            raise Stop()

    model.add_callback('on_train_batch_end', stop)
    try:
        model.train(**recipe)
    except Stop:
        pass
    write(DIAG / 'stats.json', stats)
    shutil.rmtree(DIAG / 'runs' / phase, ignore_errors=True)


def summarize():
    agg = collections.defaultdict(lambda: collections.Counter())
    means = collections.defaultdict(list)
    for s in stats:
        if not s['positives']:
            continue
        k = s['phase']
        for f in ('positives', 'clipped', 'no_overlap', 'no_overlap_and_clipped'):
            agg[k][f] += s[f]
        means[k].append(s)
    out = {}
    for k, c in agg.items():
        ms = means[k]
        out[k] = dict(iterations=len(ms), positives=c['positives'], clipped_pct=100 * c['clipped'] / c['positives'], zero_gradient_pct=100 * c['no_overlap_and_clipped'] / c['positives'], no_overlap_pct=100 * c['no_overlap'] / c['positives'],
                      mean_loss_share_clipped_pct=100 * sum(m['loss_share_clipped'] for m in ms) / len(ms), mean_iou=sum(m['mean_iou'] for m in ms) / len(ms), mean_weight=sum(m['mean_weight'] for m in ms) / len(ms), mean_published_weight=sum(m['mean_published_weight'] for m in ms) / len(ms))
    write(DIAG / 'summary.json', out)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    try:
        part_a()
        L.FocalEIoULoss.forward = wrapped
        # Resume: keep phases already recorded in full; a partly recorded phase is rerun.
        if (DIAG / 'stats.json').exists():
            previous = read(DIAG / 'stats.json')
            counts = collections.Counter(s['phase'] for s in previous)
            stats.extend(s for s in previous if counts[s['phase']] >= (150 if s['phase'].startswith('from_init') else 50))
        done = {s['phase'] for s in stats}
        yaml_model = str(MATCHED_RELEASE / 'historical/yolov8s_cbam.yaml')
        if 'from_init_seed0' not in done:
            run('from_init_seed0', yaml_model, 150, 0)
        for seed, eid_path in read(DIAG / 'last_weights.json').items():
            if f'final_state_seed{seed}' in done:
                continue
            # Learning rate and weight decay are zero, so no parameter changes; momentum stays nonzero only because
            # the vendor SGD uses Nesterov. BatchNorm runs in training mode, as it does during real training.
            run(f'final_state_seed{seed}', eid_path, 50, int(seed), lr0=0.0, lrf=1.0, warmup_epochs=0.0, weight_decay=0.0, mosaic=0.7, close_mosaic=0)
        summarize()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        write(DIAG / 'stats.json', stats)
        raise
