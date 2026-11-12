from common import CODE, now, sha, write, load, protocol
from pathlib import Path
import argparse
import math
from ultralytics import YOLO
import evaluate_pretrained_sgd_robustness as robust


def evaluate(root, key, smoke=False):
    p = protocol(root)
    job = next(x for x in p['jobs'] if x['key'] == key)
    run = root / ('smoke_runs' if smoke else 'runs') / key
    done = load(run / 'completion.json')
    evaluator = robust.load_evaluator(CODE / 'evaluation_code/evaluate_e3_coco_areas.py')
    evaluator.synthetic_self_test()
    for check in (['best.pt'] if smoke else ['best.pt', 'best_map.pt']):
        checkpoint = done['checkpoints'][check]
        assert sha(checkpoint['path']) == checkpoint['sha256']
        model = None
        try:
            for split in (['val'] if smoke else ['val', 'test'] if check == 'best.pt' else ['test']):
                if smoke:
                    listing = root / 'smoke_inputs' / job['dataset'] / 'none/val.txt'
                else:
                    listing = Path(p['datasets'][job['dataset']]['manifests'][split])
                images = [Path(x) for x in listing.read_text().splitlines()]
                dest = root / 'evaluation' / ('smoke' if smoke else split) / f'{key}__{Path(check).stem}.json'
                fingerprint = {'protocol_sha256': sha(root / 'protocol.json'), 'checkpoint': checkpoint,
                               'image_manifest_sha256': sha(listing), 'prediction': p['prediction'],
                               'split': split, 'images': len(images), 'checkpoint_rule': check}
                if dest.exists():
                    cached = load(dest)
                    assert cached['fingerprint'] == fingerprint and cached['status'] == 'completed'
                    continue
                if model is None:
                    model = YOLO(checkpoint['path'])
                identity = robust.identity_source_equivalence(model, images, p['prediction'])
                dets, truths, counts = robust.predict_dataset(model=model, image_paths=images,
                    transform=lambda im, _: im, evaluator=evaluator, prediction=p['prediction'])
                metrics = robust.evaluate_metrics(evaluator, dets, truths,
                    list(range(p['datasets'][job['dataset']]['classes'])), p['prediction']['evaluation_max_det'])
                summary = robust.metric_summary(metrics)
                assert math.isfinite(summary['mAP50-95'])
                write(dest, {'status': 'completed', 'utc': now(), 'fingerprint': fingerprint,
                             'summary': summary, 'full_metrics': metrics, 'counts': counts,
                             'identity_source_equivalence': identity})
        finally:
            robust.release_gpu(model)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--job', required=True)
    a.add_argument('--smoke', action='store_true')
    args = a.parse_args()
    evaluate(args.root, args.job, args.smoke)
