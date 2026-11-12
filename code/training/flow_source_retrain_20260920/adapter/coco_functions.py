from pathlib import Path
import contextlib,io
import common as c
ROOT=c.ROOT
write=c.write

def coco_evaluate(gt_path, predictions, output):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gt = COCO(str(gt_path))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(gt.imgs)
        ev.params.catIds = sorted(gt.cats)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'coco_eval.log').write_text(out.getvalue())
        np.savez_compressed(output / 'coco_arrays.npz', precision=ev.eval['precision'], recall=ev.eval['recall'], scores=ev.eval['scores'])
    names = ['mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
    metrics = {key: None if value < 0 else float(value) for key, value in zip(names, ev.stats)}
    per_class = {}
    for k, cat in enumerate(ev.params.catIds):
        values = ev.eval['precision'][:, :, k, 0, -1]
        values = values[values >= 0]
        per_class[gt.cats[cat]['name']] = float(values.mean()) if len(values) else None
    return metrics, per_class

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        p = Path(td) / 'gt.json'
        gt = {'info': {}, 'images': [{'id': 1, 'width': 200, 'height': 200}], 'categories': [{'id': 1, 'name': 'target'}, {'id': 2, 'name': 'absent'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 20, 20], 'area': 400, 'iscrowd': 0}]}
        write(p, gt)
        perfect, classes = coco_evaluate(p, [{'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 20, 20], 'score': 0.9}], None)
        empty, _ = coco_evaluate(p, [], None)
        wrong, _ = coco_evaluate(p, [{'image_id': 1, 'category_id': 2, 'bbox': [5, 5, 20, 20], 'score': 0.9}], None)
        assert perfect['mAP50-95'] > .99999 and perfect['APsmall'] > .99999
        assert perfect['APmedium'] is None and classes['absent'] is None
        assert empty['mAP50-95'] == 0 and wrong['mAP50-95'] == 0
    return {'status': 'passed', 'checks': ['perfect detection', 'missing area remains null', 'absent class remains null', 'empty detections', 'wrong category']}
