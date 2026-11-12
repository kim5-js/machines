"""FloW extension runs on Vast, through the archived flow_source_retrain_20260920 adapter, unchanged.

The archived release is mirrored at its original path, so all of its hash gates (request, staging and
adapter manifests, preflight, data records, pretrained weights, runtime versions) run as they are.
This wrapper only adds jobs and one arm:
  FloW_E0/E2/E3 for seeds 42, 123, 3407 (same arms, new seeds)
  FloW_E2C for seeds 0, 1, 2, 42, 123, 3407: the E2 paired-view run with the second view replaced by an
  exact copy of the clean view after the original preprocessing. MDG is still drawn, so the random
  stream, data order, view count, optimizer updates and loss normalization match E2; only the content
  of the second view differs.
Usage: python flow_ext_train.py <job_key> [full|smoke]
"""
import json, os, sys
from pathlib import Path

ROOT = '/root/autodl-tmp/flow_source_retrain_20260920/release_v1'
EXT = Path('/root/autodl-tmp/flow_extension_20261109')
os.environ['RING3_LEGACY_RELEASE_ROOT'] = ROOT
sys.path.insert(0, ROOT + '/adapter')
import common as c  # noqa: E402

EXTRA = [dict(key=f'FloW_{arm}_seed{s}', dataset='FloW', seed=s, arm=arm, model_key='yolov8s', family='ablation')
         for s in (42, 123, 3407) for arm in ('E0', 'E2', 'E3')]
EXTRA += [dict(key=f'FloW_E2C_seed{s}', dataset='FloW', seed=s, arm='E2C', model_key='yolov8s', family='ablation')
          for s in (0, 1, 2, 42, 123, 3407)]
_job_list, _variant = c.job_list, c.variant
c.job_list = lambda req: _job_list(req) + [dict(j) for j in EXTRA]


def variant(job):
    v = _variant(job)
    if job['arm'] == 'E2C':
        v = dict(v, key='e2c')
        assert v['mdg_mode'] == 'dual_view' and v['dual_view'] and not v['degraded_small_weighting']
    return v


c.variant = variant
key = sys.argv[1]
phase = sys.argv[2] if len(sys.argv) > 2 else 'full'
evidence = {'batches': 0, 'second_view_replaced': 0, 'mdg_draws_kept': True}
if key.startswith('FloW_E2C_'):
    sys.path[:0] = [ROOT + '/vendor', ROOT + '/historical_code']
    import torch
    import train_pretrained_sgd_mainline as tpm
    original = tpm.MainlineSGDTrainer.preprocess_batch

    def preprocess_batch(self, batch):
        out = original(self, batch)  # includes the MDG draw for the second view
        n = out['img'].shape[0] // 2
        assert out['img'].shape[0] == 2 * n and n > 0
        out['img'][n:] = out['img'][:n].clone()
        assert torch.equal(out['img'][n:], out['img'][:n])
        evidence['batches'] += 1; evidence['second_view_replaced'] += 1
        if evidence['batches'] % 500 == 1:
            run = Path(ROOT) / ('smoke_runs' if phase == 'smoke' else 'runs') / key
            if run.exists():
                (run / 'e2c_evidence.json').write_text(json.dumps(evidence))
        return out

    tpm.MainlineSGDTrainer.preprocess_batch = preprocess_batch
import train  # noqa: E402  archived adapter/train.py
train.main(key, phase)
if key.startswith('FloW_E2C_'):
    run = Path(ROOT) / ('smoke_runs' if phase == 'smoke' else 'runs') / key
    assert evidence['batches'] > 0 and evidence['second_view_replaced'] == evidence['batches']
    (run / 'e2c_evidence.json').write_text(json.dumps(dict(evidence, status='completed')))
