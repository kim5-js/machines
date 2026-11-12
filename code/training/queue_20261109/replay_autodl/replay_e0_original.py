"""Replay of the September WSODD standard-training run E0 seed 42 (server side).

Same core call as pretrained_copies_train.py, which was written to reproduce the E0 recipe:
YOLO(pretrained yolov8s.pt).train(trainer=MainlineSGDTrainer, **training_kwargs) with the E0 variant.
Here data and seed are the original ones (no appended copies), so the first epochs can be compared with the
September results.csv of that run. Only project and name differ; the output goes to the replay folder.
Usage: python replay_e0_original.py
"""
import hashlib, json, os, sys
from pathlib import Path

A = Path('/root/autodl-tmp')
EXT = A / 'pretrained_copies_20261109'
OUT = A / 'replay_20261111'
HIST = A / 'flow_source_retrain_20260920/release_v1/historical_code'
VENDOR = A / 'flow_source_retrain_20260920/release_v1/vendor'
ref = json.loads((EXT / 'reference_e0_seed42.json').read_text())


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, msg):
    if not ok: raise RuntimeError(msg)


for name, digest in ref['code_sha256'].items():
    require(sha(HIST / name) == digest, 'Historical code differs from the E0 release: ' + name)
require(sha(ref['weights_path']) == ref['weights_sha256'], 'Pretrained weight changed')
sys.dont_write_bytecode = True
os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'settings'); os.environ['MPLCONFIGDIR'] = str(OUT / 'mpl')
sys.path[:0] = [str(VENDOR), str(HIST)]
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import train_pretrained_sgd_mainline as impl  # noqa: E402
require(ultralytics.__version__ == '8.4.124' and str(VENDOR) in ultralytics.__file__, 'Wrong Ultralytics')

key = 'WSODD_E0_seed42_original_recipe'
project = OUT / 'runs'
require(not (project / key).exists(), 'Existing run preserved')
kwargs = dict(ref['training_kwargs'])  # data and seed are the original ones
require(kwargs['seed'] == 42 and kwargs['data'].endswith('wsodd_sequence_disjoint_v1.yaml'), 'Unexpected reference record')
kwargs.update(project=str(project), name=key)
variant = dict(ref['variant'])
require(variant['mdg_mode'] == 'none' and not variant['dual_view'] and not variant['degraded_small_weighting'], 'Variant must be E0')
os.environ['RING3_SGD_MAINLINE_VARIANT_JSON'] = json.dumps(variant, sort_keys=True)
os.environ['RING3_SGD_MAINLINE_MDG_CONFIG'] = str(HIST / 'e1_mdg_config.json')
os.environ['RING3_SGD_MAINLINE_MDG_SHA256'] = sha(HIST / 'e1_mdg_config.json')
model = YOLO(ref['weights_path'])


def on_start(trainer):
    require(int(trainer.args.batch) == int(kwargs['batch']) and int(trainer.batch_size) == int(kwargs['batch']), 'Batch changed at start')
    print('REPLAY_E0_START', json.dumps(dict(train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                                              trainer=type(trainer).__name__, optimizer=type(trainer.optimizer).__name__, amp=bool(trainer.amp))), flush=True)


model.add_callback('on_train_start', on_start)
model.train(trainer=impl.MainlineSGDTrainer, **kwargs)
