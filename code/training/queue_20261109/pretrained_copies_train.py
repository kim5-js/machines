"""Pretrained fine-tuning with appended clean copies or GlareSoft copies (WSODD), on Vast.

Recipe: the standard-training (E0) recipe of formal_wsodd_sequence_e0_seeds42_123_3407, reproduced by its
core call: YOLO(pretrained yolov8s.pt).train(trainer=MainlineSGDTrainer, **training_kwargs) with the E0
variant (no MDG). training_kwargs are the ones recorded in the original seed-42 run metadata; only data,
seed, project and name change. Training code: the archived historical modules, byte-identical to the E0
release (checked by hash before training). Data: the matched-control overlays, i.e. the same 397 seed-42
selected images as clean copies or GlareSoft copies, 4374 training images in both arms; validation unchanged.
After training, args.yaml must equal the original E0 seed-42 args.yaml except data, seed, project, name, save_dir.
Usage: python pretrained_copies_train.py <arm: clean|glaresoft> <seed> [smoke]
"""
import hashlib, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone

A = Path('/root/autodl-tmp')
EXT = A / 'pretrained_copies_20261109'
HIST = A / 'flow_source_retrain_20260920/release_v1/historical_code'
VENDOR = A / 'flow_source_retrain_20260920/release_v1/vendor'
OVERLAY = {'clean': A / 'refsoft_reproduction_20260910/release_v1/overlays/WSODD/duplicate/dataset.yaml',
           'glaresoft': A / 'refsoft_reproduction_20260910/release_v1/overlays/WSODD/refsoft/dataset.yaml'}
arm, seed = sys.argv[1], int(sys.argv[2])
smoke = len(sys.argv) > 3 and sys.argv[3] == 'smoke'
assert arm in OVERLAY
ref = json.loads((EXT / 'reference_e0_seed42.json').read_text())


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, msg):
    if not ok: raise RuntimeError(msg)


for name, digest in ref['code_sha256'].items():
    require(sha(HIST / name) == digest, 'Historical code differs from the E0 release: ' + name)
require(sha(ref['weights_path']) == ref['weights_sha256'], 'Pretrained weight changed')
sys.dont_write_bytecode = True
os.environ['YOLO_CONFIG_DIR'] = str(EXT / 'settings'); os.environ['MPLCONFIGDIR'] = str(EXT / 'mpl')
sys.path[:0] = [str(VENDOR), str(HIST)]
import ultralytics, torch, yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import train_pretrained_sgd_mainline as impl  # noqa: E402
require(ultralytics.__version__ == '8.4.124' and str(VENDOR) in ultralytics.__file__, 'Wrong Ultralytics')

key = f'WSODD_pretrained_{arm}_seed{seed}'
project = EXT / ('smoke_runs' if smoke else 'runs')
run = project / key
require(not run.exists(), 'Existing run preserved: ' + str(run))
kwargs = dict(ref['training_kwargs'])
kwargs.update(data=str(OVERLAY[arm]), seed=seed, project=str(project), name=key)
if smoke:
    kwargs.update(epochs=2, close_mosaic=0)
variant = dict(ref['variant'], key=f'pretrained_{arm}', name=key, seed=seed)
require(variant['mdg_mode'] == 'none' and not variant['dual_view'] and not variant['degraded_small_weighting'], 'Variant must be E0')
os.environ['RING3_SGD_MAINLINE_VARIANT_JSON'] = json.dumps(variant, sort_keys=True)
os.environ['RING3_SGD_MAINLINE_MDG_CONFIG'] = str(HIST / 'e1_mdg_config.json')
os.environ['RING3_SGD_MAINLINE_MDG_SHA256'] = sha(HIST / 'e1_mdg_config.json')
started = time.monotonic()
model = YOLO(ref['weights_path'])
info = {}


def on_start(trainer):
    # Fail at once if Ultralytics halved the batch after an out-of-memory start (seen on a crowded GPU).
    require(int(trainer.args.batch) == int(ref['training_kwargs']['batch']) and int(trainer.batch_size) == int(ref['training_kwargs']['batch']),
            f'Batch changed at start: args {trainer.args.batch}, trainer {trainer.batch_size}')
    info.update(train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                trainer=type(trainer).__name__, optimizer=type(trainer.optimizer).__name__, amp=bool(trainer.amp))


model.add_callback('on_train_start', on_start)
model.train(trainer=impl.MainlineSGDTrainer, **kwargs)
args = yaml.safe_load((run / 'args.yaml').read_text())
skip = {'data', 'seed', 'project', 'name', 'save_dir'} | ({'epochs', 'close_mosaic'} if smoke else set())
diff = {k: (args.get(k), ref['args'].get(k)) for k in set(args) | set(ref['args']) if k not in skip and str(args.get(k)) != str(ref['args'].get(k))}
require(not diff, 'Arguments differ from the original E0 run: ' + repr(diff))
require(info.get('train_images') == 4374 and info.get('optimizer') == 'SGD' and info.get('amp'), 'Unexpected training setup: ' + repr(info))
for name in ('best.pt', 'last.pt'):
    require((run / 'weights' / name).exists(), 'Missing ' + name)
(run / 'completion.json').write_text(json.dumps(dict(status='completed', utc=datetime.now(timezone.utc).isoformat(), arm=arm, seed=seed, smoke=smoke,
    info=info, ultralytics=ultralytics.__file__, torch=torch.__version__, elapsed_seconds=time.monotonic() - started,
    best_sha256=sha(run / 'weights/best.pt'), last_sha256=sha(run / 'weights/last.pt')), indent=2))
print('PRETRAINED_COPIES_COMPLETED', key, flush=True)
