"""Shared settings for the 20261109 local recomputation. Inference and scoring only; no training.

Inputs are read-only: the sealed 20260924 evidence, the AutoDL backup and the original datasets.
Every output goes under OUT. Severity conditions are fixed here before any new result is seen.
"""
from pathlib import Path
import hashlib, json, os, sys

RING3 = Path('D:/ring3')
OUT = RING3 / '02_当前实验/原始证据/recompute_20261109'
EVIDENCE_ZIP = RING3 / '01_当前论文/生成脚本/20260924_统一评价/unified_evaluation_evidence.zip'
UNIFIED_CSV = RING3 / '01_当前论文/基础依赖/20260924_统一评价修订/unified_evaluations.csv'
BACKUP = RING3 / '05_备份/AutoDL_full_backup_20260928'
AUTODL_CODE = RING3 / '02_当前实验/代码/autodl'
MDG_CONFIG = AUTODL_CODE / 'e1_mdg_config.json'
DEPS = RING3 / 'environments/recompute_20261109/site'
FLOW_BUNDLE = RING3 / '02_当前实验/原始证据/flow_source_retrain_20260920/execution_bundle_v1'
MATCHED_RELEASE = RING3 / '02_当前实验/原始证据/refsoft_reproduction_20260910/release'
WSODD_IMAGES = RING3 / '02_当前实验/数据集原始包/WSODD USV_dataset/WSODD USV_dataset/image'
SDS_ROOT = RING3 / '02_当前实验/数据集原始包/SeaDronesSee Object Detection v2/Compressed Version/images'
SDS_VAL_TSV = AUTODL_CODE / 'manifests/seadronessee_source_disjoint_v1/val.tsv'

# Values recorded by the sealed 20260924 evaluation.
SEALED_COCOEVAL_SHA = '988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267'
SEALED_TRANSFORM_SHA = 'cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f'
SEALED_MDG_SHA = '41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a'
NAMESPACE_ROOT = 'ring3/cross-dataset/e0-e1/controlled-strong/v1'
INFERENCE = dict(imgsz=960, batch=1, half=False, rect=False, conf=0.001, iou=0.7, max_det=300, device=0)
SEEDS_ALL = {'WSODD': ['0', '1', '2', '42', '123', '3407'], 'SeaDronesSee': ['0', '1', '2']}

# Fixed in advance. L1/L2/G1/G2 are the training-band midpoints; L4/L5/G4/G5 extend beyond training.
SEVERITY = {
    'low_light_L1': dict(kind='low', brightness=0.925, gamma=1.085, basis='training band 1 midpoint'),
    'low_light_L2': dict(kind='low', brightness=0.88, gamma=1.165, basis='training band 2 midpoint'),
    'low_light_L4': dict(kind='low', brightness=0.70, gamma=1.50, basis='beyond training range'),
    'low_light_L5': dict(kind='low', brightness=0.50, gamma=2.00, basis='beyond training range'),
    'glare_G1': dict(kind='glare', area=0.02275, intensity=26.0, basis='training band 1 midpoint'),
    'glare_G2': dict(kind='glare', area=0.0335, intensity=34.5, basis='training band 2 midpoint'),
    'glare_G4': dict(kind='glare', area=0.08, intensity=80.0, basis='beyond training range'),
    'glare_G5': dict(kind='glare', area=0.15, intensity=120.0, basis='beyond training range'),
}
GLARE_GEOMETRY = {'aspect_ratio': [1.4, 3.2], 'softness': [1.6, 2.4]}  # midpoints 2.3 and 2.0, as in the sealed test


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def bootstrap(family='standard'):
    """Same import order as the sealed evaluator, except the Windows pycocotools build comes first."""
    sys.dont_write_bytecode = True
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'settings' / family)
    os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
    paths = [str(DEPS), str(FLOW_BUNDLE / 'vendor'), str(FLOW_BUNDLE / 'historical_code'), str(AUTODL_CODE)]
    if family == 'matched':
        paths.insert(1, str(MATCHED_RELEASE / 'vendor'))
    sys.path[:0] = paths
