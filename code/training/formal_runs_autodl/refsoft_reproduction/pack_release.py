"""Package the retained historical source without editing it (run on Windows)."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

REPO = Path('D:/ring3')
HERE = Path(__file__).resolve().parent
OLD = REPO / 'environments/cbam_eiou_light_glare_20260506'
SITE = OLD / 'yolo_cbam_eiou_light_glare_20260506/Lib/site-packages'
sys.path.insert(0, str(SITE))
import yaml


def main():
    out = REPO / '02_当前实验/原始证据/refsoft_reproduction_20260910'
    staging = out / 'release'
    staging.mkdir(parents=True, exist_ok=True)
    origins = {}

    def copy(src, rel):
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        origins[rel] = str(src)

    for p in sorted((SITE / 'ultralytics').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}:
            copy(p, 'vendor/ultralytics/' + p.relative_to(SITE / 'ultralytics').as_posix())
    for dependency in ['tqdm', 'cpuinfo']:
        assert (SITE / dependency).is_dir(), dependency
        for p in sorted((SITE / dependency).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}:
                copy(p, 'vendor/' + dependency + '/' + p.relative_to(SITE / dependency).as_posix())
    for name in ['light_glare_dataset.py', 'yolov8s_cbam.yaml', 'run_stage1_960_only2.py', 'requirements.txt']:
        copy(OLD / name, 'historical/' + name)
    for name in ['custom_losses/focal_eiou.py', 'custom_losses/yolo_adapters.py', 'custom_losses/__init__.py']:
        copy(OLD / name, 'historical/' + name)
    archive = REPO / '03_历史归档/实验/训练总存档'
    specs = {
        'baseline': 'baseline/WSODD_USV/stage1_960_20260328_045554/args.yaml',
        'bundle': 'cbam_eiou/WSODD_USV/stage1_960_20260328_045554/args.yaml',
        'wsodd_refsoft': 'cbam_eiou_light_glare/WSODD_USV/stage1_960_wsodd_lg_sweep_r010_ref_soft_20260510_041602/args.yaml',
        'flow_refsoft': 'cbam_eiou_light_glare/FloW_IMG/stage1_960_flow_ref_soft_r010_20260510_170537/args.yaml',
    }
    recipes = {}
    for name, path in specs.items():
        copy(archive / path, f'historical/{name}_args.yaml')
        recipes[name] = yaml.safe_load((archive / path).read_text(encoding='utf-8'))
    (staging / 'historical/recipes.json').write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding='utf-8')
    # Exact historical baseline loss used for an independent numerical/gradient equivalence check.
    baseline = REPO / 'environments/baseline_20251112/yolo_baseline_20251112/Lib/site-packages/ultralytics'
    copy(baseline / 'utils/loss.py', 'historical/baseline_loss.py')
    comparisons = {}
    for rel in ['data/augment.py', 'engine/trainer.py', 'nn/modules/conv.py', 'utils/metrics.py']:
        comparisons[rel] = (baseline / rel).read_bytes() == (SITE / 'ultralytics' / rel).read_bytes()
    assert all(comparisons.values()), comparisons
    for name in ['evaluate_e3_coco_areas.py', 'evaluate_pretrained_sgd_robustness.py']:
        copy(REPO / '02_当前实验/代码/autodl' / name, 'evaluation_code/' + name)
    for p in sorted(HERE.glob('*')):
        if p.is_file() and p.name != 'pack_release.py':
            copy(p, p.name)
    inventory = []
    for p in sorted(staging.rglob('*')):
        if p.is_file() and p.name != 'release_manifest.json':
            rel = p.relative_to(staging).as_posix()
            inventory.append({'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source': origins.get(rel, 'generated from archived YAML')})
    (staging / 'release_manifest.json').write_text(json.dumps({'schema': 'refsoft.release.v1', 'historical_unchanged_common_files': comparisons, 'files': inventory}, ensure_ascii=False, indent=2), encoding='utf-8')
    zpath = out / 'release.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(staging).as_posix())
    print(json.dumps({'zip': str(zpath), 'sha256': hashlib.sha256(zpath.read_bytes()).hexdigest(), 'files': len(inventory), 'bytes': zpath.stat().st_size}))


if __name__ == '__main__':
    main()
