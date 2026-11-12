"""Package the raw training results behind revision 20261109 for the supervisor (no weights).

Per training run: the arguments written by the trainer, the per-epoch results, and the completion and runtime records.
For the 18 SDS and WSODD strategy runs, whose run folders are not stored locally, the arguments and the per-epoch
history are exported from the checkpoints (the weights themselves are not copied). Also the unified evaluation
records (one evaluation.json per checkpoint and condition) and the sealed table of the first 78 evaluations.
Not included: weights, predictions, plots and sample images, datasets, code.

Version 20261112 corrects version 20261111 in two places (the 20261111 folder and archive are kept unchanged):
  1. the four FloW detector reference runs are now the runs whose checkpoints the paper evaluates
     (flow_source_retrain_20260920 baseline_*_seed0); 20261111 had the records of earlier FloW runs of the same detectors;
  2. the 15 FloW pretrained single-component runs now have their per-run records; 20261111 had only the summary tables.
Every run folder is checked against the checkpoint path of the evaluation records before the package is written.

Output  02_当前实验/已验证交付/训练原始结果_20261112/  and  训练原始结果_20261112.zip  (+ .sha256.txt)
Run     python build_raw_results_package.py      (scientific environment: PyYAML, torch, ultralytics)
"""
from __future__ import annotations

import csv, hashlib, json, re, shutil, sys, zipfile
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3'); EVID = RING3 / '02_当前实验/原始证据'; CODE = RING3 / '02_当前实验/代码/autodl'
NAME = '训练原始结果_20261112'
OUT = RING3 / '02_当前实验/已验证交付' / NAME
ADDED = EVID / 'train_20261109/results/extracted/root/autodl-tmp'
KEEP = {'args.yaml', 'results.csv', 'completion.json', 'runtime_evidence.json', 'runtime.json', 'job.json', 'e2c_evidence.json'}
FLOW = {'E0': 'Standard training', 'E2': 'Paired views', 'E3': 'Paired views + small-object weighting', 'E2C': 'Identical paired views'}
MATCHED = {'baseline': 'Scratch reference', 'weights_only': 'Adjusted loss weights', 'cbam_only': 'Attention only', 'focal_only': 'Localization loss only',
           'bundle': 'Attention + localization loss', 'duplicate': 'Clean copies', 'refsoft': 'GlareSoft copies'}
SINGLE = {'cbam_only': 'Pretrained + attention only', 'focal_eiou_only': 'Pretrained + localization loss only', 'cbam_focal': 'Pretrained + attention + localization loss',
          'cbam_focal_e2': 'Pretrained + attention + localization loss + paired views', 'cbam_focal_e3': 'Pretrained + attention + localization loss + weighted paired views'}
DETECTOR = {'yolov8s': 'YOLOv8s', 'yolo11s': 'YOLO11s', 'yolo26s': 'YOLO26s', 'rtdetr_l': 'RT-DETR-L'}
DS = {'seadronessee': 'SDS', 'wsodd': 'WSODD', 'flow': 'FloW', 'SeaDronesSee': 'SDS', 'WSODD': 'WSODD', 'FloW': 'FloW'}
index = []      # one row per training run


def epochs(path):
    return sum(1 for _ in open(path, encoding='utf-8')) - 1 if path.exists() else ''


def copy_run(src, folder, block, dataset, configuration, seed, name=None):
    dst = OUT / folder / (name or src.name); dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name in KEEP:
            shutil.copy2(f, dst / f.name)
    index.append(dict(block=block, dataset=dataset, configuration=configuration, seed=seed, folder=f'{folder}/{dst.name}', epochs_run=epochs(dst / 'results.csv'), source='run folder'))


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
# 1 detector reference: SDS and WSODD from the detector block, FloW from the runs on the 1124/335/541 partition
for d in sorted((EVID / 'detector_supplements_20260914/remote_final/inputs/training_records').iterdir()):
    m = re.fullmatch(r'(seadronessee|wsodd)__(yolov8s|yolo11s|yolo26s|rtdetr_l)', d.name)
    if m:
        copy_run(d, '01_检测器参照', 'Detector reference', DS[m.group(1)], DETECTOR[m.group(2)], '0')
for d in sorted((EVID / 'flow_source_retrain_20260920/final_recovery/evidence/runs').iterdir()):
    m = re.fullmatch(r'baseline_(yolov8s|yolo11s|yolo26s|rtdetr_l)_seed0', d.name)
    if m:
        copy_run(d, '01_检测器参照', 'Detector reference', 'FloW', DETECTOR[m.group(1)], '0', name=f'flow__{m.group(1)}')
# 3 FloW strategies and identical paired views
for root in (EVID / 'flow_source_retrain_20260920/final_recovery/evidence/runs', ADDED / 'flow_source_retrain_20260920/release_v1/runs'):
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r'FloW_(E0|E2|E3|E2C)_seed(\d+)', d.name)
        if m:
            copy_run(d, '03_训练策略_FloW', 'Training strategies' if m.group(1) != 'E2C' else 'Matched controls', 'FloW', FLOW[m.group(1)], m.group(2))
# 4 WSODD matched controls from scratch
for root in (EVID / 'refsoft_reproduction_20260910/final_evidence_20260913/runs', ADDED / 'refsoft_extension_20261109/release_v1/runs'):
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r'WSODD_(baseline|weights_only|cbam_only|focal_only|bundle|duplicate|refsoft)_seed(\d+)', d.name)
        if m:
            copy_run(d, '04_匹配对照_WSODD从头训练', 'Matched controls (from scratch)', 'WSODD', MATCHED[m.group(1)], m.group(2))
# 5 WSODD pretrained copies
for d in sorted((ADDED / 'pretrained_copies_20261109/runs').iterdir()):
    m = re.fullmatch(r'WSODD_pretrained_(clean|glaresoft)_seed(\d+)', d.name)
    if m:
        copy_run(d, '05_匹配对照_WSODD预训练副本', 'Matched controls (pretrained fine-tuning)', 'WSODD', 'Pretrained + clean copies' if m.group(1) == 'clean' else 'Pretrained + GlareSoft copies', m.group(2))

# 2 SDS and WSODD strategy runs: arguments and per-epoch history exported from the checkpoints
sys.path.insert(0, str(CODE))
import torch
jobs = json.loads((EVID / 'recompute_20261109/jobs.json').read_text(encoding='utf-8'))['jobs']
runs = {}
for job in jobs:
    if job.get('arm') in ('E0', 'E1') and job['dataset'] in ('SeaDronesSee', 'WSODD'):
        runs[(job['dataset'], job['arm'], str(job['seed']))] = job['experiment_id']
weights = EVID / 'recompute_20261109/weights'
for (dataset, arm, seed), exp in sorted(runs.items()):
    path = weights / f'{exp}_last.pt' if (weights / f'{exp}_last.pt').exists() else weights / f'{exp}_best.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    folder = OUT / '02_训练策略_SDS_WSODD' / f'{DS[dataset]}_{arm}_seed{seed}'; folder.mkdir(parents=True)
    (folder / 'train_args_from_checkpoint.yaml').write_text(yaml.safe_dump(dict(ckpt['train_args']), allow_unicode=True, sort_keys=True), encoding='utf-8')
    history = ckpt.get('train_results')
    n = ''
    if isinstance(history, dict) and history:
        keys = list(history); n = len(history[keys[0]])
        with open(folder / 'results_from_checkpoint.csv', 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh); w.writerow(keys); w.writerows(zip(*[history[k] for k in keys]))
    (folder / 'checkpoint_record.json').write_text(json.dumps(dict(experiment_id=exp, exported_from=path.name, epoch_saved=ckpt.get('epoch'), best_fitness=ckpt.get('best_fitness'),
                                                                  date=ckpt.get('date'), version=ckpt.get('version'), history_rows=n), ensure_ascii=False, indent=1, default=str) + '\n', encoding='utf-8')
    index.append(dict(block='Training strategies', dataset=DS[dataset], configuration='Standard training' if arm == 'E0' else 'Fixed single view', seed=seed,
                      folder=f'02_训练策略_SDS_WSODD/{folder.name}', epochs_run=n, source=f'exported from {path.name} (weights not copied)'))
shutil.copy2(EVID / 'cross_dataset_e0_e1_20260909/cross_dataset_e0_e1_multiseed_table.csv', OUT / '02_训练策略_SDS_WSODD/cross_dataset_e0_e1_multiseed_table.csv')

# 6 FloW pretrained single-component runs: per-run records and the final tables of that block
single = RING3 / '03_历史归档/独立补充实验/flow_source_cbam_20260920'
(OUT / '06_FloW预训练单组件').mkdir()
for d in sorted((single / 'final_recovery_20260921/evidence/runs').iterdir()):
    m = re.fullmatch(r'FloW_(cbam_only|focal_eiou_only|cbam_focal|cbam_focal_e2|cbam_focal_e3)_seed(\d+)', d.name)
    if m:
        copy_run(d, '06_FloW预训练单组件', 'Pretrained single-component (FloW)', 'FloW', SINGLE[m.group(1)], m.group(2))
for f in sorted(single.glob('FINAL_*')):
    if f.suffix in ('.csv', '.md', '.json'):
        shutil.copy2(f, OUT / '06_FloW预训练单组件' / f.name)

# 7 unified evaluation records
ev = OUT / '07_统一评价记录'; (ev / '前78项').mkdir(parents=True)
shutil.copy2(RING3 / '01_当前论文/基础依赖/20260924_统一评价修订/unified_evaluations.csv', ev / '前78项/unified_evaluations.csv')
counts = {}
for label, root in (('条件评价与last检查点', EVID / 'recompute_20261109/vast_results/output/evaluations'), ('新增训练的评价', EVID / 'train_20261109/results/extracted/output/evaluations')):
    n = 0
    for d in sorted(root.iterdir()):
        if (d / 'evaluation.json').exists():
            (ev / label / d.name).mkdir(parents=True); shutil.copy2(d / 'evaluation.json', ev / label / d.name / 'evaluation.json'); n += 1
            if (d / 'coco_eval.log').exists():
                shutil.copy2(d / 'coco_eval.log', ev / label / d.name / 'coco_eval.log')
    counts[label] = n
shutil.copy2(RING3 / '01_当前论文/补充材料/20261109/Supplementary_Data/S1/Table_S1_all_evaluations.csv', ev / 'Table_S1_all_evaluations.csv')

with open(OUT / '运行索引.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['block', 'dataset', 'configuration', 'seed', 'folder', 'epochs_run', 'source']); w.writeheader(); w.writerows(index)
assert counts == {'条件评价与last检查点': 263, '新增训练的评价': 270} and len(index) == 123

# every run folder must be the run whose checkpoint the evaluation records name
expected = {}                                                   # run folder -> server folder of the evaluated checkpoint
ARM = {'A_baseline': 'baseline', 'B_weights_only': 'weights_only', 'C_CBAM_Focal_EIoU': 'bundle', 'D_clean_duplicate': 'duplicate', 'E_refsoft': 'refsoft'}
for r in csv.DictReader((ev / '前78项/unified_evaluations.csv').open(encoding='utf-8-sig')):
    run_dir = r['checkpoint'].rsplit('/weights/', 1)[0]
    if r['family'] == 'detector':
        key = f"01_检测器参照/{ {'SeaDronesSee': 'seadronessee', 'WSODD': 'wsodd', 'FloW': 'flow'}[r['dataset']] }__{r['arm_raw']}"
    elif r['family'] == 'strategy' and r['dataset'] == 'FloW':
        key = f"03_训练策略_FloW/FloW_{r['arm_raw']}_seed{r['seed']}"
    elif r['family'] == 'matched':
        key = f"04_匹配对照_WSODD从头训练/WSODD_{ARM[r['arm_raw']]}_seed{r['seed']}"
    else:
        continue                                                # SDS and WSODD strategy runs are exported from the checkpoints themselves
    assert expected.setdefault(key, run_dir) == run_dir, key
for d in sorted((ev / '新增训练的评价').iterdir()):
    job = json.loads((d / 'evaluation.json').read_text(encoding='utf-8'))['job']
    run = job['experiment_id']
    key = ('03_训练策略_FloW/' if run.startswith('FloW_') else '05_匹配对照_WSODD预训练副本/' if 'pretrained' in run else '04_匹配对照_WSODD从头训练/') + run
    assert expected.setdefault(key, job['weights'].rsplit('/weights/', 1)[0]) == job['weights'].rsplit('/weights/', 1)[0], key
checked = 0
for r in index:
    if r['folder'] in expected:
        args = yaml.safe_load((OUT / r['folder'] / 'args.yaml').read_text(encoding='utf-8'))
        assert str(args['save_dir']).rstrip('/') == expected[r['folder']], (r['folder'], args['save_dir'], expected[r['folder']])
        checked += 1
unchecked = sorted(r['folder'] for r in index if r['folder'] not in expected)
assert checked == len(expected) == 90 and all(u.startswith(('02_', '06_')) for u in unchecked), (checked, len(expected), unchecked[:5])
print('run folders matched to the evaluated checkpoint path:', checked, '| not checkable this way (exported from checkpoints, or the single-component block):', len(unchecked))
assert not list(OUT.rglob('*.pt')) and not [p for p in OUT.rglob('*') if 'prediction' in p.name.lower()]
by_block = {}
for r in index:
    by_block[r['block']] = by_block.get(r['block'], 0) + 1
print('training runs:', len(index), by_block, '| evaluation records:', counts)
(OUT / '先读我.md').write_text(f'''# 训练原始结果（不含权重）· 2026-11-12

对应稿件 machines-4634859 修回工作版本 20261109。这里是各次训练自己留下的记录和统一评价的记录，没有权重、预测文件、训练过程图片、数据集和代码。

**这一版替换 2026-11-11 的版本，改了两处：**

1. `01_检测器参照` 里 FloW 的四次训练，上一版放的是同样四种检测器更早一轮 FloW 训练的记录，不是论文评价所用检查点对应的那一轮。这一版换成了论文所用的那四次（训练参数逐项相同，逐轮记录不同）。SDS 和 WSODD 的八次没有问题。
2. `06_FloW预训练单组件` 上一版只有汇总表，并写着逐次运行的记录不在本地。记录其实在，这一版补上了 15 次训练各自的参数和逐轮记录。

其余内容与上一版相同。这一版在生成时逐个核对了运行文件夹与评价记录里检查点路径的对应关系（90 次训练可以这样核对，全部一致）。

## 内容

| 文件夹 | 内容 | 训练次数 |
|---|---|---|
| `01_检测器参照` | 四种检测器在三个数据集上各训练一次（种子 0）：`args.yaml`、`results.csv`；FloW 的四次另有完成和运行环境记录 | 12 |
| `02_训练策略_SDS_WSODD` | SDS、WSODD 的标准训练（E0）和固定单视图增强（E1）。这些运行的文件夹不在本地，训练参数和逐轮记录是从检查点里导出的（`train_args_from_checkpoint.yaml`、`results_from_checkpoint.csv`），检查点本身没有放进来；另有当时的汇总表 | 18 |
| `03_训练策略_FloW` | FloW 的标准训练（E0）、双视图（E2）、双视图加小目标加权（E3）和干净重复视图对照（E2C），各六个种子 | 24 |
| `04_匹配对照_WSODD从头训练` | 七种从头训练配置，各六个种子 | 42 |
| `05_匹配对照_WSODD预训练副本` | 预训练微调下追加干净副本或 GlareSoft 副本，各六个种子 | 12 |
| `06_FloW预训练单组件` | FloW 预训练微调下的五种配置（只加注意力、只加定位损失、两者都加、再加双视图、再加带小目标加权的双视图），各三个种子：每次训练的记录，以及当时的结果表和报告（`FINAL_*`） | 15 |
| `07_统一评价记录` | 每次评价的 `evaluation.json` 和评分日志：`条件评价与last检查点` 263 项（其中名为 GATE 的 5 项是对已有结果的复现验证，不计入论文的评价），`新增训练的评价` 270 项；`前78项` 是最初 78 项评价的封存汇总表；`Table_S1_all_evaluations.csv` 是补充材料里 621 项评价的总表（78 + 258 + 270，加上 `06` 里 FloW 预训练单组件的 15 项） | — |
| `运行索引.csv` | 每次训练一行：实验块、数据集、配置（论文里的名称）、种子、所在文件夹、实际训练轮数 | {len(index)} |

## 名称对照

| 记录里的名称 | 论文里的名称 |
|---|---|
| E0 | Standard training（标准训练） |
| E1 | Fixed single view（固定单视图增强） |
| E2 | Paired views（双视图） |
| E3 | Paired views + small-object weighting（双视图加小目标加权） |
| E2C | Identical paired views（干净重复视图对照） |
| baseline | Scratch reference（从头训练参照） |
| weights_only | Adjusted loss weights（调整损失权重） |
| cbam_only | Attention only（只加注意力） |
| focal_only | Localization loss only（只加定位损失） |
| bundle | Attention + localization loss（注意力加定位损失） |
| duplicate | Clean copies（干净副本） |
| refsoft | GlareSoft copies（GlareSoft 副本） |
| pretrained_clean、pretrained_glaresoft | 预训练微调下的干净副本、GlareSoft 副本 |
| cbam_only、focal_eiou_only、cbam_focal（FloW） | Pretrained + attention only、+ localization loss only、+ attention + localization loss |
| cbam_focal_e2、cbam_focal_e3（FloW） | 上一项再加 paired views、再加 weighted paired views |
| normal | Normal（常规输入） |
| low_light_L1、L2、strong、L4、L5 | Low light, level 1–5（strong 即第 3 档） |
| glare_G1、G2、strong、G4、G5 | Localized glare, level 1–5（strong 即第 3 档） |

## 说明

- 每个运行文件夹里：`args.yaml` 是训练器保存的全部参数，`results.csv` 是逐轮的训练损失和验证指标，`completion.json`、`runtime_evidence.json`、`runtime.json`、`job.json` 是完成状态和运行环境的记录（有的实验块没有其中几项）。
- 训练设了提前停止（patience 100），所以 `results.csv` 的行数可能少于 150。
- 论文里报告的数值不是 `results.csv` 里的验证指标，而是用统一评价流程（官方 COCOeval）在测试集上重新评分的结果，见 `07_统一评价记录`。
- 记录是原样复制的，里面有服务器路径、日期和内部编号。
- 文件清单和 SHA-256 见 `文件清单与SHA256.csv`。
''', encoding='utf-8')
files = sorted(p for p in OUT.rglob('*') if p.is_file())
with open(OUT / '文件清单与SHA256.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.writer(fh); w.writerow(['file', 'bytes', 'sha256'])
    for p in files:
        w.writerow([p.relative_to(OUT).as_posix(), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()])
archive = OUT.with_suffix('.zip') if False else OUT.parent / (NAME + '.zip')
if archive.exists():
    archive.unlink()
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, (Path(NAME) / p.relative_to(OUT)).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None; names = z.namelist()
(OUT.parent / (NAME + '.zip.sha256.txt')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n', encoding='utf-8')
total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
print(f'files: {len(names)} | folder {total / 1048576:.1f} MB | zip {archive.stat().st_size / 1048576:.1f} MB | weights or predictions inside: {any(n.endswith(".pt") or "prediction" in n.lower() for n in names)}')
