"""Verify collected COCO evidence, build comparison tables, and update the supplement report."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

PROJECT = Path(r'D:/ring3')
WORK = PROJECT / 'artifacts/detector_supplements_20260914'
OUT = PROJECT / 'deliverables/detector_supplements_20260914'
EVIDENCE = WORK / 'remote_final'
REMOTE = EVIDENCE / 'release_batch1_v1'
sys.path.insert(0, str(WORK / 'local_dependencies'))
os.environ['MPLCONFIGDIR'] = str(WORK / 'matplotlib')
MODELS = ['yolov8s', 'yolo11s', 'yolo26s', 'rtdetr_l']
DATASETS = ['wsodd', 'flow', 'seadronessee']
NAMES = {'yolov8s': 'YOLOv8s', 'yolo11s': 'YOLO11s', 'yolo26s': 'YOLO26s', 'rtdetr_l': 'RT-DETR-L'}
DS = {'wsodd': 'WSODD', 'flow': 'FloW', 'seadronessee': 'SeaDronesSee'}
METRICS = ['mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify():
    import numpy as np
    req = json.loads((REMOTE / 'request.json').read_text(encoding='utf-8'))
    assert sha(REMOTE / 'request.json') == 'e91bbe4b522cc4c4223a97f6a268fc32759d9a2e97187122718521175d5ad67c'
    manifest = json.loads((EVIDENCE / 'evidence_manifest.json').read_text(encoding='utf-8'))
    for name, value in manifest['files'].items():
        assert sha(EVIDENCE / name) == value['sha256'], name
        assert (EVIDENCE / name).stat().st_size == value['bytes']
    summary = json.loads((REMOTE / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed' and summary['jobs'] == len(summary['results']) == 12
    frozen_jobs = {j['key']: j for j in req['jobs']}
    for name, expected in req['source_files'].items():
        assert sha(REMOTE / name) == expected, name
    records = {}
    validation = []
    for r in summary['results']:
        job = r['job']; key = job['key']; ds = job['dataset']
        assert key not in records and job['seed'] == 0 and r['training_performed'] is False
        assert job == frozen_jobs[key], key
        assert r['request_sha256'] == sha(REMOTE / 'request.json')
        assert r['evaluator'] == req['evaluator'] and r['inference'] == req['inference']
        assert r['dataset_protocol'] == req['datasets'][ds]
        batch = r['actual_batch_probe']
        assert batch['loader_batch_sizes'] == [1] and batch['maximum_forward_batch'] == 1
        assert batch['input_dtypes'] == ['torch.float32'] and batch['forward_calls'] == r['evaluation_images']
        assert batch['input_shapes'] == [[1, 3, 960, 960]], key
        d = REMOTE / 'evaluations' / key
        assert json.loads((d / 'result.json').read_text(encoding='utf-8')) == r
        assert sha(d / 'predictions.json.gz') == r['predictions_sha256']
        assert sha(d / 'coco_arrays.npz') == r['arrays_sha256']
        assert sha(REMOTE / 'inputs' / ds / 'ground_truth.json') == req['datasets'][ds]['ground_truth_sha256']
        assert sha(REMOTE / 'inputs' / ds / 'images.json') == req['datasets'][ds]['image_manifest_sha256']
        gt = json.loads((REMOTE / 'inputs' / ds / 'ground_truth.json').read_text(encoding='utf-8'))
        ids = {x['id'] for x in gt['images']}; classes = {x['id'] for x in gt['categories']}
        assert len(ids) == r['evaluation_images'] == req['datasets'][ds]['images']
        progress = json.loads((d / 'progress.json').read_text(encoding='utf-8'))
        assert progress['images'] == progress['target_images'] == len(ids)
        with gzip.open(d / 'predictions.json.gz', 'rt', encoding='utf-8') as f:
            predictions = json.load(f)
        assert len(predictions) == r['predictions']
        counts = Counter()
        for pred in predictions:
            assert pred['image_id'] in ids and pred['category_id'] in classes
            assert 0 <= pred['score'] <= 1 and all(math.isfinite(x) for x in pred['bbox'])
            assert pred['bbox'][2] >= 0 and pred['bbox'][3] >= 0
            counts[pred['image_id']] += 1
        assert max(counts.values(), default=0) <= 300
        arrays = np.load(d / 'coco_arrays.npz')
        precision = arrays['precision']
        assert precision.shape == (10, 101, len(classes), 4, 3), precision.shape
        def mean_valid(x):
            valid = x[x >= 0]
            return float(valid.mean()) if len(valid) else None
        derived = {'mAP50-95': mean_valid(precision[:, :, :, 0, 2]), 'mAP50': mean_valid(precision[0, :, :, 0, 2]),
                   'mAP75': mean_valid(precision[5, :, :, 0, 2]), 'APsmall': mean_valid(precision[:, :, :, 1, 2]),
                   'APmedium': mean_valid(precision[:, :, :, 2, 2]), 'APlarge': mean_valid(precision[:, :, :, 3, 2])}
        for metric, value in derived.items():
            expected = r['metrics'][metric]
            assert (value is None and expected is None) or (value is not None and expected is not None and abs(value - expected) < 1e-12), (key, metric)
        for k, cat in enumerate(sorted(gt['categories'], key=lambda c: c['id'])):
            value = mean_valid(precision[:, :, k, 0, 2])
            expected = r['per_class_AP'][cat['name']]
            assert (value is None and expected is None) or abs(value - expected) < 1e-12
        validation.append({'job': key, 'prediction_records': len(predictions), 'evaluation_images': len(ids),
                           'images_with_no_retained_detections': len(ids - set(counts)), 'precision_array_metrics_match': True})
        records[key] = r
    assert set(records) == {d + '__' + m for d in DATASETS for m in MODELS}
    return req, records, {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(), 'verified_evidence_files': len(manifest['files']),
                          'records': validation, 'method': 'Hash and input identity checks, image coverage, prediction schema, and independent reconstruction of six AP metrics and per-class AP from COCO precision arrays.'}


def training_conditions(records):
    import yaml
    keys = ['epochs', 'imgsz', 'batch', 'seed', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'patience', 'amp', 'pretrained', 'close_mosaic', 'cos_lr', 'mosaic', 'mixup', 'copy_paste', 'deterministic', 'data', 'model']
    with (OUT / 'detector_training_conditions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', *keys]); writer.writeheader()
        for key, r in records.items():
            args = yaml.safe_load((REMOTE / 'inputs/training_records' / key / 'args.yaml').read_text(encoding='utf-8'))
            writer.writerow({'run': r['job']['run_dir'].split('/')[-1], **{k: args.get(k) for k in keys}})


def tables(records):
    fields = ['dataset', 'model', 'seed', 'evaluation_split', 'evaluation_images', 'evaluator', *METRICS, 'weights_sha256', 'record_type']
    with (OUT / 'detector_unified_coco_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for d in DATASETS:
            for m in MODELS:
                r = records[d + '__' + m]
                writer.writerow({'dataset': DS[d], 'model': NAMES[m], 'seed': 0, 'evaluation_split': r['evaluation_split'],
                                 'evaluation_images': r['evaluation_images'], 'evaluator': 'pycocotools 2.0.10 COCOeval bbox',
                                 **{k: r['metrics'][k] for k in METRICS}, 'weights_sha256': r['job']['weights_sha256'], 'record_type': 'unified_detector_reevaluation'})
    rows = []
    rankings = {}
    for d in DATASETS:
        ranks = sorted(MODELS, key=lambda m: records[d + '__' + m]['metrics']['mAP50-95'], reverse=True)
        rankings[d] = ranks
        for rank, m in enumerate(ranks, 1):
            r = records[d + '__' + m]
            rows.append({'dataset': DS[d], 'model': NAMES[m], 'rank': rank, 'mAP50_95_percent': r['metrics']['mAP50-95'] * 100,
                         'delta_vs_YOLOv8s_pp': (r['metrics']['mAP50-95'] - records[d + '__yolov8s']['metrics']['mAP50-95']) * 100})
    with (OUT / 'detector_rankings.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(OUT / 'detector_analysis.json', {'rankings': rankings, 'results': rows, 'seed_count': 1, 'statistical_significance_established': False})
    return rankings


def figures(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    colors = ['#34628d', '#749ba7', '#c06536', '#767084']
    split_labels = ['fixed test (2,406 images)', 'original val (200 images)', 'source-disjoint val (1,555 images)']
    ymax = max(55, max(r['metrics']['mAP50-95'] * 100 for r in records.values()) + 9)
    for ax, ds, label in zip(axes, DATASETS, split_labels):
        values = [records[ds + '__' + m]['metrics']['mAP50-95'] * 100 for m in MODELS]
        ax.bar(range(4), values, color=colors, width=.7)
        ax.set_xticks(range(4), [NAMES[m] for m in MODELS], rotation=28, ha='right')
        for i, v in enumerate(values): ax.text(i, v + .6, f'{v:.2f}', ha='center', fontsize=10)
        ax.set_title(DS[ds] + '\n' + label, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, ymax)
        ax.set_axisbelow(True); ax.grid(axis='y', color='#dddddd', linewidth=.6)
    axes[0].set_ylabel('COCO mAP50–95 (%)')
    fig.suptitle('Unified reevaluation of existing seed 0 detector checkpoints', y=.98, fontsize=14)
    fig.text(.5, .025, 'Same COCO evaluator; dataset-specific fixed evaluation protocols. Single-seed descriptive comparison.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.79, bottom=.24, wspace=.22)
    for ext in ['png', 'pdf', 'svg', 'tiff']:
        fig.savefig(OUT / ('detector_unified_comparison.' + ext), dpi=400, facecolor='white')
    plt.close(fig)


def report(records, ranks):
    old = (WORK / 'REPORT_before_uploaded_execution.md').read_text(encoding='utf-8')
    overlap = old.split('## 已完成的划分重叠统计', 1)[1].split('## 待执行的四 detector 统一重新评价', 1)[0]
    rows = []
    for m in MODELS:
        cells = []
        for d in DATASETS:
            value = records[d + '__' + m]['metrics']['mAP50-95'] * 100
            text = f'{value:.2f}'
            cells.append('**' + text + '**' if ranks[d][0] == m else text)
        rows.append('| ' + NAMES[m] + ' | ' + ' | '.join(cells) + ' |')
    small = ['| ' + NAMES[m] + ' | ' + ' | '.join(f"{records[d+'__'+m]['metrics']['APsmall']*100:.2f}" if records[d+'__'+m]['metrics']['APsmall'] is not None else 'NA' for d in DATASETS) + ' |' for m in MODELS]
    rank_text = '\n'.join('- ' + DS[d] + '：' + ' > '.join(NAMES[m] for m in ranks[d]) + '。' for d in DATASETS)
    ranking_varies = len({tuple(ranks[d]) for d in DATASETS}) > 1
    interpretation = ('这三套固定数据集/评价协议组合的模型排序不同，支持相对表现依赖评价条件的描述。' if ranking_varies else '这三套固定数据集/评价协议组合的模型排序相同，未复现文档预期的排名变化。')
    hypothesis = {'FloW 中 YOLO26s 第一': ranks['flow'][0] == 'yolo26s', 'SeaDronesSee 中 YOLO26s 第一': ranks['seadronessee'][0] == 'yolo26s', 'WSODD 中 RT-DETR-L 或 YOLOv8s 第一': ranks['wsodd'][0] in {'rtdetr_l', 'yolov8s'}}
    hypothesis_text = '附件原先提出的三个趋势，按本次统一 mAP50–95 逐项核对：' + '；'.join(k + ('，复现' if v else '，未复现') for k, v in hypothesis.items()) + '。'
    flow_yolo = [records['flow__'+m]['metrics']['mAP50-95']*100 for m in MODELS[:3]]
    hypothesis_text += f'FloW中三个YOLO的最大差距为{max(flow_yolo)-min(flow_yolo):.2f}个百分点；这些小幅排序差异仍需多seed实验判断稳定性。'
    dump(OUT / 'document_hypothesis_checks.json', {'metric': 'mAP50-95', 'hypotheses_reproduced': hypothesis, 'ranking_varies_across_dataset_protocols': ranking_varies, 'causal_domain_effect_established': False})
    text = '''# Detector 统一重评与划分重叠补充

两项补充均已完成：12份已有detector权重在固定评价协议下重新推理，并统一使用官方COCOeval计算指标；六套划分的sequence/source overlap已逐图重算。原E0/E2/E3训练队列和mdg自动跟踪继续暂停，未启动新训练。

## 四 detector 统一重新评价

下表是本次重新推理得到的COCO mAP50–95，单位为百分比。每个数据集的四模型使用相同的固定评价集；不把不同数据集上的绝对数值当成任务难度相同的横向比较。所有权重均来自已完成的正式seed0基线。

| Model | WSODD fixed test | FloW original val | SeaDronesSee source-disjoint val |
|---|---:|---:|---:|
''' + '\n'.join(rows) + '''

各数据集的本次排名为：

''' + rank_text + '''

''' + interpretation + '''由于每个模型只有seed0、各数据集的评价协议也不同，本结果是描述性比较，不能作为统计显著性或纯粹domain因果效应的证据。

''' + hypothesis_text + '''

APsmall也单列如下，单位为百分比，不能用总体mAP排名替代小目标表现：

| Model | WSODD fixed test | FloW original val | SeaDronesSee source-disjoint val |
|---|---:|---:|---:|
''' + '\n'.join(small) + '''

完整mAP50、mAP50–95、mAP75、APsmall/medium/large见detector_unified_coco_metrics.csv；逐模型排名及相对YOLOv8s差值见detector_rankings.csv，差值单位为百分点。

![统一评价结果](detector_unified_comparison.png)

### 协议与结论边界

| 数据集 | 原训练集 | 固定评价集 | 图像数 | 结果性质 |
|---|---:|---|---:|---|
| WSODD | 3977 | sequence-disjoint test | 2406 | 固定测试集 |
| FloW | 1000 | 原val | 200 | 用于模型选择的validation，存在来源组重叠 |
| SeaDronesSee | 8922 | source-disjoint val | 1555 | 来源组互斥validation，曾用于模型选择 |

不把后来重划的FloW test用于这四个旧模型，因为该test中包含这些模型训练时见过的图像。SeaDronesSee使用其已有基线训练对应的8922/1555协议，没有切换到原8930/1547划分。

官方pycocotools 2.0.10 COCOeval，bbox，IoU 0.50至0.95，maxDets=[1,10,100]。GT由对应固定YOLO标签转换，类别索引统一加1，保留14/1/5类语义；面积按原图像素计算，iscrowd=0，不把ignored类别恢复成检测类。缺失面积或无GT类别保持null。

推理固定imgsz=960、batch=1、conf=0.001、iou=0.70、max_det=300、FP32、无TTA。各架构保留原predictor：YOLO letterbox，RT-DETR stretch；YOLO26与RT-DETR保留原端到端后处理，不额外强加NMS。最终预测都交给同一COCOeval。SeaDronesSee在这里是基于固定YOLO标签转换的COCO bbox评价，不是官方评测服务器成绩。

本次没有训练、修改权重或依据新评价挑选checkpoint。12份best.pt、训练记录、dataset YAML、图像/标签清单和GT均有SHA-256绑定。checkpoint内epoch字段已被清除为-1；权重清单里的CSV最佳epoch不冒充checkpoint的精确保存轮次。

训练记录中的共同参数为150轮、imgsz=960、batch=8、seed0、AdamW；同一数据集的四模型lr0一致（WSODD 0.000556、FloW 0.002、SeaDronesSee 0.001111），详见detector_training_conditions.csv。历史RT-DETR训练经既有稳定性修复使用AMP=false，而YOLO训练使用AMP=true。统一评价器没有消除历史训练方案差异；不能将所有差异单独归因于网络结构。原生训练日志指标在权重清单中另列，没有混入本表。

### 结果验证与证据

所有12组均完成真实图像GPU推理、覆盖检查与COCO评价。已保存逐图预测、统一GT、每类AP、COCO precision/recall数组、日志、固定脚本和依赖版本。下载后逐文件核对哈希，并从COCO precision数组独立重算六项AP及每类AP，与报告数值一致。5项评价自检覆盖完美匹配、空预测、错误类别、缺失面积组和无GT类别。另外将WSODD/YOLOv8s的完整2,406张图GT与64,524条预测下载到独立Windows Python环境，直接调用官方COCOeval重跑：12项AP/AR和precision、recall、scores数组均完全一致，未导入项目中的评价实现。

执行中发现路径列表会绕过声明的batch=1。初轮的3份完成结果及RT-DETR未完成输出已保留，均不纳入正式表；独立修复版本使用每次一张图，并断言实际前向张量的batch维为1、dtype为FP32，全部12组据此重新评价。权重、评价集及声明的推理参数未改变，恢复记录与两版代码均保留。

原训练仍暂停在SeaDronesSee E0 seed0第53轮进行中（前52轮完成），6/27组已完成的原结果及best/last检查点保留。未自动恢复训练或原heartbeat。

## 已完成的划分重叠统计
''' + overlap
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')


def main():
    req, records, verification = verify()
    training_conditions(records)
    rankings = tables(records)
    figures(records)
    report(records, rankings)
    dump(OUT / 'unified_evaluation_verification.json', verification)
    dump(WORK / 'FINAL_ANALYSIS.json', {'rankings': rankings, 'results': list(records.values()), 'verification': verification})
    inventory = OUT / 'detector_checkpoint_inventory.csv'
    with inventory.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    for row in rows: row['unified_reevaluation_status'] = 'completed_verified'
    with inventory.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'status': 'completed_verified', 'results': [{'job': k, 'metrics': r['metrics']} for k, r in records.items()], 'rankings': rankings}, indent=2))


if __name__ == '__main__':
    main()
