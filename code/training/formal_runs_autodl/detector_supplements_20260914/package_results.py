"""Build the final user-facing supplement package after all real evaluations pass."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil, sys, zipfile

PROJECT = Path('D:/ring3')
WORK = PROJECT / 'artifacts/detector_supplements_20260914'
OUT = PROJECT / 'deliverables/detector_supplements_20260914'
CODE = PROJECT / 'autodl/detector_supplements_20260914'
ARCHIVE = PROJECT / 'deliverables/detector_supplements_completed_20260914.zip'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    assert read(OUT / 'unified_evaluation_verification.json')['status'] == 'passed'
    assert read(WORK / 'first_result_audit/independent_coco_recompute.json')['status'] == 'passed'
    remote = WORK / 'remote_final'
    final = read(remote / 'final_remote_verification.json')
    assert final['status'] == 'passed' and len(final['verified_jobs']) == 12
    assert all(p['state'] == 'T' for p in final['legacy_processes'])
    raw = WORK / 'detector_supplements_20260914_final_evidence.zip'
    meta = read(WORK / 'detector_supplements_20260914_final_evidence.metadata.json')
    assert sha(raw) == meta['sha256'] and raw.stat().st_size == meta['bytes']
    evidence = OUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    shutil.copyfile(raw, evidence / 'AutoDL_raw_evidence.zip')
    dump(evidence / 'AutoDL_raw_evidence.metadata.json', meta)
    for name in ['final_remote_verification.json', 'evidence_manifest.json']:
        shutil.copyfile(remote / name, evidence / name)
    for name in ['source_request.docx', 'legacy_pause_record.json', 'execution_bundle_v2.zip', 'remote_input_inventory.json', 'local_coco_self_test.json']:
        shutil.copyfile(WORK / name, evidence / name)
    for name in ['independent_coco_recompute.json', 'independent_coco_recompute.log']:
        shutil.copyfile(WORK / 'first_result_audit' / name, evidence / name)
    shutil.copyfile(WORK / 'recovery_batch1/recovery_record.json', evidence / 'batch1_recovery_record.json')
    for sub in ['initial_uploaded_bundle', 'executed_release_batch1_v1', 'local_analysis']:
        (OUT / 'scripts' / sub).mkdir(exist_ok=True)
    for name in ['prepare.py', 'evaluate.py', 'run_queue.py', 'overlap.py']:
        initial = OUT / 'scripts' / name
        if initial.exists():
            target = OUT / 'scripts/initial_uploaded_bundle' / name
            assert not target.exists(), target
            assert initial.resolve().is_relative_to(OUT.resolve()) and target.resolve().is_relative_to(OUT.resolve())
            initial.rename(target)
        shutil.copyfile(remote / 'release_batch1_v1/code' / name, OUT / 'scripts/executed_release_batch1_v1' / name)
    for name in ['finalize_results.py', 'audit_first_result.py', 'archive_remote_evidence.py', 'collect_evidence.py', 'package_results.py']:
        shutil.copyfile(CODE / name, OUT / 'scripts/local_analysis' / name)
    (OUT / 'scripts/README.md').write_text('''# 代码版本说明

- executed_release_batch1_v1：生成正式12组结果的实际AutoDL源码，包含逐图batch=1与FP32前向断言。
- initial_uploaded_bundle：初轮上传版本，因路径列表实际batch=16而排除其结果，仅保留作审计。
- local_analysis.py：已完成的split-overlap分析与图表生成脚本。
- local_analysis：本地正式结果校验、COCO独立复算和归档脚本；其中固定路径反映本次实际工作区，迁移运行时需显式修改路径。

执行环境及脚本SHA-256见原始证据ZIP中的release_batch1_v1/request.json。已完成的评估无需重新推理；预测和GT足够独立复算COCO指标。
''', encoding='utf-8')
    with (OUT / 'detector_unified_coco_metrics.csv').open(encoding='utf-8-sig') as f:
        records = list(csv.DictReader(f))
    assert len(records) == 12 and all(r['record_type'] == 'unified_detector_reevaluation' for r in records)
    verification = read(OUT / 'verification.json')
    verification.update({'verified_utc': datetime.now(timezone.utc).isoformat(), 'new_detector_results_present': True, 'unified_detector_results': 12, 'unified_evaluation_verification': 'passed', 'independent_full_coco_recompute': 'passed', 'raw_remote_archive_sha256': meta['sha256'], 'initial_batch16_results_excluded': 3, 'legacy_training': 'paused', 'legacy_automation': 'PAUSED'})
    sys.path.insert(0, str(WORK / 'local_dependencies'))
    from PIL import Image
    with Image.open(OUT / 'detector_unified_comparison.tiff') as im:
        verification['image_export']['detector_unified_comparison.tiff'] = {'pixels': list(im.size), 'dpi': [float(v) for v in im.info['dpi']]}
        assert min(im.info['dpi']) >= 300
    dump(OUT / 'verification.json', verification)
    for path in [WORK / 'UPLOAD_APPROVAL.md', OUT / 'UPLOAD_APPROVAL.md']:
        text = path.read_text(encoding='utf-8-sig')
        text = text.replace('12组统一重评正在执行。', '12组统一重评已完成并通过结果校验。')
        path.write_text(text, encoding='utf-8')
    (OUT / 'README.md').write_text('''# 两项补充的最终交付

从 REPORT.md 阅读结果与结论边界。两项补充均已完成，原E0/E2/E3训练和mdg自动跟踪仍暂停，未新训练模型。

## 结果

- detector_unified_coco_metrics.csv：正式12组统一COCO评价，指标为0至1；报告与图中的指标乘100显示为百分比。
- detector_rankings.csv：逐数据集排名及相对YOLOv8s的百分点差值。
- detector_checkpoint_inventory.csv：12份既有权重来源和哈希，native训练指标另列，不能与本次COCO指标混用。
- detector_training_conditions.csv：从冻结args.yaml逐组提取的训练条件。
- split_overlap_table.csv、split_overlap_summary.json：六套划分的分组重叠，另附WSODD阈值敏感性结果。
- detector_unified_comparison、WSODD_split_overlap、WSODD_sequence_membership：PNG/PDF/SVG/TIFF图件，TIFF为400 dpi。

## 原始证据与复核

- evidence/AutoDL_raw_evidence.zip：整个本次AutoDL补充目录的审计归档；正式结果仅位于release_batch1_v1/evaluations。父级evaluations中的3份初轮完成结果因实际batch=16已排除，恢复记录说明全过程。
- 每组包含predictions.json.gz、result.json、coco_arrays.npz、coco_eval.log和进度记录；固定GT、训练记录、图像/标签哈希、脚本、依赖来源和协议保存在inputs、code及request.json中。
- 原数据集图像与best.pt大文件没有重复打包，仍在已冻结的AutoDL路径；完成时再次逐一核验了12份权重和4161张评价图像/对应标签的SHA-256。
- evidence/overlap：六套协议和两项敏感性检查的逐图group归属与组级统计；evidence/inputs保存16份对应源文件。
- unified_evaluation_verification.json：12组覆盖、预测结构、哈希与从precision数组重构指标的核验结果。
- evidence/independent_coco_recompute.json：独立Windows环境重跑首组完整COCO评分，12项指标及3套数组完全相同。
- SHA256_MANIFEST.json：本目录所有交付文件的哈希，清单自身除外；外层ZIP另有校验记录。

原始证据ZIP解压后，使用官方pycocotools==2.0.10即可对inputs中的ground_truth.json和对应predictions.json.gz执行COCOeval bbox，maxDets=[1,10,100]。缺失面积组/无GT类别的-1统计值在报告中保持空值，不改为0。vendor符号链接未重复存档，其目标依赖已保存在父级vendor。

这些结果是seed0的既有权重重评。FloW与SeaDronesSee使用曾参与checkpoint选择的validation集；只有WSODD使用固定test。不能据此声称多seed显著性、官方评测服务器成绩，或反光增强的独立收益。

附件 source_request.docx 是用户要求执行两项补充的来源；其中预期排名作为待检验假设，实际结果见document_hypothesis_checks.json。
''', encoding='utf-8')
    digest_map = {str(p.relative_to(OUT)).replace('\\', '/'): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'SHA256_MANIFEST.json'}
    dump(OUT / 'SHA256_MANIFEST.json', digest_map)
    assert not ARCHIVE.exists(), 'Preserve prior final archive; investigate before replacing.'
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for name in sorted(digest_map):
            z.write(OUT / name, name, compress_type=zipfile.ZIP_STORED if name.endswith('.zip') else zipfile.ZIP_DEFLATED)
        z.write(OUT / 'SHA256_MANIFEST.json', 'SHA256_MANIFEST.json')
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(digest_map) + 1
        for name, digest in digest_map.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    result = {'status': 'completed_verified', 'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'members': len(digest_map) + 1, 'files_verified': len(digest_map), 'unified_evaluations': 12, 'split_overlap_protocols': 6, 'sensitivity_checks': 2, 'raw_evidence_sha256': meta['sha256'], 'legacy_training': 'paused', 'legacy_automation': 'PAUSED', 'verified_utc': datetime.now(timezone.utc).isoformat()}
    dump(ARCHIVE.with_suffix('.verification.json'), result)
    dump(WORK / 'FINAL_PACKAGE_VERIFICATION.json', result)
    state = read(WORK / 'STATUS.json')
    state.update({'status': 'both_supplements_completed_verified', 'updated_utc': result['verified_utc'], 'unified_detector_evaluations_completed': 12, 'current_job': None, 'queue_pid': None, 'final_package': result, 'legacy_training': 'paused', 'legacy_automation': 'PAUSED'})
    state['execution_bundle']['remote_evaluations_started'] = 12
    state['execution_bundle']['remote_evaluations_completed'] = 12
    state['queue_completed_utc'] = read(remote / 'release_batch1_v1/summary.json')['completed_utc']
    dump(WORK / 'STATUS.json', state)
    runbook = WORK / 'RUNBOOK.md'
    old = runbook.read_text(encoding='utf-8-sig')
    old = old.replace('当前阶段：四detector统一重新评价正在AutoDL运行；split overlap六套协议已完成并交付。', '当前阶段：两项补充均已完成并通过核验；正式12组评价和split overlap六套协议已打包交付。')
    old = old.replace('当前队列PID：973565，2026-09-14T13:41:28Z启动。', '评估队列PID 973565已正常完成；2026-09-14T13:41:28Z启动，完成时刻见STATUS.json。')
    runbook.write_text(old + '\n最终交付：' + str(ARCHIVE) + '\nSHA-256：' + result['sha256'] + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
