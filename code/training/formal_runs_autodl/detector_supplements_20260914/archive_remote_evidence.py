"""Run only after the 12 corrected jobs finish; freeze and archive existing evidence."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, zipfile

ROOT = Path('/root/autodl-tmp/detector_supplements_20260914')
ACTIVE = ROOT / 'release_batch1_v1'
ARCHIVE = ROOT.parent / 'detector_supplements_20260914_final_evidence.zip'
EXPECTED_REQUEST = 'e91bbe4b522cc4c4223a97f6a268fc32759d9a2e97187122718521175d5ad67c'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def process(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'starttime': int(fields[19]), 'cmd': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return {'pid': pid, 'state': 'exited'}

assert ROOT.resolve() == ROOT and ACTIVE.resolve().is_relative_to(ROOT)
assert not ARCHIVE.exists(), 'Preserve any prior archive; investigate before rerunning.'
state = json.loads((ACTIVE / 'state.json').read_text())
summary = json.loads((ACTIVE / 'summary.json').read_text())
assert state['status'] == summary['status'] == 'completed'
assert summary['jobs'] == len(summary['results']) == 12
assert not list((ACTIVE / 'failures').glob('*.json'))
assert sha(ACTIVE / 'request.json') == EXPECTED_REQUEST
req = json.loads((ACTIVE / 'request.json').read_text())
assert not req['training'] and req['mode'] == 'existing_best_checkpoint_inference_only'
assert len(req['jobs']) == 12 and len({j['key'] for j in req['jobs']}) == 12
for key, digest in req['scripts'].items():
    assert sha(ACTIVE / 'code' / key) == digest, key
for name, digest in req['source_files'].items():
    assert sha(ACTIVE / name) == digest, name
assert sha(ACTIVE / 'input_contract.json') == req['input_contract_sha256']
checked_images = 0
checked_labels = 0
for ds, protocol in req['datasets'].items():
    assert sha(protocol['source_yaml']) == protocol['source_yaml_sha256'], ds
    assert sha(protocol['ground_truth']) == protocol['ground_truth_sha256'], ds
    assert sha(protocol['image_manifest']) == protocol['image_manifest_sha256'], ds
    images = json.loads(Path(protocol['image_manifest']).read_text())
    assert len(images) == protocol['images']
    for image in images:
        assert sha(image['path']) == image['image_sha256'], image['path']
        assert sha(image['label']) == image['label_sha256'], image['label']
        checked_images += 1
        checked_labels += 1
    print(json.dumps({'source_reverified': ds, 'images': len(images)}), flush=True)
results = {r['job']['key']: r for r in summary['results']}
verified_jobs = []
for job in req['jobs']:
    key = job['key']
    assert sha(job['weights']) == job['weights_sha256'], key
    assert Path(job['weights']).stat().st_size == job['weights_bytes']
    assert sha(Path(job['run_dir']) / 'args.yaml') == job['args_sha256'], key
    assert sha(Path(job['run_dir']) / 'results.csv') == job['csv_sha256'], key
    r = json.loads((ACTIVE / 'evaluations' / key / 'result.json').read_text())
    assert r == results[key] and r['job'] == job
    assert r['status'] == 'completed' and r['training_performed'] is False
    assert r['request_sha256'] == EXPECTED_REQUEST
    assert r['evaluator'] == req['evaluator'] and r['inference'] == req['inference']
    assert r['dataset_protocol'] == req['datasets'][job['dataset']]
    b = r['actual_batch_probe']
    assert b['loader_batch_sizes'] == [1] and b['maximum_forward_batch'] == 1
    assert b['input_dtypes'] == ['torch.float32'] and b['input_shapes'] == [[1, 3, 960, 960]]
    assert b['forward_calls'] == r['evaluation_images'] == r['dataset_protocol']['images']
    for name, digest in [('predictions.json.gz', r['predictions_sha256']), ('coco_arrays.npz', r['arrays_sha256'])]:
        assert sha(ACTIVE / 'evaluations' / key / name) == digest, key
    verified_jobs.append({'job': key, 'weights_sha256': job['weights_sha256'], 'evaluation_images': r['evaluation_images'], 'actual_batch_probe': b})
legacy = [process(943702), process(967972)]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '62aefc31-0202-416b-bf6b-3d688e3dae9d'
assert legacy[0]['state'] == legacy[1]['state'] == 'T'
assert legacy[0]['starttime'] == 6322022410 and legacy[1]['starttime'] == 6327230568
stopped_group = []
active_processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    item = process(int(p.name))
    if item.get('pgid') == 967972:
        assert item['state'] == 'T', item
        stopped_group.append(item)
    if str(ACTIVE / 'code/run_queue.py') in item.get('cmd', '') or str(ACTIVE / 'code/evaluate.py') in item.get('cmd', ''):
        assert item['state'] == 'Z', item
        active_processes.append(item)
paused = ROOT.parent / 'legacy_e2e3_multiseed_20260911/release_v2/user_pause/20260914T122415Z/preserved_run'
for name, digest in [('weights/best.pt', '0ad9bc15c16b1b6104d37cc402a1eacabf95ce085d8a326c583d689fadc1f82d'), ('weights/last.pt', '0cbcd999534a582db2766c23d50eb79890e3bf0ef0237383d8c9c099d622b44f')]:
    assert sha(paused / name) == digest, name
original_results = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'evaluations').glob('*/result.json'))
assert len(original_results) == 3
verification = {'status': 'passed', 'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'active_release': str(ACTIVE), 'request_sha256': EXPECTED_REQUEST, 'verified_jobs': verified_jobs, 'evaluation_images_rehashed': checked_images, 'evaluation_labels_rehashed': checked_labels, 'checkpoint_and_training_record_hashes_unchanged': True, 'legacy_processes': legacy, 'legacy_stopped_process_group': stopped_group, 'supplement_processes': active_processes, 'excluded_initial_completed_results': original_results, 'exclusion_reason': 'Actual inference batch was 16 instead of frozen batch 1; only corrected release_batch1_v1 results are valid.', 'legacy_preserved_checkpoint_hashes_match': True, 'gpu': subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()}
write(ROOT / 'final_remote_verification.json', verification)
files = {}
links = {}
for path, dirs, names in os.walk(ROOT, followlinks=False):
    parent = Path(path)
    for name in list(dirs):
        item = parent / name
        if item.is_symlink():
            target = item.resolve()
            assert target.is_relative_to(ROOT), str(item)
            links[str(item.relative_to(ROOT))] = {'target': str(target), 'archived_once_at': str(target.relative_to(ROOT))}
            dirs.remove(name)
    for name in names:
        item = parent / name
        if item.name == 'evidence_manifest.json' and item.parent == ROOT: continue
        assert not item.is_symlink() and item.is_file() and item.resolve().is_relative_to(ROOT), str(item)
        files[str(item.relative_to(ROOT))] = {'sha256': sha(item), 'bytes': item.stat().st_size}
manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'root': str(ROOT), 'active_release': 'release_batch1_v1', 'files': dict(sorted(files.items())), 'symlinks_not_duplicated': links, 'original_dataset_images_and_checkpoint_blobs_included': False, 'provenance': 'Original images, labels, and checkpoints remain at frozen AutoDL paths; their hashes were reverified at completion. Archive contains unified GT, image/label hashes, predictions, evaluation arrays, logs, code, contracts, training records, and dependency provenance.'}
write(ROOT / 'evidence_manifest.json', manifest)
with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
    for name in sorted(files): z.write(ROOT / name, name)
    z.write(ROOT / 'evidence_manifest.json', 'evidence_manifest.json')
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(files) + 1
    for name, record in files.items():
        data = z.read(name)
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
meta = {'status': 'archived_verified', 'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'members': len(files) + 1, 'verified_files': len(files), 'valid_results': 12, 'excluded_completed_initial_results': 3, 'legacy_training': 'paused', 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(ARCHIVE.with_suffix('.metadata.json'), meta)
print(json.dumps(meta, indent=2), flush=True)
