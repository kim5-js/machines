"""Assemble the review repository for revision 20261109: code, records, checkpoints, evaluation outputs, supplement.

The folder written here is the working tree of a separate Git repository (weights and predictions go to Git LFS).
It holds no manuscript text, no reviewer correspondence and no dataset images.

Checkpoints are selected from the evaluation records, not from folder names: every checkpoint hash named by one of
the evaluations behind the paper is located (local run folders first, then the server backup archives), copied and
re-hashed. A run's last.pt is added next to its best.pt when it can be found. The pretrained weights the runs
started from are added as well, and the code is compared with the file hashes each experiment block recorded.

Output  02_当前实验/已验证交付/论文复核仓库_20261112/      (an existing .git folder inside it is kept)
Run     python build_review_repository.py [--reuse-large] (.venv; standard library only)
        --reuse-large keeps checkpoints/ and evaluation_outputs/ from the previous build; every file is still checked
Needs   02_当前实验/已验证交付/训练原始结果_20261112/       (build_raw_results_package.py)
"""
from __future__ import annotations

import csv, hashlib, json, os, re, shutil, sys, tarfile, zipfile, zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3'); EVID = RING3 / '02_当前实验/原始证据'; GEN = RING3 / '01_当前论文/生成脚本'
NAME = '论文复核仓库_20261112'
OUT = RING3 / '02_当前实验/已验证交付' / NAME
RECORDS = RING3 / '02_当前实验/已验证交付/训练原始结果_20261112'
BACKUP = RING3 / '05_备份/AutoDL_full_backup_20260928'
SINGLE = RING3 / '03_历史归档/独立补充实验/flow_source_cbam_20260920'
NOV = EVID / 'train_20261109/leftovers_core/extracted'                    # what was left on the training machine after the added runs
MB = 1024 ** 2
DSKEY = {'SeaDronesSee': 'seadronessee', 'WSODD': 'wsodd', 'FloW': 'flow'}
ARM = {'A_baseline': 'baseline', 'B_weights_only': 'weights_only', 'C_CBAM_Focal_EIoU': 'bundle', 'D_clean_duplicate': 'duplicate', 'E_refsoft': 'refsoft'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(4 * MB), b''):
            h.update(block)
    return h.hexdigest()


def same(src, dst):
    return dst.exists() and dst.stat().st_size == src.stat().st_size and int(dst.stat().st_mtime) == int(src.stat().st_mtime)


def copy_tree(src, dst, keep=lambda rel: True, max_bytes=20 * MB):
    """Copy the files of src for which keep(relative posix path) is true; returns the number of files."""
    n = 0
    for p in sorted(Path(src).rglob('*')):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and '__pycache__' not in rel and not rel.endswith('.pyc') and p.stat().st_size <= max_bytes and keep(rel):
            if not same(p, dst / rel):
                (dst / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst / rel)
            n += 1
    return n


# ---------------------------------------------------------------- start from an empty tree, keeping .git
REUSE = '--reuse-large' in sys.argv
OUT.mkdir(parents=True, exist_ok=True)
for child in OUT.iterdir():
    if child.name != '.git' and not (REUSE and child.name in ('checkpoints', 'evaluation_outputs')):
        shutil.rmtree(child) if child.is_dir() else child.unlink()

# ---------------------------------------------------------------- 1 records
n_records = copy_tree(RECORDS, OUT / 'records')
index = list(csv.DictReader((OUT / 'records/运行索引.csv').open(encoding='utf-8-sig')))
assert len(index) == 123, len(index)
RUNS = {r['folder'] for r in index}

# ---------------------------------------------------------------- 2 checkpoints named by the evaluation records
def run_folder(family, dataset, arm, seed, experiment=None):
    if experiment and not experiment.startswith('AD'):                     # added runs carry the run name
        return ('03_训练策略_FloW/' if experiment.startswith('FloW_') else '05_匹配对照_WSODD预训练副本/' if 'pretrained' in experiment else '04_匹配对照_WSODD从头训练/') + experiment
    if family == 'detector':
        return f'01_检测器参照/{DSKEY[dataset]}__{arm}'
    if family == 'matched':
        return f'04_匹配对照_WSODD从头训练/WSODD_{ARM.get(arm, arm)}_seed{seed}'
    if dataset == 'FloW':
        return f'03_训练策略_FloW/FloW_{arm}_seed{seed}'
    return f"02_训练策略_SDS_WSODD/{'SDS' if dataset == 'SeaDronesSee' else 'WSODD'}_{arm}_seed{seed}"


wanted = {}                 # (run folder, file name) -> dict(sha, keys, server)
def want(folder, name, sha, key, server=None):
    assert folder in RUNS, folder
    w = wanted.setdefault((folder, name), dict(sha=sha, keys=[], server=None))
    assert w['sha'] == sha, (folder, name, key)
    w['keys'].append(key); w['server'] = w['server'] or server


sealed = list(csv.DictReader((OUT / 'records/07_统一评价记录/前78项/unified_evaluations.csv').open(encoding='utf-8-sig')))
for r in sealed:
    want(run_folder(r['family'], r['dataset'], r['arm_raw'], r['seed']), 'best.pt', r['checkpoint_sha256'], r['evaluation_key'], r['checkpoint'])
EVAL_SOURCES = (('条件评价与last检查点', EVID / 'recompute_20261109/vast_results/output'), ('新增训练的评价', EVID / 'train_20261109/results/extracted/output'))
for label, root in EVAL_SOURCES:
    for d in sorted((root / 'evaluations').iterdir()):
        job = json.loads((d / 'evaluation.json').read_text(encoding='utf-8'))['job']
        name = 'last.pt' if job['weights'].replace('\\', '/').rsplit('/', 1)[1].endswith('last.pt') else 'best.pt'
        want(run_folder(job['family'], job['dataset'], job['arm'], job['seed'], job['experiment_id']), name, job['weights_sha256'], d.name,
             job['weights'] if job['weights'].startswith('/root/') else None)
assert len({w['sha'] for w in wanted.values()}) == len(wanted) == 114, len(wanted)
server_dir = {}             # run folder -> folder of the run on the training server
for (folder, name), w in wanted.items():
    if w['server']:
        server_dir[folder] = w['server'].rsplit('/weights/', 1)[0]

LOCAL_ROOTS = [EVID / 'flow_source_retrain_20260920/final_recovery/evidence/runs', EVID / 'train_20261109/weights/extracted', EVID / 'recompute_20261109/weights']
local = {}                  # sha -> path, run folders preferred over the copies made for re-evaluation
for root in LOCAL_ROOTS:
    for p in sorted(root.rglob('*.pt')):
        if p.name in ('best.pt', 'last.pt') or root.name == 'weights':
            local.setdefault(sha256(p), p)
members = {}                # path inside the backup archives -> archive
for z in sorted(BACKUP.glob('*_model_weights_*.zip')):
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if i.filename.endswith('.pt'):
                members[i.filename] = z


def from_backup(server_path, dst):
    """Extract the file stored at server_path on the training server from the backup archives; returns its SHA-256 or None."""
    tail = server_path.replace('\\', '/').split('/root/', 1)[-1]
    hits = [m for m in members if m == tail or m.endswith('/' + tail)]
    if len(hits) != 1:
        return None
    h = hashlib.sha256(); dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(members[hits[0]]) as f, f.open(hits[0]) as src, open(dst, 'wb') as out:
        for block in iter(lambda: src.read(4 * MB), b''):
            h.update(block); out.write(block)
    return h.hexdigest()


CK = OUT / 'checkpoints'
table = []                  # rows of the checkpoint index
source_of = {}              # run folder -> local path of best.pt when it came from a run folder
for (folder, name), w in sorted(wanted.items()):
    dst = CK / folder / name
    if w['sha'] in local:
        if not same(local[w['sha']], dst):
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(local[w['sha']], dst)
        origin = 'local copy'
        if name == 'best.pt' and local[w['sha']].name == 'best.pt':
            source_of[folder] = local[w['sha']]
    else:
        assert w['server'], (folder, name)
        got = sha256(dst) if dst.exists() else from_backup(w['server'], dst); origin = 'server backup archive'
        assert got == w['sha'], (folder, name, got)
    assert sha256(dst) == w['sha'], (folder, name)
    table.append(dict(run=folder, file=name, bytes=dst.stat().st_size, sha256=w['sha'], evaluations=len(w['keys']), evaluation_keys=' '.join(sorted(w['keys'])), obtained_from=origin))

# the 15 FloW pretrained single-component runs: their records carry no checkpoint hash, the files come from the block's recovered evidence
recovery = (SINGLE / 'final_recovery_20260921/evidence/recovery_manifest.json').read_text(encoding='utf-8')
for r in index:
    if r['folder'].startswith('06_'):
        src = SINGLE / 'final_recovery_20260921/evidence/runs' / r['folder'].split('/')[1] / 'weights'
        for name in ('best.pt', 'last.pt'):
            dst = CK / r['folder'] / name
            if not same(src / name, dst):
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src / name, dst)
            digest = sha256(dst); assert digest in recovery, (r['folder'], name)
            table.append(dict(run=r['folder'], file=name, bytes=dst.stat().st_size, sha256=digest, evaluations=2 if name == 'best.pt' else 0,
                              evaluation_keys='validation and test evaluation of the block (hash listed in its recovery manifest)' if name == 'best.pt' else '', obtained_from='local copy'))

# last.pt of the remaining runs, where it can be found
missing_last = []
for r in index:
    folder = r['folder']; dst = CK / folder / 'last.pt'
    if any(t['run'] == folder and t['file'] == 'last.pt' for t in table):
        continue
    assert (CK / folder / 'best.pt').exists(), folder
    if folder in source_of and (source_of[folder].parent / 'last.pt').exists():
        if not same(source_of[folder].parent / 'last.pt', dst):
            shutil.copy2(source_of[folder].parent / 'last.pt', dst)
        origin = 'local copy'
    elif folder in server_dir and (dst.exists() or from_backup(server_dir[folder] + '/weights/last.pt', dst)):
        origin = 'server backup archive'
    else:
        missing_last.append(folder); continue
    table.append(dict(run=folder, file='last.pt', bytes=dst.stat().st_size, sha256=sha256(dst), evaluations=0, evaluation_keys='', obtained_from=origin))
table.sort(key=lambda t: (t['run'], t['file']))
with open(CK / '检查点索引.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['run', 'file', 'bytes', 'sha256', 'evaluations', 'evaluation_keys', 'obtained_from']); w.writeheader(); w.writerows(table)
n_best = sum(t['file'] == 'best.pt' for t in table); n_last = sum(t['file'] == 'last.pt' for t in table)
assert n_best == 123 and {t['run'] for t in table} == RUNS
on_disk = {p.relative_to(CK).as_posix() for p in CK.rglob('*.pt') if not p.relative_to(CK).as_posix().startswith('00_')}
assert on_disk == {t['run'] + '/' + t['file'] for t in table}, sorted(on_disk ^ {t['run'] + '/' + t['file'] for t in table})[:5]
ck_bytes = sum(t['bytes'] for t in table)
print(f'checkpoints: best {n_best}, last {n_last}, {ck_bytes / 1024 ** 3:.2f} GB | verified against evaluation records: {len(wanted)} | from backup: {sum(t["obtained_from"] != "local copy" for t in table)} | runs without last.pt: {len(missing_last)}')
for m in missing_last:
    print('   no last.pt:', m)

# pretrained starting weights named by the run arguments: public release weights, taken from the bundles sent to the training machine
START = {'yolov8s.pt': EVID / 'train_20261109/matched/matched_train_bundle.tar', 'yolo11s.pt': EVID / 'train_20261109/flow/flow_train_bundle.tar',
         'yolo26s.pt': EVID / 'train_20261109/flow/flow_train_bundle.tar', 'rtdetr-l.pt': EVID / 'train_20261109/flow/flow_train_bundle.tar'}
named_by = Counter()
for f in list((OUT / 'records').glob('0*/*/args.yaml')) + list((OUT / 'records').glob('0*/*/train_args_from_checkpoint.yaml')):
    for n in {m.rsplit('/', 1)[-1] for m in re.findall(r'^(?:model|pretrained):\s*(\S+\.pt)\s*$', f.read_text(encoding='utf-8'), re.M)}:
        named_by[n] += 1
assert set(named_by) == set(START), named_by
hash_records = {'flow_source_retrain_20260920/request.json': NOV / 'root/autodl-tmp/flow_source_retrain_20260920/release_v1/request.json',
                'flow_source_cbam_20260920/request.json': SINGLE / 'final_recovery_20260921/evidence/request.json'}
hash_text = {k: v.read_text(encoding='utf-8') for k, v in hash_records.items()}
START_DIR = CK / '00_预训练起始权重'; START_DIR.mkdir(parents=True, exist_ok=True)
start_rows = []
for name, tar in START.items():
    dst = START_DIR / name
    with tarfile.open(tar) as t:
        member = next(m for m in t if m.isfile() and m.name.endswith('autodl-tmp/weights/' + name))
        if not (dst.exists() and dst.stat().st_size == member.size):
            dst.write_bytes(t.extractfile(member).read())
    digest = sha256(dst); where = [k for k, text in hash_text.items() if digest in text]
    assert where, name                                                       # the block requests record the hash of the weights they started from
    hits = [m for m in members if m.endswith('autodl-tmp/weights/' + name)]
    assert len(hits) == 1 and zipfile.ZipFile(members[hits[0]]).getinfo(hits[0]).CRC == zlib.crc32(dst.read_bytes()) & 0xffffffff, name      # same file as on the first server
    start_rows.append(dict(file=name, bytes=dst.stat().st_size, sha256=digest, runs_naming_it=named_by[name], hash_recorded_in=' '.join(where)))
with open(START_DIR / '起始权重.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['file', 'bytes', 'sha256', 'runs_naming_it', 'hash_recorded_in']); w.writeheader(); w.writerows(start_rows)
print('pretrained starting weights:', {r['file']: r['runs_naming_it'] for r in start_rows})

# ---------------------------------------------------------------- 3 evaluation outputs
EO = OUT / 'evaluation_outputs'
n_eval = {}
for label, root in EVAL_SOURCES:
    n_eval[label] = copy_tree(root, EO / label, max_bytes=60 * MB)
(EO / '前78项').mkdir(parents=True, exist_ok=True)
if not same(GEN / '20260924_统一评价/unified_evaluation_evidence.zip', EO / '前78项/unified_evaluation_evidence.zip'):
    shutil.copy2(GEN / '20260924_统一评价/unified_evaluation_evidence.zip', EO / '前78项/unified_evaluation_evidence.zip')
shutil.copy2(OUT / 'records/07_统一评价记录/前78项/unified_evaluations.csv', EO / '前78项/unified_evaluations.csv')
n_eval['FloW预训练单组件'] = copy_tree(SINGLE / 'final_recovery_20260921/evidence/evaluations', EO / 'FloW预训练单组件', max_bytes=60 * MB)
pred = sorted(EO.rglob('predictions.json.gz'))
assert len(pred) == 263 + 270 + 30, len(pred)
print('evaluation outputs:', n_eval, '| prediction files:', len(pred))

# ---------------------------------------------------------------- 4 code
CODE = OUT / 'code'
TEXT = ('.py', '.sh', '.ps1', '.yaml', '.yml', '.json', '.md', '.txt', '.cfg', '.toml', '.csv')
DATA_LISTS = ('data_records.json', 'source_files.json', 'staging_manifest.json', 'recovery_manifest.json')      # file-by-file listings of the datasets and recovered evidence
def code(rel):
    parts = rel.split('/')
    return rel.endswith(TEXT) and parts[-1] not in DATA_LISTS and not ({'runs', 'smoke_runs', 'smoke_data', 'evaluation_data', 'verified_data', 'evaluations', 'logs', 'settings', 'anchors', 'results', 'mpl'} & set(parts[:-1]))


n_code = {}
flow = NOV / 'root/autodl-tmp/flow_source_retrain_20260920/release_v1'
n_code['training/flow_source_retrain_20260920'] = copy_tree(flow, CODE / 'training/flow_source_retrain_20260920', code)
n_code['training/flow_extension_20261109'] = copy_tree(NOV / 'root/autodl-tmp/flow_extension_20261109', CODE / 'training/flow_extension_20261109', code)
n_code['training/refsoft_reproduction_20260910'] = copy_tree(EVID / 'refsoft_reproduction_20260910/release', CODE / 'training/refsoft_reproduction_20260910', code)
n_code['training/refsoft_extension_20261109'] = copy_tree(EVID / 'train_20261109/matched/code', CODE / 'training/refsoft_extension_20261109', code)
n_code['training/pretrained_copies_20261109'] = copy_tree(NOV / 'root/autodl-tmp/pretrained_copies_20261109', CODE / 'training/pretrained_copies_20261109', code)
n_code['training/flow_source_cbam_20260920'] = copy_tree(SINGLE / 'final_recovery_20260921/evidence', CODE / 'training/flow_source_cbam_20260920', code)
LEFT_OUT = ('legacy_e2e3_multiseed/README.md',                                    # gives the login address of the training server
            'formal_experiment_lock.json', 'formal_runtime_amendment.json')       # internal records of who was told to start the runs
n_code['training/formal_runs_autodl'] = copy_tree(RING3 / '02_当前实验/代码/autodl', CODE / 'training/formal_runs_autodl', lambda rel: code(rel) and rel not in LEFT_OUT)
QUEUE = ('train_queue_vast.py', 'flow_ext_train.py', 'pretrained_copies_train.py', 'build_flow_extension.py', 'build_matched_extension.py', 'post_train_eval.py', 'analyze_train_results.py',
         'check_status.py', 'gpu_pin.py', 'replay_autodl/compare_replay.py', 'replay_autodl/replay_e0_original.py', 'replay_autodl/replay_matched_original.py', 'replay_autodl/replay_runner.py')
n_code['training/queue_20261109'] = copy_tree(GEN / 'train_20261109', CODE / 'training/queue_20261109', lambda rel: rel in QUEUE)
UNIFIED = ('unified_eval.py', 'build_request.py', 'analyze_results.py', 'export_results.py', 'parallel_remaining.py', 'request.json',
           'sealed_evidence/unified_eval.py', 'sealed_evidence/request.json', 'sealed_evidence/prepared.json', 'sealed_evidence/evidence_manifest.json', 'sealed_evidence/status.json')
n_code['evaluation/unified_evaluation_20260924'] = copy_tree(GEN / '20260924_统一评价', CODE / 'evaluation/unified_evaluation_20260924', lambda rel: rel in UNIFIED)
RECOMPUTE = ('analyze_vast_results.py', 'build_extra_jobs.py', 'build_remote_jobs.py', 'build_vast_bundle.py', 'check_gate.py', 'common.py', 'evaluate_local.py', 'focal_diagnostic.py',
             'offline_analysis.py', 'prepare_local.py', 'queue_local.py', 'recompute_remote.py', 'recompute_vast.py')
n_code['evaluation/recompute_20261109'] = copy_tree(GEN / 'recompute_20261109', CODE / 'evaluation/recompute_20261109', lambda rel: rel in RECOMPUTE)
ANALYSIS = ('build_revision_data.py', 'build_round2_data.py', 'build_figures_20261109.py', 'build_figures_round2.py', 'build_supplement_20261109.py', 'build_supplement_s6.py',
            'build_raw_results_package.py', 'build_review_repository.py')
n_code['analysis'] = copy_tree(GEN / '20261109', CODE / 'analysis', lambda rel: rel in ANALYSIS)
for k, v in n_code.items():
    assert v > 0, k
# the records name the hash of the scoring module (pycocotools cocoeval.py); the same file is in the vendored code
evaluator = sha256(CODE / 'training/flow_source_retrain_20260920/vendor/pycocotools/cocoeval.py')
evaluator_named = {r['evaluator_code_sha256'] for r in sealed}
for label, root in EVAL_SOURCES:
    for d in sorted((root / 'evaluations').iterdir()):
        evaluator_named.add(json.loads((d / 'evaluation.json').read_text(encoding='utf-8'))['evaluator_code_sha256'])
assert evaluator_named == {evaluator}, (evaluator_named, evaluator)
print('code files:', n_code)
print('scoring module hash named by all', len(sealed) + 533, 'evaluation records equals the copied pycocotools/cocoeval.py:', evaluator_named == {evaluator})

# the code against the file hashes each block recorded when it was released for running
def recorded(obj, out):
    if isinstance(obj, dict):
        path = obj.get('path') or obj.get('file') or obj.get('relative_path'); h = obj.get('sha256') or obj.get('sha256_hex')
        if isinstance(path, str) and isinstance(h, str) and len(h) == 64:
            out[path.replace('\\', '/')] = h
        for k, v in obj.items():
            if isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) and isinstance(k, str) and ('/' in k or k.endswith(('.py', '.yaml', '.json', '.sh'))):
                out[k.replace('\\', '/')] = v
            recorded(v, out)
    elif isinstance(obj, list):
        for v in obj:
            recorded(v, out)


code_check = {}
for block, manifests in (('flow_source_retrain_20260920', ('execution_bundle_manifest.json', 'adapter_manifest.json')), ('flow_source_cbam_20260920', ('execution_bundle_manifest.json', 'adapter_manifest.json', 'preflight.json')),
                         ('refsoft_reproduction_20260910', ('release_manifest.json',)), ('refsoft_extension_20261109', ('release_manifest.json',))):
    base = CODE / 'training' / block; rec = {}
    for name in manifests:
        recorded(json.loads((base / name).read_text(encoding='utf-8')), rec)
    same_n = 0; absent = []
    for path, h in rec.items():
        hit = next((c for c in [base / path] + [base / path.split('/', i)[-1] for i in range(1, 4) if path.count('/') >= i] if c.is_file()), None)
        if hit is None:
            absent.append(path)
        else:
            assert sha256(hit) == h, (block, path); same_n += 1
    scripts_absent = [a for a in absent if a.endswith(('.py', '.sh', '.yaml', '.yml')) and not a.startswith(('anchors/', 'evaluation_data/', 'verified_data/', 'smoke'))]
    assert not scripts_absent, (block, scripts_absent)
    code_check[block] = same_n
server_copy = defaultdict(set)                  # the code folder as it was on the first training server (backup archives): file -> {(crc, size)}
for z in sorted(BACKUP.glob('*_project_records_*.zip')):
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if 'ring3_setup/autodl/' in i.filename and not i.is_dir():
                server_copy[i.filename.split('ring3_setup/autodl/', 1)[1]].add((i.CRC, i.file_size))
formal = CODE / 'training/formal_runs_autodl'; formal_same = 0; formal_diff = []
for f in sorted(formal.rglob('*')):
    rel = f.relative_to(formal).as_posix()
    if f.is_file() and rel in server_copy:
        data = f.read_bytes()
        if (zlib.crc32(data) & 0xffffffff, len(data)) in server_copy[rel]:
            formal_same += 1
        else:
            formal_diff.append(rel)
assert all(d.endswith('.md') for d in formal_diff), formal_diff                 # only notes were edited afterwards, no script
print('code identical to the recorded hashes:', code_check, '| formal_runs_autodl identical to the server copy:', formal_same, '| differing notes:', formal_diff)

# ---------------------------------------------------------------- 5 supplement and environment
n_supp = copy_tree(RING3 / '01_当前论文/补充材料/20261109/Supplementary_Data', OUT / 'supplementary')
ENV = OUT / 'environment'; ENV.mkdir()
shutil.copy2(BACKUP / 'environment_versions.json', ENV / 'server_environment_versions.json')
shutil.copy2(NOV / 'workspace/recompute_20261109/requirements_vast.txt', ENV / 'requirements_evaluation_and_added_runs.txt')
shutil.copy2(EVID / 'train_20261109/matched/code/historical/requirements.txt', ENV / 'requirements_matched_controls.txt')
shutil.copy2(NOV / 'workspace/train_20261109/ul_wheel/ultralytics-8.4.124-py3-none-any.whl', ENV / 'ultralytics-8.4.124-py3-none-any.whl')

# ---------------------------------------------------------------- 6 repository files
(OUT / '.gitattributes').write_text('# no line-ending conversion anywhere: the files must stay byte-identical to the hashed originals\n* -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n*.gz filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n'
                                    '*.zip filter=lfs diff=lfs merge=lfs -text\n*.whl filter=lfs diff=lfs merge=lfs -text\n', encoding='utf-8')
(OUT / '.gitignore').write_text('__pycache__/\n*.pyc\n', encoding='utf-8')
blocks = defaultdict(int)
for r in index:
    blocks[r['folder'].split('/')[0]] += 1
no_last = ('；有 ' + str(len(missing_last)) + ' 次训练只找到 best.pt，见下') if missing_last else ''
readme = f'''# 论文复核材料（machines-4634859，修回工作版本 20261109）

这个仓库放的是论文实验部分的全部材料：训练代码、评价代码、每次训练的记录、每次训练的权重、每项评价的输出，以及补充材料里的数据表。不含稿件、审稿意见和数据集图像。

权重、预测文件和压缩包存在 Git LFS 里。克隆前先装好 Git LFS（`git lfs install`），否则拿到的只是指针文件；全部下载约 {(ck_bytes + sum(p.stat().st_size for p in EO.rglob('*') if p.is_file())) / 1024 ** 3:.1f} GB。只想看代码和记录，可以用 `GIT_LFS_SKIP_SMUDGE=1 git clone ...` 跳过大文件。

## 目录

| 目录 | 内容 |
|---|---|
| `records/` | {len(index)} 次训练各自的参数和逐轮记录，533 项评价的评分记录，最初 78 项评价的封存汇总表；说明见 `records/先读我.md` |
| `checkpoints/` | 每次训练的 `best.pt` 和 `last.pt`，文件夹名与 `records/` 一一对应；`检查点索引.csv` 列出每个文件的 SHA-256 和用到它的评价。`00_预训练起始权重` 是预训练微调所用的四个公开权重 |
| `evaluation_outputs/` | 每项评价的完整输出：逐图预测（`predictions.json.gz`）、COCOeval 中间数组、评分日志和 `evaluation.json` |
| `code/training/` | 各实验块在服务器上实际运行的训练代码，按实验块分文件夹 |
| `code/evaluation/` | 统一评价脚本（官方 COCOeval）和条件评价脚本 |
| `code/analysis/` | 由评价记录生成论文的统计量、图和补充材料数据表的脚本 |
| `supplementary/` | 补充材料 S1–S6 的数据表和三个数据集的划分清单 |
| `environment/` | 服务器环境的软件版本、依赖清单和所用的 ultralytics 安装包 |

## 实验块、记录、权重和代码的对应

`records/` 和 `checkpoints/` 用同一套文件夹名。

| 文件夹 | 训练次数 | 内容 | 训练代码 |
|---|---|---|---|
| `01_检测器参照` | {blocks['01_检测器参照']} | 四种检测器在三个数据集上各一次（种子 0） | SDS、WSODD：`formal_runs_autodl`；FloW：`flow_source_retrain_20260920` |
| `02_训练策略_SDS_WSODD` | {blocks['02_训练策略_SDS_WSODD']} | SDS、WSODD 的标准训练和固定单视图增强 | `formal_runs_autodl` |
| `03_训练策略_FloW` | {blocks['03_训练策略_FloW']} | FloW 的标准训练、双视图、带小目标加权的双视图、干净重复视图对照 | `flow_source_retrain_20260920`、`flow_extension_20261109` |
| `04_匹配对照_WSODD从头训练` | {blocks['04_匹配对照_WSODD从头训练']} | 七种从头训练配置，各六个种子 | `refsoft_reproduction_20260910`、`refsoft_extension_20261109` |
| `05_匹配对照_WSODD预训练副本` | {blocks['05_匹配对照_WSODD预训练副本']} | 预训练微调下追加干净副本或 GlareSoft 副本 | `pretrained_copies_20261109` |
| `06_FloW预训练单组件` | {blocks['06_FloW预训练单组件']} | FloW 预训练微调下的五种配置，各三个种子 | `flow_source_cbam_20260920` |

记录里的名称与论文里的名称的对照，见 `records/先读我.md`。`code/training/queue_20261109` 是修回阶段追加训练的排队和启动脚本。

## 权重是怎么核对的

权重不是按文件夹名挑的，而是从评价记录出发：论文用到的每一项评价都记着所用检查点的 SHA-256，共 {len(wanted)} 个不同的检查点（{len(index) - blocks['06_FloW预训练单组件']} 次训练的 `best.pt`，加上 SDS 六次训练的 `last.pt`）。这 {len(wanted)} 个文件逐个找到、复制后重新计算哈希，全部与记录一致。`06_FloW预训练单组件` 的 {blocks['06_FloW预训练单组件']} 次训练，评价记录里没有检查点哈希，文件取自该实验块回收的证据，哈希与它的回收清单一致。

其余的 `last.pt` 没有被论文评价，只是随 `best.pt` 一起放入，共 {n_last - 6} 个{no_last}。
{''.join(chr(10) + '- 没有 last.pt：`' + m + '`' for m in missing_last)}

`checkpoints/00_预训练起始权重` 里是训练参数里指明的四个起始权重（{'、'.join(f"`{r['file']}`，{r['runs_naming_it']} 次训练" for r in start_rows)}）。它们是 Ultralytics 公开发布的权重，SHA-256 与实验块启动时记下的哈希一致。从头训练的 {len(index) - sum(r['runs_naming_it'] for r in start_rows)} 次不用起始权重。

## 代码是怎么核对的

- 四个实验块在发布到服务器运行时，各自记下了每个代码文件的 SHA-256。仓库里对应的文件逐个重算后全部一致：{'、'.join(f'`{k}` {v} 个' for k, v in code_check.items())}。
- `formal_runs_autodl`（SDS、WSODD 的训练策略和检测器参照）没有这样的清单，改为与第一台训练服务器备份里的同名文件比对：{formal_same} 个文件逐字节相同{'；' + str(len(formal_diff)) + ' 个说明文档后来改过（' + '、'.join('`' + d + '`' for d in formal_diff) + '），脚本没有差别' if formal_diff else ''}。
- 修回阶段追加训练的启动脚本和评价脚本，与训练机器上留下的副本逐字节相同。

评分用的是 pycocotools 的官方 COCOeval。评价记录里的 `evaluator_code_sha256` 是评分模块 `cocoeval.py` 的 SHA-256，78 项封存评价和 533 项后续评价记的是同一个值，与 `code/training/flow_source_retrain_20260920/vendor/pycocotools/cocoeval.py` 一致。调用它的评价脚本在 `code/evaluation/`。

## 没有放进来的东西

- 数据集图像和标注。三个数据集都是公开数据集，划分清单在 `supplementary/` 里。`evaluation_outputs/前78项/unified_evaluation_evidence.zip` 是当时封存的评价证据，里面带有三个评价划分的 COCO 格式标注文件（不含图像）。
- 训练过程中保存的示例图、曲线图和逐次训练的控制台日志。
- 论文没有用到的权重：冒烟测试和中途失败的运行；27 次从头训练另按 mAP 保存的 `best_map.pt`（评价用的是 `best.pt`）；检测器参照训练中每 10 轮保存的中间检查点。
- 随代码打包的第三方库里的非代码文件（示例图片、网页模板、pycocotools 的已编译模块）。pycocotools 2.0.10 可以用 pip 安装。
- 服务器上与论文无关的其他实验的脚本。
- 稿件、审稿意见和回复信。
- `code/training/formal_runs_autodl` 里的三个文件：一份写有训练服务器登录地址的说明，两份内部的启动授权记录。

## 说明

- 代码和记录是原样复制的，里面有服务器路径、日期和内部编号；脚本里的路径指向当时的服务器或本机目录，换机器运行需要改路径。
- 带注意力模块的检查点在加载时需要能导入 `code/training/formal_runs_autodl` 里的模型定义。
- 论文里的数值来自 `evaluation_outputs/`（测试集或验证集上用官方 COCOeval 重新评分），不是训练记录 `results.csv` 里的验证指标。
'''
(OUT / 'README.md').write_text(readme, encoding='utf-8')

files = [p for p in OUT.rglob('*') if p.is_file() and '.git' not in p.relative_to(OUT).parts]
LFS = ('.pt', '.gz', '.npz', '.zip', '.whl')
lfs_bytes = sum(p.stat().st_size for p in files if p.suffix in LFS); plain = [p for p in files if p.suffix not in LFS]
plain_bytes = sum(p.stat().st_size for p in plain)
print(f'files: {len(files)} | Git LFS {lfs_bytes / 1024 ** 3:.2f} GB in {len(files) - len(plain)} files | ordinary Git {plain_bytes / MB:.1f} MB in {len(plain)} files | largest ordinary file {max(p.stat().st_size for p in plain) / MB:.1f} MB')
(OUT.parent / (NAME + '_build_record.json')).write_text(json.dumps(dict(files=len(files), lfs_bytes=lfs_bytes, plain_bytes=plain_bytes, checkpoints=dict(best=n_best, last=n_last, verified=len(wanted), without_last=missing_last),
                                                                    evaluation_outputs=n_eval, code=n_code, records=n_records, supplement=n_supp, evaluator_hash_matches=evaluator_named == {evaluator}), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
