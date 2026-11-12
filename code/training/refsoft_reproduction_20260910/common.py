from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'vendor'))
sys.path.insert(1, str(CODE / 'historical'))
sys.path.insert(2, str(CODE / 'evaluation_code'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(CODE.parent / 'settings'))
os.environ.setdefault('MPLBACKEND', 'Agg')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def verify_release():
    m = load(CODE / 'release_manifest.json')
    for item in m['files']:
        if sha(CODE / item['path']) != item['sha256']:
            raise RuntimeError('Release mismatch: ' + item['path'])
    return sha(CODE / 'release_manifest.json')


def protocol(root):
    p = load(Path(root) / 'protocol.json')
    if verify_release() != p['release_sha256']:
        raise RuntimeError('Wrong release')
    for path, expected in p['locked_files'].items():
        if sha(path) != expected:
            raise RuntimeError('Input changed: ' + path)
    return p


def state(root, status, **details):
    write(Path(root) / 'state.json', {'status': status, 'updated_utc': now(), **details})


def event(root, name, **details):
    with (Path(root) / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'event': name, 'utc': now(), **details}, allow_nan=False) + '\n')
