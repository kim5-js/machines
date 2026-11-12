"""Verify the downloaded AutoDL evidence archive before extracting into the workspace."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile

work = Path('D:/ring3/02_当前实验/原始证据/detector_supplements_20260914')
archive = work / 'detector_supplements_20260914_final_evidence.zip'
meta = json.loads((work / 'detector_supplements_20260914_final_evidence.metadata.json').read_text(encoding='utf-8-sig'))
with archive.open('rb') as f:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
assert h.hexdigest() == meta['sha256'] and archive.stat().st_size == meta['bytes']
target = work / 'remote_final'
assert not target.exists(), 'Preserve any previous extraction before retrying.'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == len(set(names)) == meta['members']
    assert z.testzip() is None
    manifest = json.loads(z.read('evidence_manifest.json'))
    assert set(names) == set(manifest['files']) | {'evidence_manifest.json'}
    for name in names:
        rel = PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and ':' not in name, name
        assert (target / name).resolve().is_relative_to(target.resolve()), name
    for name, record in manifest['files'].items():
        data = z.read(name)
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
    target.mkdir()
    for name in names:
        out = target / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('xb') as f: f.write(z.read(name))
result = {'status': 'downloaded_extracted_verified', 'archive_sha256': meta['sha256'], 'members': len(names), 'verified_files': len(manifest['files']), 'directory': str(target)}
(work / 'DOWNLOAD_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
