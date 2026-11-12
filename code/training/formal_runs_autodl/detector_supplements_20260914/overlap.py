"""Recompute sequence/source overlap from fixed members and original acquisition metadata."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import importlib.util
import itertools
import json
import shutil
import sys
import yaml
from prepare import ROOT, SETUP, BASE, images_for, sha, write, now


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def counts(name, group_by_image, split_members, unit, details):
    assert all(len(v) == len(set(v)) for v in split_members.values()), 'Duplicate split member'
    assert set().union(*(set(v) for v in split_members.values())) == set(group_by_image)
    counters = {s: Counter(group_by_image[i] for i in ids) for s, ids in split_members.items()}
    pairs = {}
    for a, b in itertools.combinations(split_members, 2):
        shared = set(counters[a]) & set(counters[b])
        count_b = sum(counters[b][g] for g in shared)
        pairs[a + '_' + b] = {'left': a, 'right': b, 'left_groups': len(counters[a]), 'right_groups': len(counters[b]),
                              'shared_groups': len(shared), 'left_only_groups': len(set(counters[a]) - set(counters[b])),
                              'right_only_groups': len(set(counters[b]) - set(counters[a])),
                              'left_images_from_shared_groups': sum(counters[a][g] for g in shared),
                              'right_images_from_shared_groups': count_b,
                              'right_images_total': len(split_members[b]), 'right_overlap_fraction': count_b / len(split_members[b]),
                              'exact_member_overlap': len(set(split_members[a]) & set(split_members[b])), 'shared_group_ids': sorted(shared)}
    records = [{'group': g, **{s: counters[s][g] for s in counters}} for g in sorted(set(group_by_image.values()))]
    directory = ROOT / 'overlap' / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'groups.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', *counters])
        writer.writeheader(); writer.writerows(records)
    with (directory / 'image_groups.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'split', 'group'])
        writer.writeheader()
        for s, members in split_members.items():
            writer.writerows({'image': image, 'split': s, 'group': group_by_image[image]} for image in members)
    out = {'protocol': name, 'unit': unit, 'total_images': len(group_by_image), 'total_groups': len(set(group_by_image.values())),
           'split_images': {s: len(v) for s, v in split_members.items()}, 'split_groups': {s: len(v) for s, v in counters.items()},
           'pairwise': pairs, 'definition': details, 'groups_sha256': sha(directory / 'groups.csv'), 'image_groups_sha256': sha(directory / 'image_groups.csv')}
    write(directory / 'summary.json', out)
    return out


def run():
    sources = {}
    def preserve(p, rel):
        dst = ROOT / 'overlap/inputs' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        sources[str(dst.relative_to(ROOT))] = {'source': str(p), 'sha256': sha(p)}
        return dst
    manifests = SETUP / 'manifests'
    for group in ('wsodd_sequence_disjoint_v1', 'wsodd_legacy_image_split_v1', 'flow_source_temporal_disjoint_v1', 'seadronessee_source_disjoint_v1'):
        for p in (manifests / group).iterdir():
            if p.suffix in ('.txt', '.tsv', '.json'):
                preserve(p, group + '/' + p.name)
    ws_module = module('audit_wsodd', preserve(SETUP / 'audit_wsodd_split.py', 'audit_wsodd_split.py'))
    sds_module = module('group_sds', preserve(SETUP / 'build_seadronessee_protocol.py', 'build_seadronessee_protocol.py'))
    results = []
    ws_splits = {}
    for protocol, yf in (('wsodd_legacy_image_split_v1', 'wsodd_current.yaml'), ('wsodd_sequence_disjoint_v1', 'wsodd_sequence_disjoint_v1.yaml')):
        data = yaml.safe_load((SETUP / 'datasets' / yf).read_text())
        splits = {s: [p.name.casefold() for p in images_for(data, s)] for s in ('train', 'val', 'test') if s in data}
        for s, values in splits.items():
            expected = [Path(x).name.casefold() for x in (manifests / protocol / (s + '.txt')).read_text().splitlines() if x.strip()]
            assert set(values) == set(expected) and len(values) == len(expected), (protocol, s)
        ws_splits[protocol] = splits
    # Preserve original case for the timestamp parser's literal WIN_ prefix.
    all_names = sorted({p.name for s in ('train', 'val', 'test') if s in data for p in images_for(data, s)})
    assert len(all_names) == 7467
    group_map = ws_module.build_groups([Path(x) for x in all_names], 60.0)
    details = {'grouping': 'existing filename source/timestamp rule; consecutive gaps greater than 60 seconds create a new sequence; numeric names kept in one burst; unknown names singleton',
               'gap_seconds': 60, 'official_video_ids': False, 'inference_limit': 'filename-derived temporal groups, not independently annotated physical scenes', 'source_sha256': sha(SETUP / 'audit_wsodd_split.py')}
    for name, members in ws_splits.items():
        results.append(counts(name, group_map, members, 'inferred_temporal_sequence', details))
    sensitivity = []
    for gap in (30., 120.):
        mapping = ws_module.build_groups([Path(x) for x in all_names], gap)
        result = counts('wsodd_legacy_gap' + str(int(gap)), mapping, ws_splits['wsodd_legacy_image_split_v1'], 'inferred_temporal_sequence', dict(details, gap_seconds=gap))
        sensitivity.append(result)
    fp = manifests / 'flow_source_temporal_disjoint_v1/source_manifest.tsv'
    rows = list(csv.DictReader(fp.open(encoding='utf-8-sig'), delimiter='\t'))
    fg = {r['image_name'].casefold(): r['source_group'] for r in rows}
    assert len(fg) == len(rows) == 1200
    for name, yf, column in [('flow_current', 'flow_current.yaml', 'current_split'), ('flow_source_temporal_disjoint_v1', 'flow_source_temporal_disjoint_v1.yaml', 'split')]:
        data = yaml.safe_load((SETUP / 'datasets' / yf).read_text())
        members = {s: [p.name.casefold() for p in images_for(data, s)] for s in ('train', 'val', 'test') if s in data}
        for s, ids in members.items():
            assert set(ids) == {r['image_name'].casefold() for r in rows if r[column] == s}
        results.append(counts(name, fg, members, 'source_acquisition_session', {'grouping': 'existing official XML-derived source/session groups; numeric renamed IDs are identifiers only', 'source_manifest_sha256': sha(fp)}))
    source_groups, source_records, official = {}, [], {'train': [], 'val': []}
    for origin in ('train', 'val'):
        p = BASE / 'datasets/SeaDronesSee/annotations' / ('instances_' + origin + '.json')
        coco = json.loads(p.read_text(encoding='utf-8-sig'))
        sources[str(p)] = {'source': str(p), 'sha256': sha(p), 'retained_subset': 'image metadata with source fields; annotation source remains on AutoDL'}
        for im in coco['images']:
            filename = Path(im['file_name']).name
            key = origin + '/' + filename
            assert (BASE / 'datasets/SeaDronesSee/images' / origin / filename).is_file()
            source_groups[key] = sds_module.group_key(im, origin)
            official[origin].append(key)
            source_records.append({'original_split': origin, 'metadata': im})
    assert len(source_groups) == 10477
    write(ROOT / 'overlap/inputs/seadronessee_source_images.json', source_records)
    new_members = {}
    for split in ('train', 'val'):
        lines = (manifests / 'seadronessee_source_disjoint_v1' / (split + '.tsv')).read_text().splitlines()
        new_members[split] = [line.replace('\t', '/') for line in lines if line.strip()]
    details = {'grouping': 'video: drone|folder_name|video; still: whole drone|folder_name acquisition; no singleton fallback in these data',
               'source_sha256': sha(SETUP / 'build_seadronessee_protocol.py')}
    results.append(counts('seadronessee_official_train_val', source_groups, official, 'source_video_or_still_acquisition', details))
    results.append(counts('seadronessee_source_disjoint_v1', source_groups, new_members, 'source_video_or_still_acquisition', details))
    legacy = next(x for x in results if x['protocol'] == 'wsodd_legacy_image_split_v1')
    assert legacy['total_groups'] == 147 and legacy['pairwise']['train_val']['shared_groups'] == 98
    assert legacy['pairwise']['train_val']['right_images_from_shared_groups'] == 1243
    for r in results:
        if 'disjoint' in r['protocol']:
            assert all(p['shared_groups'] == 0 and p['exact_member_overlap'] == 0 for p in r['pairwise'].values())
    write(ROOT / 'overlap/summary.json', {'created_utc': now(), 'status': 'completed', 'results': results, 'wsodd_gap_sensitivity': sensitivity, 'sources': sources,
          'interpretation_limit': 'Protocol groups and evaluation sets differ; overlap statistics do not isolate the causal share of any accuracy difference.'})
    print(json.dumps({'status': 'completed', 'protocols': len(results), 'legacy': legacy['pairwise']['train_val']}), flush=True)


if __name__ == '__main__':
    run()
