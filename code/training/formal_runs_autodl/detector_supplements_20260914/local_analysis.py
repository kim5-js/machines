"""Compute both split protocols locally from source files checked against AutoDL."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys

PROJECT = Path(r'D:/ring3')
WORK = PROJECT / 'artifacts/detector_supplements_20260914'
OUT = PROJECT / 'deliverables/detector_supplements_20260914'
sys.path.insert(0, str(WORK / 'local_dependencies'))
os.environ['SUPPLEMENT_ROOT'] = str(WORK)
os.environ['MPLCONFIGDIR'] = str(WORK / 'matplotlib')
from overlap import counts, module
from prepare import sha, write, now


def main():
    inventory = json.loads((WORK / 'remote_input_inventory.json').read_text(encoding='utf-8'))
    assert inventory['training_still_paused'] and len(inventory['runs']) == 12
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / 'evidence/inputs'
    verified = {}
    def source(rel):
        p = PROJECT / rel
        expected = inventory['source_hashes'][rel]
        assert sha(p) == expected, 'Source mismatch: ' + rel
        verified[rel] = expected
        dst = inputs / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        return p
    manifests = 'autodl/manifests/'
    ws = module('local_ws_groups', source('autodl/audit_wsodd_split.py'))
    sds = module('local_sds_groups', source('autodl/build_seadronessee_protocol.py'))
    results = []
    ws_members = {}
    for protocol, actual in [('wsodd_legacy_image_split_v1', 'wsodd_current'), ('wsodd_sequence_disjoint_v1', 'wsodd_sequence_disjoint_v1')]:
        source(manifests + protocol + '/protocol.json')
        members = {}
        for split, actual_names in inventory['memberships'][actual]['splits'].items():
            lines = source(manifests + protocol + '/' + split + '.txt').read_text(encoding='utf-8-sig').splitlines()
            members[split] = [Path(x).name.casefold() for x in lines if x.strip()]
            assert set(members[split]) == {x.casefold() for x in actual_names}
            assert len(members[split]) == len(actual_names)
        ws_members[protocol] = members
    # Preserve original case for the timestamp parser's literal WIN_ prefix.
    all_names = sorted(set().union(*(set(v) for v in inventory['memberships']['wsodd_current']['splits'].values())))
    assert len(all_names) == 7467
    ws_details = {'source': 'filename-derived temporal groups from the frozen project rule', 'gap_seconds': 60,
                  'numeric_names': 'one burst', 'unknown_names': 'singleton', 'official_sequence_ids': False,
                  'interpretation': 'Sequence membership is inferred from filenames; it is not a manual physical-scene annotation.'}
    groups = ws.build_groups([Path(n) for n in all_names], 60.0)
    for protocol, members in ws_members.items():
        results.append(counts(protocol, groups, members, 'inferred_temporal_sequence', ws_details))
    sensitivity = []
    for gap in (30., 120.):
        groups_other = ws.build_groups([Path(n) for n in all_names], gap)
        sensitivity.append(counts('wsodd_legacy_gap' + str(int(gap)), groups_other, ws_members['wsodd_legacy_image_split_v1'], 'inferred_temporal_sequence', dict(ws_details, gap_seconds=gap)))
    source(manifests + 'flow_source_temporal_disjoint_v1/protocol.json')
    p = source(manifests + 'flow_source_temporal_disjoint_v1/source_manifest.tsv')
    rows = list(csv.DictReader(p.open(encoding='utf-8-sig'), delimiter='\t'))
    fg = {r['image_name'].casefold(): r['source_group'] for r in rows}
    assert len(fg) == len(rows) == 1200
    for name, col in [('flow_current', 'current_split'), ('flow_source_temporal_disjoint_v1', 'split')]:
        members = {s: [x.casefold() for x in values] for s, values in inventory['memberships'][name]['splits'].items()}
        for split, values in members.items():
            assert set(values) == {r['image_name'].casefold() for r in rows if r[col] == split}
        results.append(counts(name, fg, members, 'source_acquisition_session', {'source': 'official XML-derived frozen source/session manifest; renamed IDs are not time'}))
    sg, official = {}, {'train': [], 'val': []}
    for origin in ('train', 'val'):
        p = source('SeaDronesSee Object Detection v2/Compressed Version/annotations/instances_' + origin + '.json')
        raw = json.loads(p.read_text(encoding='utf-8-sig'))
        for im in raw['images']:
            key = origin + '/' + Path(im['file_name']).name
            sg[key] = sds.group_key(im, origin)
            official[origin].append(key)
        assert {x.split('/', 1)[1] for x in official[origin]} == set(inventory['memberships']['seadronessee_current']['splits'][origin])
    assert len(sg) == 10477
    source(manifests + 'seadronessee_source_disjoint_v1/protocol.json')
    new_members = {}
    for split in ('train', 'val'):
        p = source(manifests + 'seadronessee_source_disjoint_v1/' + split + '.tsv')
        new_members[split] = [line.replace('\t', '/') for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        assert {x.split('/', 1)[1] for x in new_members[split]} == set(inventory['memberships']['seadronessee_source_disjoint_v1']['splits'][split])
    detail = {'video': 'drone|folder_name|video', 'still': 'whole drone|folder_name acquisition', 'singleton_fallback_count': sum(x.startswith('singleton:') for x in sg.values())}
    results.append(counts('seadronessee_official_train_val', sg, official, 'source_video_or_still_acquisition', detail))
    results.append(counts('seadronessee_source_disjoint_v1', sg, new_members, 'source_video_or_still_acquisition', detail))
    legacy, strict = results[:2]
    assert legacy['total_groups'] == 147 and legacy['pairwise']['train_val']['shared_groups'] == 98
    assert legacy['pairwise']['train_val']['right_images_from_shared_groups'] == 1243
    for r in results:
        if 'disjoint' in r['protocol']:
            assert all(p['shared_groups'] == 0 and p['exact_member_overlap'] == 0 for p in r['pairwise'].values())
    summary = {'status': 'completed', 'created_utc': now(), 'results': results, 'wsodd_gap_sensitivity': sensitivity,
               'source_hashes_verified_against_autodl': verified, 'inventory_sha256': sha(WORK / 'remote_input_inventory.json'),
               'limitations': ['Groups have dataset-specific definitions.', 'No causal share of accuracy differences is identified.', 'No claim of exact-content duplicate absence from filename counts alone.']}
    write(OUT / 'split_overlap_summary.json', summary)
    for name in [r['protocol'] for r in results + sensitivity]:
        src = WORK / 'overlap' / name
        dest = OUT / 'evidence/overlap' / name
        dest.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            shutil.copyfile(p, dest / p.name)
    shutil.copyfile(WORK / 'remote_input_inventory.json', OUT / 'evidence/remote_input_inventory.json')
    columns = ['protocol', 'unit', 'train_images', 'val_images', 'train_groups', 'val_groups', 'shared_train_val_groups', 'val_images_from_train_groups', 'val_overlap_percent', 'exact_member_overlap']
    with (OUT / 'split_overlap_table.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader()
        for r in results:
            v = r['pairwise']['train_val']
            w.writerow({'protocol': r['protocol'], 'unit': r['unit'], 'train_images': r['split_images']['train'], 'val_images': r['split_images']['val'], 'train_groups': v['left_groups'], 'val_groups': v['right_groups'], 'shared_train_val_groups': v['shared_groups'], 'val_images_from_train_groups': v['right_images_from_shared_groups'], 'val_overlap_percent': 100 * v['right_overlap_fraction'], 'exact_member_overlap': v['exact_member_overlap']})
    import evaluate
    write(WORK / 'local_coco_self_test.json', evaluate.self_test())
    make_figures(legacy, strict, OUT)
    print(json.dumps({'status': 'completed', 'sources_verified': len(verified), 'protocols': [{k: r[k] for k in ('protocol', 'split_images', 'split_groups')} | {'train_val': {k: v for k, v in r['pairwise']['train_val'].items() if k != 'shared_group_ids'}} for r in results], 'sensitivity': [{'gap': r['definition']['gap_seconds'], 'groups': r['total_groups'], 'shared': r['pairwise']['train_val']['shared_groups'], 'val_exposed': r['pairwise']['train_val']['right_images_from_shared_groups']} for r in sensitivity]}))


def make_figures(legacy, strict, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={'width_ratios': [1.4, 1]})
    a = legacy['pairwise']['train_val']; b = strict['pairwise']['train_val']
    labels = ['Legacy image split', 'Sequence-disjoint split']
    segments = [('Train only', [a['left_only_groups'], b['left_only_groups']], '#35618d'),
                ('Shared train/val', [a['shared_groups'], b['shared_groups']], '#c36436'),
                ('Val only', [a['right_only_groups'], b['right_only_groups']], '#73a39e'),
                ('Test only', [0, strict['split_groups']['test']], '#c6ccd3')]
    left = [0, 0]
    for label, values, color in segments:
        ax[0].barh([1, 0], values, left=left, label=label, color=color, height=.48)
        for i, value in enumerate(values):
            if value >= 10:
                ax[0].text(left[i] + value/2, 1-i, str(value), va='center', ha='center', color='white' if label in ['Train only', 'Shared train/val'] else '#202020')
        left = [x+y for x, y in zip(left, values)]
    ax[0].set_yticks([1, 0], labels)
    ax[0].set_xlabel('Number of inferred sequence groups')
    ax[0].set_xlim(0, 150); ax[0].set_ylim(-.6, 1.65)
    ax[0].set_title('(a) Membership of all 147 groups', loc='left')
    ax[0].legend(loc='upper left', bbox_to_anchor=(0, -.23), ncol=2, frameon=False, fontsize=10)
    ax[0].annotate('2 val-only groups', (146, 1), xytext=(113, 1.46), arrowprops={'arrowstyle': '-', 'color': '#444'}, ha='center', fontsize=10)
    values = [a['right_overlap_fraction'] * 100, b['right_overlap_fraction'] * 100]
    ax[1].bar([0, 1], values, color=['#c36436', '#73a39e'], width=.58)
    ax[1].set_xticks([0, 1], ['Legacy', 'Disjoint'])
    ax[1].set_ylabel('Validation images from train-seen groups (%)')
    ax[1].set_ylim(0, 130)
    for i, (val, text) in enumerate(zip(values, ['1,243 / 1,245\n99.84%', '0 / 1,084\n0%'])):
        ax[1].text(i, val+3, text, ha='center', va='bottom', fontsize=11)
    ax[1].set_title('(b) Validation exposure', loc='left')
    for item in ax:
        item.spines[['top', 'right']].set_visible(False)
    fig.suptitle('WSODD split overlap under the fixed 60-second grouping rule', y=.99, fontsize=13)
    fig.subplots_adjust(left=.20, right=.98, top=.82, bottom=.29, wspace=.65)
    for ext in ['png', 'pdf', 'svg', 'tiff']:
        fig.savefig(out / ('WSODD_split_overlap.'+ext), dpi=400, facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    ax.text(.15, 2.75, 'Observed sequence-group membership in the WSODD legacy split', fontsize=13)
    categories = [(2.0, '98 shared groups', ['Train', 'Validation'], '#c36436'), (1.1, '47 train-only groups', ['Train'], '#35618d'), (.25, '2 val-only groups', ['Validation'], '#73a39e')]
    for y, label, targets, color in categories:
        ax.text(.5, y, label, va='center', fontsize=12)
        for idx, target in enumerate(targets):
            yy = y + (.22 if idx == 0 else -.22) if len(targets) == 2 else y
            ax.annotate('', xy=(7, yy), xytext=(4.1, y), arrowprops={'arrowstyle': '->', 'lw': 1.8, 'color': color})
            ax.text(7.2, yy, target, va='center', color=color, fontsize=12)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg', 'tiff']:
        fig.savefig(out / ('WSODD_sequence_membership.'+ext), dpi=400, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
