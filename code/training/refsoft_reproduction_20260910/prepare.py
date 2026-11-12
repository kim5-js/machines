"""Freeze current splits; call the original offline builder and build matched controls."""
from common import CODE, now, sha, write, load, verify_release, state
from pathlib import Path
import argparse
import importlib.metadata as metadata
import io
import shutil
import sys
import numpy as np
import yaml
from PIL import Image
import light_glare_dataset as old

ARMS = ['baseline', 'weights_only', 'bundle', 'duplicate', 'refsoft']
DATA = {
    'WSODD': ('wsodd_sequence_disjoint_v1.yaml', [3977, 1084, 2406]),
    'FloW': ('flow_source_temporal_disjoint_v1.yaml', [653, 203, 344]),
}
REF = dict(sample_ratio=0.10, low_light_prob=0.0, reflection_prob=1.0,
           allow_combined=False, reflection_max_area_ratio=0.035,
           reflection_min_area_factor=0.35, reflection_aspect_min=1.4,
           reflection_aspect_max=3.2, reflection_intensity_min=16.0,
           reflection_intensity_max=34.0)


def yaml_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; refusing to rebuild')
    release_hash = verify_release()
    locked = {}
    datasets = {}
    for key, (name, counts) in DATA.items():
        source = Path('/root/autodl-tmp/ring3_setup/autodl/datasets') / name
        config = yaml.safe_load(source.read_text())
        locked[str(source)] = sha(source)
        splits = {s: old.resolve_image_paths(source, s) for s in ['train', 'val', 'test']}
        assert [len(x) for x in splits.values()] == counts
        ids, content, manifests, records = {}, {}, {}, []
        for split, images in splits.items():
            ids[split] = {p.name for p in images}
            assert len(ids[split]) == len(images), (key, split, 'duplicate basename')
            content[split] = set()
            for p in images:
                label = old.resolve_label_path(p)
                digest = sha(p)
                content[split].add(digest)
                records.append({'split': split, 'image': str(p), 'image_sha256': digest,
                                'image_bytes': p.stat().st_size, 'image_mtime_ns': p.stat().st_mtime_ns,
                                'label': str(label), 'label_sha256': sha(label),
                                'label_mtime_ns': label.stat().st_mtime_ns})
            dest = root / 'manifests' / key / f'{split}.txt'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(''.join(str(p) + '\n' for p in images))
            locked[str(dest)] = sha(dest)
            manifests[split] = str(dest)
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert not ids[a] & ids[b], (key, a, b, 'overlapping identities')
            assert not content[a] & content[b], (key, a, b, 'identical image bytes')
        inventory = root / 'manifests' / key / 'inventory.json'
        write(inventory, records)
        locked[str(inventory)] = sha(inventory)

        # The unchanged historical function defines selection, RNG calls and image encoding.
        refroot = root / 'overlays' / key / 'refsoft'
        refyaml = old.build_augmented_dataset(data_yaml=source, output_root=refroot, seed=42, **REF)
        reflected = [Path(x) for x in (refroot / 'train_images.txt').read_text().splitlines()][len(splits['train']):]
        indices = sorted(np.random.default_rng(42).choice(len(splits['train']), size=max(1, int(len(splits['train']) * .1)), replace=False))
        assert len(reflected) == len(indices)
        duproot = root / 'overlays' / key / 'duplicate'
        duplicate = []
        pairs = []
        for index, rp in zip(indices, reflected, strict=True):
            original = splits['train'][index]
            lp = old.resolve_label_path(original)
            dp = duproot / 'aug_train/images' / rp.name
            dl = old.resolve_label_path(dp)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dl.parent.mkdir(parents=True, exist_ok=True)
            clean = np.asarray(Image.open(original).convert('RGB'))
            Image.fromarray(clean).save(dp, quality=95)
            shutil.copyfile(lp, dl)
            assert sha(lp) == sha(dl) == sha(old.resolve_label_path(rp))
            rng = np.random.default_rng(42 + int(index))
            assert old.choose_transform(rng, low_light_prob=0., reflection_prob=1., allow_combined=False) == (False, True)
            expected = old.apply_reflection_glare(clean, rng, max_area_ratio=.035,
                min_area_factor=.35, aspect_min=1.4, aspect_max=3.2, intensity_min=16., intensity_max=34.)
            buf = io.BytesIO()
            Image.fromarray(expected).save(buf, format=Image.open(rp).format, quality=95)
            assert buf.getvalue() == rp.read_bytes(), ('historical pixel/encoding mismatch', rp)
            assert rp.read_bytes() != dp.read_bytes(), ('reflection made no difference', rp)
            duplicate.append(dp)
            pairs.append({'index': int(index), 'source': str(original), 'refsoft': str(rp),
                          'duplicate': str(dp), 'label_sha256': sha(lp),
                          'refsoft_sha256': sha(rp), 'duplicate_sha256': sha(dp)})
            for p in [rp, old.resolve_label_path(rp), dp, dl]:
                locked[str(p)] = sha(p)
        overlay_yamls = {}
        for mode, extras in [('none', []), ('duplicate', duplicate), ('refsoft', reflected)]:
            directory = root / 'overlays' / key / mode
            directory.mkdir(parents=True, exist_ok=True)
            listing = directory / 'train_images.txt'
            listing.write_text(''.join(str(p) + '\n' for p in splits['train'] + extras))
            output = directory / 'dataset.yaml'
            # Absolute val/test paths correct the original builder's relative test relocation.
            yaml_write(output, {'path': str(directory), 'train': str(listing),
                               'val': manifests['val'], 'test': manifests['test'],
                               'nc': config['nc'], 'names': config['names']})
            for p in [listing, output]:
                locked[str(p)] = sha(p)
            overlay_yamls[mode] = str(output)
        pair_file = root / 'manifests' / key / 'paired_duplicates.json'
        write(pair_file, pairs)
        locked[str(pair_file)] = sha(pair_file)
        datasets[key] = {'source_yaml': str(source), 'counts': dict(zip(splits, counts)),
                         'classes': config['nc'], 'manifests': manifests, 'inventory': str(inventory),
                         'overlay_yamls': overlay_yamls, 'paired_duplicates': str(pair_file),
                         'duplicate_count': len(indices), 'content_disjoint': True,
                         'historical_reflection_exact_encoding_checked': len(indices)}

        # Smoke uses actual train images and true val images, never the fixed test set.
        selected = [splits['train'][int(i)] for i in indices[:16]]
        rest = [p for p in splits['train'] if p not in set(selected)][:16]
        for mode, extras in [('none', []), ('duplicate', duplicate[:4]), ('refsoft', reflected[:4])]:
            directory = root / 'smoke_inputs' / key / mode
            directory.mkdir(parents=True, exist_ok=True)
            tr, va = directory / 'train.txt', directory / 'val.txt'
            tr.write_text(''.join(str(p) + '\n' for p in selected + rest + extras))
            va.write_text(''.join(str(p) + '\n' for p in splits['val'][:8]))
            cfg = directory / 'data.yaml'
            yaml_write(cfg, {'path': str(directory), 'train': str(tr), 'val': str(va),
                             'nc': config['nc'], 'names': config['names']})
            for p in [tr, va, cfg]:
                locked[str(p)] = sha(p)

    jobs = []
    for seed in [42, 0, 1]:
        for dataset in DATA:
            for arm in ARMS:
                jobs.append({'key': f'{dataset}_{arm}_seed{seed}', 'dataset': dataset, 'arm': arm, 'seed': seed})
    p = {'schema': 'ring3.refsoft_reproduction.v1', 'created_utc': now(), 'root': str(root),
         'release_sha256': release_hash, 'datasets': datasets, 'arms': ARMS,
         'seeds': [42, 0, 1], 'augmentation_seed': 42, 'reflection': REF, 'jobs': jobs,
         'maximum_full_runs': len(jobs), 'locked_files': locked,
         'prediction': {'imgsz': 960, 'batch': 4, 'chunk_size': 4, 'confidence_floor': .001,
                        'nms_iou': .7, 'prediction_max_det': 300, 'evaluation_max_det': 100, 'device': 0},
         'environment': {name: metadata.version(name) for name in ['torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'scipy', 'opencv-python', 'ultralytics-thop']},
         'imported_ultralytics': 'vendored historical 8.3.182 (installed 8.4.124 unchanged)',
         'primary_checkpoint': 'best.pt by historical validation fitness',
         'secondary_checkpoint': 'best_map.pt by validation mAP50-95, first strict maximum',
         'test_policy': 'all pre-registered jobs, no test-driven selection'}
    write(root / 'protocol.json', p)
    state(root, 'prepared', maximum_full_runs=len(jobs))
    print({'root': str(root), 'datasets': {k: v['counts'] for k, v in datasets.items()},
           'duplicates': {k: v['duplicate_count'] for k, v in datasets.items()}, 'jobs': len(jobs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
