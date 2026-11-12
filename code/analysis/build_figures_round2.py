"""Round-2 figures of revision 20261109: Figures 1, 2, 4, 5, 7 and 8. Reads results and source images; trains and infers nothing.

Figure 1  study design and coverage (schematic)
Figure 2  SeaDronesSee annotation example and object-size distributions of the three evaluation sets
Figure 4  the seven matched from-scratch configurations (schematic)
Figure 5  example test inputs at severity levels 3 and 5 (two SeaDronesSee images), via build_figures_20261109.figure5
Figure 7  paired effects of the training strategies, per seed, mean and 95% CI
Figure 8  paired differences of the matched controls, per seed, mean and 95% CI
Only SeaDronesSee images are shown (the dataset is released under CC0 1.0); WSODD and FloW enter Figure 2 as size statistics only.

Inputs (read-only)
  图表/20261109/绘图数据/paired_effects.json        (from build_round2_data.py)
  02_当前实验/原始证据/recompute_20261109/data/<dataset>/ground_truth.json   (annotations of the evaluation sets)
  SeaDronesSee images 5186.jpg and 5695.jpg of the validation split (source-disjoint split used in this study)
Outputs
  图表/20261109/导出图/Figure{1,2,4,5,7,8}.{pdf,png,tiff}, preview PNGs, figure_audit_round2.json
Run with the scientific Python environment:  python build_figures_round2.py [figure numbers]
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_figures_20261109 as base  # shared style, save() with bounds and font checks, sealed transforms for Figure 5
from build_figures_20261109 import plt, np, Image, Line2D, Rectangle, patheffects, INK, GRAY, WIDTH, RING3, FIG, OUT, SEED_STYLE, audit, save, sha
from matplotlib.patches import FancyArrowPatch

PLOT = FIG / '绘图数据' / 'paired_effects.json'
GT = RING3 / '02_当前实验/原始证据/recompute_20261109/data'
SDS_IMAGES = RING3 / '02_当前实验/数据集原始包/SeaDronesSee Object Detection v2/Compressed Version/images'
BLUE, ORANGE, GREEN, DARK = '#0072B2', '#D55E00', '#009E73', '#4d4d4d'
MINUS = '−'


def effects():
    return json.loads(PLOT.read_text(encoding='utf-8'))


def draw_rows(ax, rows, metric, y_of, color_of, xlim, jitter=.16, step=2):
    ax.axvline(0, color=GRAY, lw=.7, ls=(0, (4, 3)), zorder=1)
    for r in rows:
        y = y_of[r['label']]; d = r[metric]
        n = len(d['per_seed'])
        for i, (s, v) in enumerate(zip(r['seeds'], d['per_seed'])):
            off = (i - (n - 1) / 2) * (2 * jitter / max(n - 1, 1))
            ax.plot(v, y + off, marker=SEED_STYLE[s][1], ms=4.2, mfc=color_of(r), mec='white', mew=.4, ls='none', zorder=3, alpha=.95)
        ax.plot([d['ci_low'], d['ci_high']], [y, y], color=INK, lw=1.0, zorder=4)
        for x in (d['ci_low'], d['ci_high']):
            ax.plot([x, x], [y - .11, y + .11], color=INK, lw=1.0, zorder=4)
        ax.plot(d['mean'], y, marker='D', ms=4.6, mfc=INK, mec='white', mew=.5, ls='none', zorder=5)
    ax.set_xlim(*xlim)
    ax.set_xticks([t for t in range(-20, 21, step) if xlim[0] <= t <= xlim[1]])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', labelsize=8.5, width=.6, length=3)


def seed_legend(fig, y, color=DARK):
    handles = [Line2D([], [], marker=SEED_STYLE[s][1], ls='none', ms=4.6, mfc=color, mec='white', mew=.4, label=f'Seed {s}') for s in ('0', '1', '2', '42', '123', '3407')]
    handles.append(Line2D([], [], marker='D', ls='-', lw=1.0, color=INK, ms=4.6, mfc=INK, mec='white', mew=.5, label='Mean and 95% CI'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y), ncol=7, frameon=False, fontsize=8.5, handletextpad=.3, columnspacing=1.1)


# ---------------------------------------------------------------- Figure 7
def figure7():
    rows = effects()['figure7']
    names = {('SeaDronesSee', 'E1'): 'SDS · Fixed single view\n(n = 3)', ('WSODD', 'E1'): 'WSODD · Fixed single view\n(n = 6)',
             ('FloW', 'E2'): 'FloW · Paired views\n(n = 6)', ('FloW', 'E3'): 'FloW · Paired views\n+ small-object weighting\n(n = 6)'}
    colors = {'SeaDronesSee': BLUE, 'WSODD': ORANGE, 'FloW': GREEN}
    for r in rows:
        r['label'] = names[(r['dataset'], r['new'])]
    y_of = {r['label']: len(rows) - 1 - i for i, r in enumerate(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 7.6 / 2.54), sharey=True)
    fig.subplots_adjust(left=.245, right=.985, bottom=.275, top=.90, wspace=.07)
    for ax, metric, title, xlim in ((axes[0], 'mAP50-95', '(a) mAP50–95', (-4.2, 3.2)), (axes[1], 'APsmall', r'(b) $AP_{\mathrm{small}}$', (-4.6, 4.2))):
        draw_rows(ax, rows, metric, y_of, lambda r: colors[r['dataset']], xlim)
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='bold', pad=6)
        ax.set_ylim(-.6, len(rows) - .4)
    axes[0].set_yticks(list(y_of.values())); axes[0].set_yticklabels(list(y_of.keys()), fontsize=8.8, linespacing=1.05)
    fig.text(.615, .135, 'Difference from standard training (pp)', ha='center', va='center', fontsize=9)
    seed_legend(fig, .0)
    save(fig, 'Figure7', {'data_sha256': sha(PLOT), 'rows': [dict(label=r['label'].replace('\n', ' '), n=r['mAP50-95']['n'],
         **{m: {k: r[m][k] for k in ('mean', 'ci_low', 'ci_high', 'per_seed')} for m in ('mAP50-95', 'APsmall')}) for r in rows]})


# ---------------------------------------------------------------- Figure 8
def figure8():
    rows = {(r['dataset'], r['new'], r['ref']): r for r in effects()['figure8']}
    layout = [('WSODD, from scratch: operation chain', [(('WSODD', 'B', 'A'), 'Loss-weight adjustment'), (('WSODD', 'C', 'B'), 'Attention + localization loss'),
                                                       (('WSODD', 'D', 'C'), 'Clean-copy addition'), (('WSODD', 'E', 'D'), 'Matched localized glare increment'),
                                                       (('WSODD', 'E', 'C'), 'GlareSoft vs component configuration'), (('WSODD', 'E', 'A'), 'Complete configuration vs scratch reference')]),
              ('WSODD, from scratch: single components', [(('WSODD', 'B+CBAM', 'B'), 'Attention only'), (('WSODD', 'B+Focal', 'B'), 'Localization loss only'),
                                                         (('WSODD', 'C', 'B+CBAM'), 'Localization loss added to attention'), (('WSODD', 'C', 'B+Focal'), 'Attention added to localization loss')]),
              ('Other matched controls', [(('WSODD', 'P-glare', 'P-clean'), 'Pretrained: GlareSoft vs clean copies'), (('WSODD', 'P-clean', 'E0'), 'Pretrained: clean copies vs standard training'),
                                         (('FloW', 'E2', 'E2C'), 'FloW: paired views vs identical paired views'), (('FloW', 'E2C', 'E0'), 'FloW: identical paired views vs standard training')])]
    y, y_of, used, heads, seps = 0.0, {}, [], [], []
    for gi, (head, items) in enumerate(layout):
        heads.append((y, head)); y -= .78
        for key, label in items:
            r = dict(rows[key], label=label); used.append(r); y_of[label] = y; y -= 1.0
        if gi < len(layout) - 1:
            seps.append(y + .42); y -= .34
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 14.4 / 2.54), sharey=True)
    fig.subplots_adjust(left=.385, right=.985, bottom=.135, top=.955, wspace=.06)
    color = lambda r: ORANGE if r['label'] == 'Matched localized glare increment' else BLUE
    for ax, metric, title, xlim in ((axes[0], 'mAP50-95', '(a) mAP50–95', (-5.7, 2.7)), (axes[1], 'APsmall', r'(b) $AP_{\mathrm{small}}$', (-8.2, 7.2))):
        draw_rows(ax, used, metric, y_of, color, xlim, jitter=.20, step=2 if metric == 'mAP50-95' else 4)
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='bold', pad=5)
        ax.set_ylim(y + .55, .35)
        for s in seps:
            ax.axhline(s, color='#bbbbbb', lw=.5, ls=(0, (2, 2)))
    axes[0].set_yticks(list(y_of.values())); labels = axes[0].set_yticklabels(list(y_of.keys()), fontsize=8.4)
    for t in labels:
        if t.get_text() == 'Matched localized glare increment':
            t.set_color(ORANGE); t.set_fontweight('bold')
    trans = axes[0].get_yaxis_transform()
    for yy, head in heads:
        axes[0].text(-1.25, yy - .30, head, transform=trans, ha='left', va='center', fontsize=8.6, fontweight='bold', color=INK, clip_on=False)
    fig.text(.685, .078, 'Paired difference (pp)', ha='center', va='center', fontsize=9)
    seed_legend(fig, .0, BLUE)
    save(fig, 'Figure8', {'data_sha256': sha(PLOT), 'rows': [dict(label=r['label'], dataset=r['dataset'], new=r['new'], ref=r['ref'],
         **{m: {k: r[m][k] for k in ('mean', 'ci_low', 'ci_high', 'per_seed')} for m in ('mAP50-95', 'APsmall')}) for r in used]})


# ---------------------------------------------------------------- schematics
def box(ax, x, y, w, h, title, lines, color=DARK, fs=8.6, lw=.9, title_fs=9.2):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw))
    ax.text(x + w / 2, y + h - .30, title, ha='center', va='top', fontsize=title_fs, fontweight='bold', color=color)
    ax.text(x + w / 2, y + h - .30 - .62, '\n'.join(lines), ha='center', va='top', fontsize=fs, color=color, linespacing=1.32)


def arrow(ax, p, q, color=GRAY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='->', mutation_scale=9, lw=.8, color=color, shrinkA=0, shrinkB=0))


def canvas(height_cm):
    fig = plt.figure(figsize=(WIDTH, height_cm / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 18); ax.set_ylim(0, height_cm); ax.axis('off')
    return fig, ax


def figure1():
    fig, ax = canvas(8.3)
    w, h = 8.1, 2.95
    box(ax, .55, 4.75, w, h, '(a) Detector reference', ['SDS · WSODD · FloW', 'Four detectors, one seed each', 'Descriptive scores · Section 4.1'], '#555555')
    box(ax, 9.35, 4.75, w, h, '(b) Multi-seed stability', ['SDS (3) · WSODD (6) · FloW (6)', 'Training change vs standard training', 'Within-seed differences · Section 4.2'], BLUE)
    box(ax, .55, 1.15, w, h, '(c) Matched attribution', ['WSODD (6) · FloW (6)', 'Clean vs GlareSoft copies · identical vs degraded views', 'Operational contrasts · Section 4.3'], ORANGE)
    box(ax, 9.35, 1.15, w, h, '(d) Condition response', ['SDS (3) · WSODD (6) · FloW (6)', 'Normal · low light · localized glare, severity levels 1–5', 'Condition interactions · Section 4.4'], '#2b2b2b')
    ax.text(9, .52, 'One official COCOeval protocol for all reported scores; seeds per dataset in parentheses', ha='center', va='center', fontsize=8.8, color='#555555')
    save(fig, 'Figure1', {'note': 'Schematic; no data plotted.'})


def figure4():
    fig, ax = canvas(11.6)
    ax.text(.45, 11.15, '(a) Configuration sequence', fontsize=9.5, fontweight='bold', color=INK, va='center')
    w, h, y = 4.5, 2.25, 8.35
    box(ax, .55, y, w, h, 'Scratch reference', ['YOLOv8s from scratch', 'CIoU · box / DFL: 7.5 / 1.5'])
    box(ax, 6.75, y, w, h, 'Adjusted weights', ['YOLOv8s + CIoU', 'box / DFL: 4.0 / 1.3'])
    box(ax, 12.95, y, w, h, 'Attention +\nlocalization loss', ['', 'Adjusted weights', 'CBAM + Focal-EIoU variant'])
    arrow(ax, (5.05, y + h / 2), (6.75, y + h / 2)); arrow(ax, (11.25, y + h / 2), (12.95, y + h / 2))
    ax.text(.45, 7.55, '(b) Single components', fontsize=9.5, fontweight='bold', color=INK, va='center')
    y2, h2 = 5.55, 1.55
    box(ax, .9, y2, 5.2, h2, 'Attention only', ['Adjusted weights + CBAM'], '#555555')
    box(ax, 6.9, y2, 5.6, h2, 'Localization loss only', ['Adjusted weights + Focal-EIoU variant'], '#555555')
    arrow(ax, (7.9, y), (4.6, y2 + h2), '#999999'); arrow(ax, (9.7, y), (9.7, y2 + h2), '#999999')
    ax.text(.45, 4.75, '(c) Matched appended samples', fontsize=9.5, fontweight='bold', color=INK, va='center')
    ax.text(9, 4.18, '397 images selected once; 4374 images in each expanded set', ha='center', va='center', fontsize=8.8, color=INK)
    y3, h3 = 1.75, 1.75
    box(ax, 1.55, y3, 6.6, h3, 'Clean copies', ['Component model + 397 clean copies', 'RGB encoding, quality 95'], BLUE)
    box(ax, 9.85, y3, 6.6, h3, 'GlareSoft copies', ['Component model + 397 glare copies', 'RGB encoding, quality 95'], ORANGE)
    arrow(ax, (15.2, y), (13.55, 4.38), '#999999')
    arrow(ax, (8.2, 3.95), (5.2, y3 + h3), BLUE); arrow(ax, (9.8, 3.95), (12.8, y3 + h3), ORANGE)
    ax.plot([4.85, 4.85, 13.15, 13.15], [y3 - .15, y3 - .45, y3 - .45, y3 - .15], color=INK, lw=.8)
    ax.text(9, .92, 'Matched localized glare increment', ha='center', va='center', fontsize=9.2, fontweight='bold', color=INK)
    ax.text(9, .40, 'Same selected images, copy counts, processing and labels · Seeds 0, 1, 2, 42, 123, 3407', ha='center', va='center', fontsize=8.6, color='#555555')
    save(fig, 'Figure4', {'note': 'Schematic; no data plotted.'})


# ---------------------------------------------------------------- Figure 2
def sds_boxes(name):
    gt = json.loads((GT / 'SeaDronesSee/ground_truth.json').read_text(encoding='utf-8'))
    im = next(i for i in gt['images'] if Path(i['file_name']).name == name)
    cats = {c['id']: c['name'] for c in gt['categories']}
    return im, [dict(xyxy=[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]], name=cats[a['category_id']]) for a in gt['annotations'] if a['image_id'] == im['id']]


def sds_path(name):
    hits = list(SDS_IMAGES.glob(f'*/{name}'))
    assert len(hits) == 1, (name, hits)
    return hits[0]


def gt_box(ax, b, lw=.7):
    x0, y0, x1, y1 = b['xyxy']
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#E69F00', linewidth=lw)
    r.set_path_effects([patheffects.Stroke(linewidth=lw + .5, foreground='#302c20'), patheffects.Normal()]); ax.add_patch(r)


def figure2():
    name, crop = '5186.jpg', (1450, 1460, 2850, 1980)
    meta, boxes = sds_boxes(name)
    pixels = np.asarray(Image.open(sds_path(name)).convert('RGB'))
    assert pixels.shape[1] == meta['width'] and pixels.shape[0] == meta['height']
    height_cm = 10.9
    fig = plt.figure(figsize=(WIDTH, height_cm / 2.54))
    # (a) full image
    wa = 7.1; ha = wa * meta['height'] / meta['width']; top = .62
    ax = fig.add_axes([.35 / 18, 1 - (top + ha) / height_cm, wa / 18, ha / height_cm]); ax.imshow(pixels); ax.set_xticks([]); ax.set_yticks([])
    for b in boxes:
        gt_box(ax, b, .6)
    ax.add_patch(Rectangle((crop[0], crop[1]), crop[2] - crop[0], crop[3] - crop[1], fill=False, edgecolor='white', linewidth=.9, linestyle=(0, (4, 3))))
    for s in ax.spines.values():
        s.set_linewidth(.4); s.set_edgecolor('#aaaaaa')
    fig.text(.35 / 18, 1 - (top - .12) / height_cm, '(a) SDS · validation', fontsize=9, ha='left', va='bottom', color=INK)
    # (b) enlarged region
    wb = 9.9; hb = wb * (crop[3] - crop[1]) / (crop[2] - crop[0])
    bx = fig.add_axes([7.75 / 18, 1 - (top + hb) / height_cm, wb / 18, hb / height_cm])
    bx.imshow(pixels[crop[1]:crop[3], crop[0]:crop[2]], extent=(crop[0], crop[2], crop[3], crop[1])); bx.set_xticks([]); bx.set_yticks([])
    labelled = set()
    for b in boxes:
        x0, y0, x1, y1 = b['xyxy']
        if x0 >= crop[0] and y0 >= crop[1] and x1 <= crop[2] and y1 <= crop[3]:
            gt_box(bx, b, .9)
            if b['name'] not in labelled and b['name'] in ('boat', 'swimmer'):
                labelled.add(b['name'])
                bx.text(x0, y0 - 8, b['name'], fontsize=8.2, color=INK, ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='#555555', linewidth=.4, pad=1.2))
    for s in bx.spines.values():
        s.set_linewidth(.4); s.set_edgecolor('#aaaaaa')
    fig.text(7.75 / 18, 1 - (top - .12) / height_cm, '(b) Enlarged region', fontsize=9, ha='left', va='bottom', color=INK)
    # (c) object-size distributions of the three evaluation sets
    cx = fig.add_axes([1.35 / 18, .98 / height_cm, 16.3 / 18, 3.95 / height_cm])
    edges = np.logspace(math.log10(4), math.log10(1024), 49)
    stats = {}
    for ds, label, color, ls in (('SeaDronesSee', 'SDS validation', BLUE, '-'), ('WSODD', 'WSODD test', ORANGE, '--'), ('FloW', 'FloW test', GREEN, '-.')):
        gt = json.loads((GT / ds / 'ground_truth.json').read_text(encoding='utf-8'))
        side = np.sqrt(np.array([a['bbox'][2] * a['bbox'][3] for a in gt['annotations']], dtype=float))
        hist, _ = np.histogram(side, bins=edges)
        cx.step(edges[:-1], 100 * hist / len(side), where='post', color=color, lw=1.2, ls=ls, label=f'{label} (n = {len(side):,})'.replace(',', ' ' if False else ','))
        stats[ds] = dict(objects=int(len(side)), small=int((side ** 2 < 32 ** 2).sum()), medium=int(((side ** 2 >= 32 ** 2) & (side ** 2 < 96 ** 2)).sum()), large=int((side ** 2 >= 96 ** 2).sum()),
                         below_range=int((side < edges[0]).sum()), above_range=int((side >= edges[-1]).sum()))
    for x in (32, 96):
        cx.axvline(x, color=GRAY, lw=.7, ls=(0, (4, 3)))
    ymax = cx.get_ylim()[1]
    for x, t in ((11, 'small'), (55, 'medium'), (310, 'large')):
        cx.text(x, ymax * .93, t, fontsize=8.6, color='#555555', ha='center', va='center')
    cx.set_xscale('log'); cx.set_xlim(4, 1024); cx.set_xticks([4, 8, 16, 32, 64, 96, 128, 256, 512, 1024]); cx.set_xticklabels(['4', '8', '16', '32', '64', '96', '128', '256', '512', '1024'])
    cx.minorticks_off(); cx.tick_params(labelsize=8.5, width=.6, length=3)
    cx.set_xlabel('Square root of bounding-box area (px, original image)', fontsize=9); cx.set_ylabel('Share of objects (%)', fontsize=9)
    for side_ in ('top', 'right'):
        cx.spines[side_].set_visible(False)
    cx.legend(frameon=False, fontsize=8.6, loc='upper right', bbox_to_anchor=(1.0, .86), handlelength=2.6)
    fig.text(.35 / 18, (.98 + 3.95 + .28) / height_cm, '(c) Object size in the three evaluation sets', fontsize=9, ha='left', va='bottom', color=INK)
    save(fig, 'Figure2', {'image': name, 'image_sha256': sha(sds_path(name)), 'crop_xyxy': list(crop), 'boxes': len(boxes), 'size_statistics': stats,
                          'note': 'Only a SeaDronesSee image is shown (CC0 1.0 per the dataset repository). WSODD and FloW appear as annotation statistics.'})


# ---------------------------------------------------------------- Figure 5 (two SeaDronesSee images)
def figure5():
    selected = []
    for name in ('5186.jpg', '5695.jpg'):
        meta, boxes = sds_boxes(name)
        selected.append(dict(dataset='SDS', image_id=name, boxes=boxes, path=sds_path(name)))
    base.figure5(selected=selected)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['1', '2', '4', '5', '7', '8']
    for n in todo:
        {'1': figure1, '2': figure2, '4': figure4, '5': figure5, '7': figure7, '8': figure8}[n]()
    path = OUT / 'figure_audit_round2.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'outputs': []}
    names = {o['figure'] for o in audit['outputs']}
    audit['outputs'] = [o for o in old['outputs'] if o['figure'] not in names] + audit['outputs']
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(sorted(o['figure'] for o in audit['outputs'])))
