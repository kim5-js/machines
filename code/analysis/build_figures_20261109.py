"""Figures 5 and 9 of revision 20261109. Reads results and source images; trains and infers nothing.

Figure 5  example test inputs at severity levels 3 and 5, produced with the sealed evaluator's own transform functions
Figure 9  paired difference (fixed single view minus standard training) across severity levels, per seed, mean and 95% CI

Inputs (read-only)
  图表/20261109/绘图数据/severity_strategy.json            (from build_revision_data.py)
  02_当前实验/代码/autodl/evaluate_pretrained_sgd_robustness.py   (sealed transform source, hash checked)
  03_历史归档/论文/早期版本/figure_revision_20260921/assets      (the two example images and their ground truth)
Outputs
  图表/20261109/导出图/Figure5.{pdf,png,tiff}, Figure9.{pdf,png,tiff}, preview PNGs, figure_audit.json

Run with the scientific Python environment:
  python build_figures_20261109.py
"""
from pathlib import Path
import datetime, hashlib, importlib.util, json, math, os, sys

VERSION = '20261109'
RING3 = Path('D:/ring3')
FIG = RING3 / '01_当前论文/图表' / VERSION
DATA = FIG / '绘图数据' / 'severity_strategy.json'
OUT = FIG / '导出图'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(RING3 / '01_当前论文/排版与校验' / VERSION / '工作目录/mpl_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

FONT = font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'axes.unicode_minus': True, 'mathtext.fontset': 'stix',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .6, 'axes.edgecolor': '#777777',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
INK, GRAY = '#242424', '#777777'
# Okabe-Ito colours; every seed also has its own marker so colour is never the only cue.
SEED_STYLE = {'0': ('#0072B2', 'o'), '1': ('#D55E00', '^'), '2': ('#009E73', 's'), '42': ('#CC79A7', 'v'), '123': ('#E69F00', 'P'), '3407': ('#56B4E9', 'X')}
WIDTH, DPI = 18 / 2.54, 500
audit = {'version': VERSION, 'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'font_path': FONT,
         'dpi': DPI, 'matplotlib_version': matplotlib.__version__, 'outputs': []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name, extra=None):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer); fb = fig.bbox
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fb.x1 + 1 and box.y1 <= fb.y1 + 1, (name, item.get_text())
            assert item.get_fontsize() >= 8, (name, item.get_text(), item.get_fontsize())
    files = []
    for ext in ('pdf', 'png'):
        p = OUT / f'{name}.{ext}'; fig.savefig(p, dpi=DPI); files.append({'name': p.name, 'sha256': sha(p)})
    with Image.open(OUT / f'{name}.png') as im:
        im.convert('RGB').save(OUT / f'{name}.tiff', compression='tiff_lzw', dpi=(DPI, DPI))
    fig.savefig(OUT / f'{name}_preview.png', dpi=150)
    audit['outputs'].append({'figure': name, 'size_cm': [round(v * 2.54, 2) for v in fig.get_size_inches()], 'files': files, 'bounds_check': 'passed', 'min_font_pt': 8, **(extra or {})})
    plt.close(fig)


# ---------------------------------------------------------------- Figure 9
def figure9():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    recs = {(r['dataset'], r['kind'], r['level']): r for r in data['records'] if r['metric'] == 'mAP50-95'}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 12.4 / 2.54), sharey='row')
    fig.subplots_adjust(left=.085, right=.985, bottom=.185, top=.93, hspace=.60, wspace=.08)
    plotted = {}
    for ri, (ds, label) in enumerate((('WSODD', 'WSODD'), ('SeaDronesSee', 'SDS'))):
        seeds = data['seeds'][ds]
        for ci, (kind, cname) in enumerate((('low', 'low light'), ('glare', 'localized glare'))):
            ax = axes[ri, ci]
            ax.axvspan(3.5, 5.45, color='#EFEFEF', lw=0, zorder=0)
            ax.axhline(0, color='#606060', lw=.75, ls=(0, (4, 3)), zorder=1)
            levels = list(range(6))
            for si, seed in enumerate(seeds):
                color, marker = SEED_STYLE[seed]
                ys = [recs[(ds, kind, l)]['diff']['values'][si] for l in levels]
                xs = [l + (si - (len(seeds) - 1) / 2) * .045 for l in levels]
                ax.plot(xs, ys, color=color, marker=marker, ms=3.9, lw=.7, alpha=.78, mec='white', mew=.35, zorder=2)
            means = [recs[(ds, kind, l)]['diff']['mean'] for l in levels]
            lo = [recs[(ds, kind, l)]['diff']['mean'] - recs[(ds, kind, l)]['diff']['ci_low'] for l in levels]
            hi = [recs[(ds, kind, l)]['diff']['ci_high'] - recs[(ds, kind, l)]['diff']['mean'] for l in levels]
            ax.errorbar(levels, means, yerr=[lo, hi], fmt='D-', color=INK, mfc=INK, mec='white', mew=.45, ms=4.8, lw=.9,
                        elinewidth=1.0, capsize=3.2, capthick=.8, zorder=4)
            ticks = ['0\n(normal)'] + [f"{l}\n({recs[(ds, kind, l)]['standard_change_from_normal']['mean']:+.2f})".replace('-', '\u2212') for l in levels[1:]]
            ax.set_xticks(levels, ticks); ax.set_xlim(-.45, 5.45)
            ax.tick_params(labelsize=8.6, labelleft=(ci == 0))
            ax.set_title(f"({chr(97 + ri * 2 + ci)}) {label} · {cname} (n = {len(seeds)})", loc='left', fontsize=9.8, fontweight='bold', pad=7)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.set_axisbelow(True); ax.grid(axis='y', color='#E5E5E5', lw=.55)
            ax.yaxis.set_major_locator(MaxNLocator(6))
            if ci == 0:
                ax.set_ylabel('Paired difference (pp)', fontsize=9.3, labelpad=5)
            ax.set_xlabel('Severity level (change of the standard-training model, pp)', fontsize=8.8, labelpad=3)
            plotted[f'{label}|{kind}'] = dict(mean=means, ci_low=[recs[(ds, kind, l)]['diff']['ci_low'] for l in levels], ci_high=[recs[(ds, kind, l)]['diff']['ci_high'] for l in levels],
                                              seeds=seeds, values=[recs[(ds, kind, l)]['diff']['values'] for l in levels])
        # Label the shaded band once per row, in the glare panel where no data reach the top edge.
        axes[ri, 1].text(4.47, axes[ri, 1].get_ylim()[1], 'beyond training range', ha='center', va='top', fontsize=8, color='#555555')
    handles = [Line2D([], [], color=SEED_STYLE[s][0], marker=SEED_STYLE[s][1], lw=.7, ms=4.4, label=f'Seed {s}') for s in data['seeds']['WSODD']]
    handles.append(Line2D([], [], color=INK, marker='D', ms=4.6, lw=.9, label='Mean and 95% CI'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .012), ncol=7, frameon=False, fontsize=8.6, columnspacing=1.1, handlelength=1.5)
    save(fig, 'Figure9', {'data_sha256': sha(DATA), 'plotted': plotted,
                           'statistics': 'Within-seed paired differences; mean with t-based 95% confidence interval (n - 1 degrees of freedom).'})


# ---------------------------------------------------------------- Figure 5
def figure5(selected=None):
    source = RING3 / '02_当前实验/代码/autodl/evaluate_pretrained_sgd_robustness.py'
    assert sha(source) == 'cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f', 'transform source differs from the sealed evaluator'
    sys.path.insert(0, str(source.parent)); sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('sealed_robust', source)
    import ast, random
    tree = ast.parse(source.read_text(encoding='utf-8'))
    wanted = {'midpoint', 'deterministic_rng', 'low_light_transform', 'glare_transform'}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(body) == 4
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + body, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {'math': math, 'random': random, 'hashlib': hashlib, 'np': np, 'Path': Path}
    exec(compile(module, str(source), 'exec'), scope)  # the sealed functions themselves, not a reimplementation
    assets = RING3 / '03_历史归档/论文/早期版本/figure_revision_20260921/assets'
    # Round 2 passes its own image records (SeaDronesSee only); the default keeps the round-1 selection.
    if selected is None:
        selected = json.loads((assets / 'asset_audit.json').read_text(encoding='utf-8'))['degradation_selected_assets']
    namespace_root = 'ring3/cross-dataset/e0-e1/controlled-strong/v1'
    geometry = {'aspect_ratio': [1.4, 3.2], 'softness': [1.6, 2.4]}
    levels = json.loads(DATA.read_text(encoding='utf-8'))['level_parameters']
    columns = [('Normal', None, None), ('Low light, level 3', 'low', 3), ('Low light, level 5', 'low', 5),
               ('Localized glare, level 3', 'glare', 3), ('Localized glare, level 5', 'glare', 5)]
    height_cm = 5.62
    fig = plt.figure(figsize=(WIDTH, height_cm / 2.54))
    panel_w, gap, left = 3.42, .13, .22
    records = []
    for row, rec in enumerate(selected):
        key = 'SeaDronesSee' if rec['dataset'] == 'SDS' else 'WSODD'
        path = rec.get('path') or assets / 'sources' / rec['dataset'] / rec['image_id']
        normal = np.asarray(Image.open(path).convert('RGB'))
        h, w = normal.shape[:2]
        for col, (title, kind, lvl) in enumerate(columns):
            if kind is None:
                pixels = normal
            elif kind == 'low':
                b, g = levels['low'][lvl]
                pixels = scope['low_light_transform']({'brightness': [b, b], 'gamma': [g, g]})(normal, path)
            else:
                a, inten = levels['glare'][lvl]
                pixels = scope['glare_transform'](spec={'target_area_ratio': [a, a], 'intensity_255': [inten, inten]}, glare_config=geometry, namespace=namespace_root + '/' + key)(normal, path)
            x = left + col * (panel_w + gap); top = .58 + row * 2.80; ph = panel_w * h / w
            ax = fig.add_axes([x / 18, 1 - (top + ph) / height_cm, panel_w / 18, ph / height_cm])
            ax.imshow(pixels); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(.4); s.set_edgecolor('#aaaaaa')
            for box in rec['boxes']:
                x0, y0, x1, y1 = box['xyxy']
                r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#E69F00', linewidth=.6)
                r.set_path_effects([patheffects.Stroke(linewidth=.9, foreground='#302c20'), patheffects.Normal()]); ax.add_patch(r)
            label = f"({chr(97 + row * 5 + col)}) " + (f"{rec['dataset']} · " if col == 0 else '') + title
            fig.text(x / 18, 1 - (top - .10) / height_cm, label, fontsize=8.2, ha='left', va='bottom', color=INK)
            records.append(dict(dataset=rec['dataset'], image=rec['image_id'], condition=title, mean_gray=float(np.asarray(Image.fromarray(pixels).convert('L')).mean()),
                                rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    save(fig, 'Figure5', {'transform_source_sha256': sha(source), 'namespace_root': namespace_root, 'level_parameters': levels, 'panels': records,
                           'note': 'Ground-truth boxes only; no model predictions are drawn.'})


if __name__ == '__main__':
    figure9(); figure5()
    (OUT / 'figure_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([o['figure'] for o in audit['outputs']]))
