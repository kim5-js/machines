"""Replay of a September matched-control job with the 20261109 extension code (server side).

The extension protocol lists only the new jobs, so a September job is passed to the same train() function directly.
train.py of the extension differs from the September train.py by the six lines that add the two single-component
arms; for the five original arms the recipe, model, seed handling and callbacks are the same code.
The run goes to the extension root, not to the September release, so no original run directory is touched.
Usage (cwd = extension code directory): python replay_matched_original.py <arm> <seed>
"""
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/refsoft_extension_20261109/release_v1')
sys.path.insert(0, str(ROOT / 'code'))
import train  # noqa: E402

arm, seed = sys.argv[1], int(sys.argv[2])
assert arm in {'baseline', 'weights_only', 'bundle', 'duplicate', 'refsoft'} and seed in (0, 1, 42)
train.train(ROOT, dict(key=f'WSODD_{arm}_seed{seed}', dataset='WSODD', arm=arm, seed=seed))
