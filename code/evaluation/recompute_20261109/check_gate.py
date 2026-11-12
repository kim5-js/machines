"""Compare local reproductions with the sealed 20260924 values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
out = []
for g in read(OUT / 'gate_jobs.json'):
    e = read(OUT / 'evaluations' / g['key'] / 'evaluation.json')
    d = {k: round(100 * (e['metrics'][k] - g['sealed'][k]), 4) for k in g['sealed']}
    out.append(dict(key=g['key'], pixels_identical=e['input_pixels_sha256'] == g['sealed_input_pixels_sha256'], diff_pp=d, max_abs_diff_pp=max(abs(v) for v in d.values()), seconds=round(e['elapsed_seconds']), runtime=e['runtime']))
    print(out[-1])
write(OUT / 'gate_result.json', out)
