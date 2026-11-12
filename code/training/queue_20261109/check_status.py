"""Print a compact status of the training queue and the post-training evaluation on the Vast instance."""
import json, os, glob, subprocess, time
TR = '/workspace/train_20261109'
st = json.load(open(f'{TR}/queue.status.json'))
q = json.load(open(f'{TR}/queue.json'))['jobs']
done = [j for j in q if os.path.exists(j['done'])]
blocks = {}
for j in q:
    b = blocks.setdefault(j['block'], [0, 0]); b[1] += 1
    if os.path.exists(j['done']): b[0] += 1
print('utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'queue', st.get('state', 'running'), 'done', len(done), '/', len(q), 'failed', st.get('failed'), 'elapsed_h', st.get('elapsed_hours'))
print('blocks', {k: f'{v[0]}/{v[1]}' for k, v in blocks.items()})
for k, v in st.get('running', {}).items():
    j = next(x for x in q if x['key'] == k)
    p = os.path.join(j['rundir'], 'progress.json'); ep = '?'
    if os.path.exists(p):
        d = json.load(open(p))
        ep = f"{d.get('epoch')}/{d.get('epochs', d.get('target_epochs'))} sec/ep {round(float(d.get('recent_seconds_per_epoch', d.get('epoch_seconds', 0)) or 0))}"
    else:
        r = os.path.join(j['rundir'], 'results.csv')
        if os.path.exists(r):
            ep = f"{max(0, len(open(r).read().strip().splitlines()) - 1)} epochs in results.csv"
    print('running', k, 'gpu', v['gpu'], 'h', v['hours'], 'epoch', ep)
fr = os.path.join(TR, 'failed_runs')
print('failed_runs dir', sorted(os.listdir(fr)) if os.path.isdir(fr) else [])
pe = os.path.join(TR, 'post_train_eval.log')
print('post_eval_log', open(pe).read().strip().splitlines()[-1:] if os.path.exists(pe) else None)
print('runner', subprocess.run("pgrep -fc '[t]rain_queue_vast'", shell=True, capture_output=True, text=True).stdout.strip(),
      'watcher', subprocess.run("pgrep -fc '[v]env/bin/python /workspace/train_20261109/post_train_eval.py'", shell=True, capture_output=True, text=True).stdout.strip())
