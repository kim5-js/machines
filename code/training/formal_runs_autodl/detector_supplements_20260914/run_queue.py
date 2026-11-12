"""Serial, inference-only queue with immutable inputs and exclusive process lock."""
import fcntl
import json
import os
import subprocess
import sys
import traceback
from prepare import ROOT, now, sha, write


def main():
    with (ROOT / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        req = json.loads((ROOT / 'request.json').read_text())
        for job in req['jobs']:
            out = ROOT / 'evaluations' / job['key']
            if (out / 'result.json').exists():
                result = json.loads((out / 'result.json').read_text())
                assert result['request_sha256'] == sha(ROOT / 'request.json') and result['status'] == 'completed'
                assert sha(out / 'predictions.json.gz') == result['predictions_sha256']
                continue
            assert not out.exists(), 'Preserve incomplete attempt for investigation'
            log = ROOT / 'logs' / (job['key'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x') as f:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'code/evaluate.py'), job['key']], stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                write(ROOT / 'state.json', {'status': 'evaluating', 'job': job['key'], 'pid': os.getpid(), 'child_pid': child.pid, 'updated_utc': now()})
                assert child.wait() == 0, 'Evaluation failed: ' + str(log)
        results = [json.loads((ROOT / 'evaluations' / j['key'] / 'result.json').read_text()) for j in req['jobs']]
        assert len(results) == 12 and all(r['status'] == 'completed' for r in results)
        write(ROOT / 'summary.json', {'status': 'completed', 'jobs': 12, 'results': results, 'completed_utc': now(), 'request_sha256': sha(ROOT / 'request.json')})
        write(ROOT / 'state.json', {'status': 'completed', 'jobs': 12, 'completed_utc': now()})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(ROOT / 'state.json', {'status': 'failed', 'utc': now(), 'traceback': traceback.format_exc()})
        raise
