"""Run a training script on the GPU named by CUDA_VISIBLE_DEVICES, unchanged.

The archived recipes pass device=0, and Ultralytics then rewrites CUDA_VISIBLE_DEVICES to "0", which would
put every job on physical GPU 0. Initializing CUDA here first fixes the device mapping, so the script's
cuda:0 is the physical GPU chosen by the queue. The script, its arguments and the recipe are not modified.
Usage: python gpu_pin.py <script.py> [script args...]
"""
import json, os, runpy, sys
from pathlib import Path

visible = os.environ.get('CUDA_VISIBLE_DEVICES')
assert visible is not None and visible.isdigit(), 'CUDA_VISIBLE_DEVICES must name one GPU'
import torch  # noqa: E402
torch.cuda.init()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
props = torch.cuda.get_device_properties(0)
print('GPU_PIN ' + json.dumps(dict(cuda_visible_devices=visible, name=props.name, uuid=str(getattr(props, 'uuid', '')))), flush=True)
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
