#!/usr/bin/env python3
"""KV260 on-board VART inference: xmodel + host-preprocessed input_640.npy -> DPU -> yolo_decode
Usage: LD_PRELOAD=/usr/local/lib/libxcl_stub.so python3 board_infer.py [npy_path]
"""
import os, sys, json
import numpy as np
import xir, vart

BASE = os.path.dirname(os.path.abspath(__file__))
XM = os.path.join(BASE, 'YoloV8DetectExport.xmodel')
NPY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'input_640.npy')
sys.path.insert(0, BASE)
from yolo_decode import decode_all

print('[board_infer] loading xmodel:', XM)
g = xir.Graph.deserialize(XM)
# The root subgraph has no runner attribute; the DPU subgraph is among its
# children (fingerprint-only compile). Traverse children to find device=='DPU'.
root = g.get_root_subgraph()
sub = None
for s in root.get_children():
    if s.has_attr('device') and s.get_attr('device') == 'DPU':
        sub = s
        break
if sub is None:
    raise RuntimeError('no DPU subgraph found')
print('[board_infer] root subgraph:', sub.get_name(), 'attr dtype:', sub.get_attr('dpu') if sub.has_attr('dpu') else '?')

runner = vart.Runner.create_runner(sub, "run")
in_t = runner.get_input_tensors()
out_t = runner.get_output_tensors()
print('[board_infer] INPUT :', [(t.name, t.dims, str(t.dtype)) for t in in_t])
print('[board_infer] OUTPUT:', [(t.name, t.dims, str(t.dtype)) for t in out_t])

inp = np.load(NPY)
print('[board_infer] loaded input', inp.shape, inp.dtype, 'range', float(inp.min()), float(inp.max()))

# Input buffer: VART accepts float32 (quantized internally per fix_point)
input_data = inp.astype(np.float32) if inp.dtype != np.float32 else inp
output_data = [np.empty(t.dims, dtype=np.float32) for t in out_t]

import time
t0 = time.time()
job = runner.execute_async([input_data], output_data)
runner.wait(job)
elapsed = time.time() - t0
print(f'[board_infer] DPU executed in {elapsed*1000:.1f} ms')

for i, o in enumerate(output_data):
    print(f'[board_infer] OUT{i}', o.shape, 'min', float(o.min()), 'max', float(o.max()), 'mean', float(o.mean()))

heads = [output_data[0], output_data[1], output_data[2]]
dets = decode_all(heads, score_thr=0.25, iou_thr=0.45)
print(f'[board_infer] DETECTIONS: {len(dets)}')
for d in dets:
    print('  ', json.dumps(d, default=lambda x: [float(v) for v in x] if hasattr(x, '__iter__') else float(x)))

# Save outputs for host-side comparison
np.savez(os.path.join(BASE, 'board_out.npz'),
         o0=output_data[0].astype(np.float32),
         o1=output_data[1].astype(np.float32),
         o2=output_data[2].astype(np.float32))
with open(os.path.join(BASE, 'board_dets.json'), 'w') as f:
    json.dump([{k: (list(map(float, v)) if isinstance(v, (list, tuple, np.ndarray)) else float(v)) for k, v in d.items()} for d in dets], f, indent=1)
print('[board_infer] saved board_out.npz + board_dets.json')
