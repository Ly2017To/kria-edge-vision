#!/usr/bin/env python
# Resume route B full-model quantization: skip calib, reuse the quant_info.json
# saved by the interrupted run, and re-run only PHASE 2 deploy (export xmodel).
# NNDCT quant_mode='test' auto-loads output_dir/quant_info.json when present.
import sys
sys.path.insert(0, '/workspace')
from quantize_yolov8n_full import make_model, collect_images, run_deploy, OUT

import os, glob

imgs = collect_images()
print(f'calibration images: {len(imgs)}', flush=True)
assert imgs, 'no calibration images'

# sanity: calib artifacts must already exist
need = [f'{OUT}/quant_info.json', f'{OUT}/bias_corr.pth']
missing = [p for p in need if not os.path.exists(p)]
if missing:
    print('WARN missing calib artifacts (will attempt full re-run):', missing, flush=True)
else:
    print('reusing calib artifacts:', [os.path.basename(p) for p in need], flush=True)

run_deploy(make_model(), imgs)
