#!/usr/bin/env python
# Quantize FULL YOLOv8n (backbone + neck + Detect-head convs) with pytorch_nndct 3.5 (CPU).
# Route B for the cafe/bar general-detection demo:
#   * Detect-head convs (cv2/cv3 1x1) ARE kept on the DPU (pure conv, DPU-native)
#   * DFL decode + NMS are excluded from the model and run on board CPU (numpy)
#   * forward returns the 3 RAW head tensors (1,144,80,80)/(1,144,40,40)/(1,144,20,20)
#     -> all three are leaf nodes -> exported xmodel has exactly 3 outputs
#   * calibration set = COCO128 (covers all 80 COCO classes incl. the person / cup /
#     bottle / chair / table that the cafe/bar demo needs); auto-downloads,
#     falls back to legacy frames only if offline
#
# Usage (inside wod container):
#   python /workspace/quantize_yolov8n_full.py
# Output: /workspace/quantize_result_full/YoloV8DetectExport_int.xmodel  (3 outputs)
import os, glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_nndct import torch_quantizer

os.environ['YOLO_CONFIG_DIR'] = '/workspace/.yolo'
WS = '/workspace'
SIZE = 640
OUT = f'{WS}/quantize_result_full'
COCO_DIR = f'{WS}/.calib_coco128'

# DPU has no native SiLU -> convert to hard-swish at quantization time
# (official nndct option, same as the backbone script that already worked).
from nndct_shared.utils.option_list import NndctOption
NndctOption.nndct_convert_silu_to_hswish.value = True


class YoloV8DetectExport(nn.Module):
    """Full YOLOv8n without DFL/NMS.

    Runs backbone+neck+Detect-head convs using ultralytics' own layer-index
    logic (m.f), then the Detect head is patched to return the 3 raw
    (reg+cls) feature maps instead of running the DFL decode path.
    """

    def __init__(self, full_model):
        super().__init__()
        self.seq = full_model.model            # nn.Sequential incl. Detect (last)
        self.detect = full_model.model[-1]
        self.detect.forward = self._head_forward   # raw tensors only, no DFL

    def _head_forward(self, x):
        d = self.detect
        for i in range(d.nl):
            x[i] = torch.cat((d.cv2[i](x[i]), d.cv3[i](x[i])), 1)   # (1,144,H,W)
        return tuple(x)

    def forward(self, x):
        y = []
        for i, m in enumerate(self.seq):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return x   # tuple of 3 raw head tensors -> leaf nodes -> 3 xmodel outputs


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def ensure_coco128():
    """Return COCO128 train images (calibration set), downloading on first run."""
    for pat in (f'{COCO_DIR}/coco128/images/train2017/*.jpg',
                f'{COCO_DIR}/images/train2017/*.jpg'):
        imgs = sorted(glob.glob(pat))
        if imgs:
            return imgs
    print(f'COCO128 not present under {COCO_DIR}; downloading ...', flush=True)
    os.makedirs(COCO_DIR, exist_ok=True)
    zip_path = f'{COCO_DIR}/coco128.zip'
    try:
        import urllib.request, zipfile
        urllib.request.urlretrieve('https://ultralytics.com/assets/coco128.zip', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(COCO_DIR)
        os.remove(zip_path)
    except Exception as e:
        print(f'WARN COCO128 download failed: {e!r}', flush=True)
        return []
    for pat in (f'{COCO_DIR}/coco128/images/train2017/*.jpg',
                f'{COCO_DIR}/images/train2017/*.jpg'):
        imgs = sorted(glob.glob(pat))
        if imgs:
            return imgs
    return []


def collect_images():
    imgs = ensure_coco128()
    if not imgs:
        print('WARN: falling back to legacy calibration images '
              '(crossfit frames + bus.jpg) - NOT representative of general detection',
              flush=True)
        imgs = sorted(glob.glob(f'{WS}/output/*.jpg')) + [f'{WS}/bus.jpg']
        for v in sorted(glob.glob(f'{WS}/videos/*.mp4')):
            cap = cv2.VideoCapture(v)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if n > 0:
                for fi in range(0, n, max(1, n // 8)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, fr = cap.read()
                    if ok:
                        p = f'{WS}/.calib_tmp_{os.path.basename(v)}_{fi}.jpg'
                        cv2.imwrite(p, fr)
                        imgs.append(p)
            cap.release()
    return [p for p in imgs if os.path.exists(p)]


def loader(imgs, batch=1):
    for i in range(0, len(imgs), batch):
        arrs = []
        for p in imgs[i:i + batch]:
            img = cv2.imread(p)
            if img is None:
                continue
            arrs.append(letterbox(img))
        if not arrs:
            continue
        arr = np.stack(arrs).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        yield torch.from_numpy(arr)


def make_model():
    full = YOLO(f'{WS}/yolov8n.pt').model
    full.eval()
    m = YoloV8DetectExport(full)
    m.eval()
    print('head strides:', full.model[-1].strides.tolist(), flush=True)
    return m


def run_calib(model, imgs):
    print('--- PHASE 1: calib ---', flush=True)
    dummy = next(loader(imgs))
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=model,
        input_args=dummy,
        output_dir=OUT,
        device=torch.device('cpu'),
        bitwidth=8,
        target='DPUCZDX8G_ISA1_B4096',
        app_deploy='CV',
    )
    qmodel = quantizer.quant_model
    qmodel.eval()
    with torch.no_grad():
        for x in loader(imgs):
            _ = qmodel(x)
    quantizer.export_quant_config()
    print('calib done, quant config saved', flush=True)


def run_deploy(model, imgs):
    print('--- PHASE 2: deploy (export xmodel) ---', flush=True)
    dummy = next(loader(imgs))
    quantizer = torch_quantizer(
        quant_mode='test',
        module=model,
        input_args=dummy,
        output_dir=OUT,
        device=torch.device('cpu'),
        bitwidth=8,
        target='DPUCZDX8G_ISA1_B4096',
        app_deploy='CV',
    )
    qmodel = quantizer.quant_model
    qmodel.eval()
    with torch.no_grad():
        for x in loader(imgs):
            _ = qmodel(x)
    quantizer.export_xmodel(output_dir=OUT)
    print(f'DONE: quantized xmodel -> {OUT}', flush=True)
    verify_xmodel()


def verify_xmodel():
    """Load the exported xmodel back and assert it has exactly 3 outputs."""
    try:
        import xir
        xs = sorted(glob.glob(f'{OUT}/*_int.xmodel'))
        if not xs:
            print('XMODEL NOT FOUND under', OUT, flush=True)
            return
        g = xir.Graph.deserialize(xs[0])
        print(f'xmodel: {xs[0]}', flush=True)
        for o in g.get_head_ops():
            t = g.get_tensor(o.get_name())
            print(f'  INPUT  {o.get_name()} shape={t.dims}', flush=True)
        outs = sorted(g.get_tail_ops(), key=lambda o: o.get_name())
        print('  OUTPUT COUNT:', len(outs), flush=True)
        for o in outs:
            t = g.get_tensor(o.get_name())
            print(f'  OUTPUT {o.get_name()} shape={t.dims}', flush=True)
        if len(outs) == 3:
            print('ROUTE B OK: exactly 3 outputs', flush=True)
        else:
            print(f'ROUTE B FAIL: expected 3 outputs, got {len(outs)}', flush=True)
    except ImportError:
        print('xir not importable here; check output count on board later', flush=True)
    except Exception as e:
        print('xmodel verify failed:', repr(e), flush=True)


def main():
    imgs = collect_images()
    print(f'calibration images: {len(imgs)}', flush=True)
    if not imgs:
        raise SystemExit('no calibration images')
    os.makedirs(OUT, exist_ok=True)
    run_calib(make_model(), imgs)
    run_deploy(make_model(), imgs)


if __name__ == '__main__':
    main()
