#!/usr/bin/env python
# End-to-end sanity: PyTorch forward -> 3 raw heads -> NHWC (simulate DPU) ->
# yolo_decode.decode_all -> detections. Validates the full decode chain
# (DFL softmax + per-class NMS) before board bring-up.
import sys
sys.path.insert(0, '/workspace')
import cv2, glob, numpy as np, torch
from ultralytics import YOLO
from quantize_yolov8n_full import YoloV8DetectExport, letterbox
from yolo_decode import decode_all, draw_detections

WS = '/workspace'
img_path = sorted(glob.glob(f'{WS}/.calib_coco128/coco128/images/train2017/*.jpg'))[0]
print('test image:', img_path)

img = cv2.imread(img_path)
lb = letterbox(img)                                   # (640,640,3) BGR uint8
inp = lb[None].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
t = torch.from_numpy(inp)

full = YOLO(f'{WS}/yolov8n.pt').model
full.eval()
m = YoloV8DetectExport(full)
m.eval()
with torch.no_grad():
    heads = m(t)                                      # 3 x (1,144,H,W) NCHW

# simulate VART DPU output: NHWC (1,H,W,144)
heads_nhwc = [h.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
              for h in heads]
print('simulated DPU shapes:', [h.shape for h in heads_nhwc])

dets = decode_all(heads_nhwc, score_thr=0.25, iou_thr=0.45)
print(f'decoded {len(dets)} detections')
for d in sorted(dets, key=lambda d: -d["score"])[:10]:
    print(f'  {d["cls"]:3d} score={d["score"]:.3f} box={[int(v) for v in d["box"]]}')

# draw on the letterboxed canvas to eyeball
out = draw_detections(lb.copy(), dets)
cv2.imwrite(f'{WS}/decode_check_letterboxed.jpg', out)
print('saved decode_check_letterboxed.jpg')
