#!/usr/bin/env python3
"""DPU detector wrapper for the KV260 YOLOv8n full-model xmodel (Route B).

Encapsulates: xir deserialize -> DPU subgraph -> VART runner -> 3 head
tensors -> yolo_decode (DFL softmax16 + NMS) -> letterbox-inverse boxes.

Requires: LD_PRELOAD=/usr/local/lib/libxcl_stub.so (XRT 2.13 workaround).
"""
import os
import time

import numpy as np
import xir
import vart
from PIL import Image

from yolo_decode import decode_all, letterbox_inverse


class DpuDetector:
    def __init__(self, xmodel, score_thr=0.25, iou_thr=0.45):
        self.xmodel = xmodel
        self.score_thr = score_thr
        self.iou_thr = iou_thr

        g = xir.Graph.deserialize(xmodel)
        # CRITICAL: keep the graph object alive. The VART runner holds
        # references into this C++ graph; if `g` is garbage-collected the
        # next execute_async() segfaults (use-after-free) in
        # DpuKernel::get_fingerprint(). board_infer.py works because its
        # graph is a module-level global that never gets collected.
        self._graph = g
        root = g.get_root_subgraph()
        # Locate the compiled DPU subgraph. Some builds make the root itself
        # the DPU graph; fingerprint-only builds (Route B v3) place the DPU
        # subgraph among root children (a sibling CPU subgraph may also carry
        # a 'runner' attr pointing at libvart-cpu-runner.so - avoid it).
        sub = None
        if root.has_attr('device') and root.get_attr('device') == 'DPU':
            sub = root
        else:
            for k in root.get_children():
                if k.has_attr('device') and k.get_attr('device') == 'DPU':
                    sub = k
                    break
        if sub is None:
            raise RuntimeError('no DPU subgraph found in xmodel')
        self.runner = vart.Runner.create_runner(sub, 'run')
        self.in_t = self.runner.get_input_tensors()
        self.out_t = self.runner.get_output_tensors()
        print('[infer] input :', [(t.name, t.dims) for t in self.in_t])
        print('[infer] output:', [(t.name, t.dims) for t in self.out_t])
        self.warmup()

    def warmup(self, n=2):
        """First calls are slower (cache/clock ramp); prime the runner."""
        shape = self.in_t[0].dims
        for _ in range(n):
            z = np.zeros(shape, dtype=np.float32)
            out = [np.empty(t.dims, dtype=np.float32) for t in self.out_t]
            job = self.runner.execute_async([z], out)
            self.runner.wait(job)

    def letterbox(self, rgb, size=640):
        """Match quantize-script preprocessing, PIL-based (no cv2)."""
        h, w = rgb.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        if (nh, nw) != (h, w):
            im = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
            resized = np.asarray(im)
        else:
            resized = rgb  # already target size - avoid copy
        canvas = np.full((size, size, 3), 114, np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas

    def detect(self, rgb):
        """Detect on a full RGB frame.

        Returns (dets, times):
          dets  = list of {'box': (x1,y1,x2,y2) in ORIGINAL frame pixels,
                           'cls': int, 'score': float}
          times = dict of per-stage wall-clock ms:
                  lb_ms   letterbox + float normalize (CPU)
                  dpu_ms  pure DPU execute_async + wait
                  dec_ms  DFL softmax16 decode + NMS + letterbox-inverse (CPU)
        """
        t0 = time.time()
        lb = self.letterbox(rgb)
        inp = (lb.astype(np.float32) / 255.0)[None, ...]  # [1,640,640,3]
        t1 = time.time()
        out = [np.empty(t.dims, dtype=np.float32) for t in self.out_t]
        job = self.runner.execute_async([inp], out)
        self.runner.wait(job)
        t2 = time.time()
        dets = decode_all(out, score_thr=self.score_thr, iou_thr=self.iou_thr)
        for d in dets:
            box = letterbox_inverse(
                np.asarray(d['box'], dtype=np.float32), rgb.shape[:2])
            d['box'] = [float(v) for v in box]
        t3 = time.time()
        return dets, {
            'lb_ms': (t1 - t0) * 1000.0,
            'dpu_ms': (t2 - t1) * 1000.0,
            'dec_ms': (t3 - t2) * 1000.0,
        }
