#!/usr/bin/env python3
"""KV260 YOLOv8n detection web service.

Endpoints:
  GET /             real-time detection web page (video + detections)
  GET /video_feed   MJPEG stream of annotated frames
  GET /api/status   JSON: fps, dpu_ms, source, detections, ...
  POST /api/source  {"mode": "auto"|"cam"|"demo"} switch input source

Run:
  LD_PRELOAD=/usr/local/lib/libxcl_stub.so python3 app.py [--port 8080]
"""
import os
import sys
import time
import json
import argparse
import queue
import threading
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from camera import CameraManager          # noqa: E402
from infer import DpuDetector             # noqa: E402
from draw import draw_detections          # noqa: E402

PARENT = os.path.dirname(BASE)
XM = os.path.join(PARENT, 'YoloV8DetectExport.xmodel')
DEMO_DIR = os.path.join(BASE, 'demo_imgs')

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='0.0.0.0')
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--xmodel', default=XM)
ap.add_argument('--jpg-quality', type=int, default=72)
args = ap.parse_args()

print('[app] loading detector:', args.xmodel)
detector = DpuDetector(args.xmodel)
cam = CameraManager(DEMO_DIR)

state = {
    'lock': threading.Lock(),
    'jpeg': None,
    'dets': [],
    'fps': 0.0,
    'total_ms': 0.0,   # per-frame WORK / end-to-end latency = sum of EMA parts
    'clock_ms': 0.0,   # wall-clock frame PERIOD = 1000/fps (pipeline throughput)
    'read_ms': 0.0,    # frame read / wait (CPU)
    'lb_ms': 0.0,      # letterbox + normalize (CPU)
    'dpu_ms': 0.0,     # pure DPU inference
    'dec_ms': 0.0,     # DFL softmax + NMS + letterbox-inverse (CPU)
    'draw_ms': 0.0,    # PIL box drawing (CPU)
    'enc_ms': 0.0,     # JPEG encode (CPU)
    'source': 'starting',
    'det_count': 0,
    'frame': 0,
    'ts': 0.0,
}


def _jpeg_encode(rgb, quality=args.jpg_quality):
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def _ema(old, cur):
    return cur if old == 0.0 else 0.9 * old + 0.1 * cur


def worker():
    """3-stage pipeline: capture thread -> detect thread -> draw/encode (this).

    Single-thread latency is the SUM of stages (read+lb+DPU+dec+draw+enc
    ~= 230 ms). With a pipeline, throughput becomes the SLOWEST stage, so
    the camera wait (read) and CPU decode overlap with the DPU. total_ms is
    kept as the sum of EMA parts - it now measures per-frame WORK rather than
    wall-clock latency (which in the pipeline is ~max(stage) instead of
    ~sum(stage)).

    Threading rules: capture_loop owns cam.read(); detect_loop owns
    detector.detect() (the DPU VART runner is NOT thread-safe - exactly one
    caller). Both queues are bounded (maxsize=2) so a slow downstream stage
    naturally backpressures instead of buffering stale frames, which would
    inflate latency.
    """
    fps_ema = 0.0
    prev_done = None
    q_frames = queue.Queue(maxsize=2)   # (frame, read_ms)   capture -> detect
    q_res = queue.Queue(maxsize=2)      # (frame, dets, times) detect -> draw

    def capture_loop():
        while True:
            t0 = time.time()
            frame = cam.read()
            t1 = time.time()
            if frame is None:
                time.sleep(0.05)
                continue
            q_frames.put((frame, (t1 - t0) * 1000.0))

    def detect_loop():
        while True:
            frame, read_ms = q_frames.get()
            dets, times = detector.detect(frame)
            times['read_ms'] = read_ms
            q_res.put((frame, dets, times))

    for fn in (capture_loop, detect_loop):
        threading.Thread(target=fn, daemon=True).start()

    while True:
        frame, dets, times = q_res.get()
        t0 = time.time()
        rgb = draw_detections(frame, dets)
        t1 = time.time()
        jpg = _jpeg_encode(rgb)
        t2 = time.time()
        # fps = pipeline throughput = interval between completed frames
        if prev_done is not None:
            fps_ema = 1.0 / (t2 - prev_done) if fps_ema == 0.0 \
                else 0.9 * fps_ema + 0.1 / (t2 - prev_done)
        prev_done = t2
        read_ema = _ema(state['read_ms'], times['read_ms'])
        lb_ema = _ema(state['lb_ms'], times['lb_ms'])
        dpu_ema = _ema(state['dpu_ms'], times['dpu_ms'])
        dec_ema = _ema(state['dec_ms'], times['dec_ms'])
        draw_ema = _ema(state['draw_ms'], (t1 - t0) * 1000.0)
        enc_ema = _ema(state['enc_ms'], (t2 - t1) * 1000.0)
        total = read_ema + lb_ema + dpu_ema + dec_ema + draw_ema + enc_ema
        clock = 1000.0 / fps_ema if fps_ema > 0.0 else 0.0
        with state['lock']:
            state['jpeg'] = jpg
            state['dets'] = dets
            state['fps'] = fps_ema
            state['total_ms'] = total
            state['clock_ms'] = clock
            state['read_ms'] = read_ema
            state['lb_ms'] = lb_ema
            state['dpu_ms'] = dpu_ema
            state['dec_ms'] = dec_ema
            state['draw_ms'] = draw_ema
            state['enc_ms'] = enc_ema
            state['source'] = cam.mode_src
            state['det_count'] = len(dets)
            state['frame'] += 1
            state['ts'] = time.time()


app = Flask(__name__,
            template_folder=os.path.join(BASE, 'templates'),
            static_folder=os.path.join(BASE, 'static'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen():
        boundary = b'frame'
        last = -1
        while True:
            with state['lock']:
                jpg = state['jpeg']
                frm = state['frame']
            if jpg is None or frm == last:
                time.sleep(0.02)
                continue
            last = frm
            yield (b'--' + boundary + b'\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(0.02)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with state['lock']:
        dets = list(state['dets'])
        payload = {
            'fps': round(state['fps'], 1),
            'total_ms': round(state['total_ms'], 1),
            'clock_ms': round(state['clock_ms'], 1),
            'read_ms': round(state['read_ms'], 1),
            'lb_ms': round(state['lb_ms'], 1),
            'dpu_ms': round(state['dpu_ms'], 1),
            'dec_ms': round(state['dec_ms'], 1),
            'draw_ms': round(state['draw_ms'], 1),
            'enc_ms': round(state['enc_ms'], 1),
            'cpu_ms': round(state['lb_ms'] + state['dec_ms'], 1),
            'other_ms': round(state['read_ms'] + state['draw_ms'] + state['enc_ms'], 1),
            'source': state['source'],
            'frame': state['frame'],
            'det_count': state['det_count'],
            'detections': [
                {'cls': int(d['cls']),
                 'box': [round(v, 1) for v in d['box']],
                 'score': round(float(d['score']), 3)}
                for d in dets
            ],
            'ts': state['ts'],
        }
    return jsonify(payload)


@app.route('/api/source', methods=['POST'])
def api_source():
    mode = request.json.get('mode', 'auto') if request.is_json else 'auto'
    cam.set_mode(mode)
    return jsonify({'mode': mode, 'source': cam.mode_src})


if __name__ == '__main__':
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    print('[app] serving http://{}:{}'.format(args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True)
