#!/usr/bin/env python3
"""PIL-based detection overlay (replaces the cv2 draw_detections).

Apple-inspired minimal overlay: thin 2px boxes with a soft rounded dark
label chip (white text). English COCO class names.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Apple system color accents (distinguishable on any frame)
PALETTE = [
    (94, 92, 230),    # indigo
    (255, 55, 95),    # pink
    (48, 209, 88),    # green
    (255, 159, 10),   # orange
    (191, 90, 242),   # purple
    (100, 210, 255),  # cyan
    (255, 214, 10),   # yellow
    (64, 200, 224),   # teal
]

_FONT_CACHE = {}


def _font(size):
    key = ('sans', size)
    if key not in _FONT_CACHE:
        candidates = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
        ]
        picked = next((p for p in candidates if os.path.exists(p)), None)
        try:
            _FONT_CACHE[key] = ImageFont.truetype(picked or candidates[0], size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def draw_detections(rgb, dets):
    """Draw minimal Apple-style boxes + English labels on an RGB frame."""
    im = Image.fromarray(rgb).convert('RGB')
    d = ImageDraw.Draw(im)
    f = _font(16)
    W, H = im.size
    for det in dets:
        cl = int(det['cls'])
        x1, y1, x2, y2 = (int(v) for v in det['box'])
        col = PALETTE[cl % len(PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        name = COCO_NAMES.get(cl, 'cls{}'.format(cl))
        label = '{} {:.2f}'.format(name, det['score'])
        tb = d.textbbox((0, 0), label, font=f)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        ly = max(0, y1 - th - 10)
        rx1 = max(0, min(x1, W - tw - 12))
        rx2 = min(W, rx1 + tw + 12)
        ry2 = min(H, ly + th + 8)
        try:
            d.rounded_rectangle([rx1, ly, rx2, ry2], radius=6, fill=(10, 12, 16))
        except Exception:
            d.rectangle([rx1, ly, rx2, ry2], fill=(10, 12, 16))
        d.text((rx1 + 6, ly + 3), label, fill=(245, 245, 247), font=f)
    return np.asarray(im)
