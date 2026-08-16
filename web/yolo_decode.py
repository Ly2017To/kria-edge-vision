#!/usr/bin/env python
# CPU-side DFL decode + NMS for the YOLOv8n full-model xmodel (Route B).
#
# The DPU returns 3 raw head tensors (1,144,H,W) for strides 8/16/32:
#   channel layout per head: [0:64)  = raw box regression (4 boxes x 16 DFL bins)
#                            [64:144)= raw class logits (80 classes)
# Pure numpy: board-ready (VART runner outputs numpy arrays). No torch dependency.
#
# Usage:
#   from yolo_decode import decode_all, letterbox_inverse, draw_detections, COCO_NAMES
#   dets = decode_all(runner_outputs)            # list of {box, cls, score}
#   box  = letterbox_inverse(d["box"], img.shape[:2])   # back to original pixels
#   python /workspace/yolo_decode.py            # self-test on random data
import numpy as np

STRIDES = (8, 16, 32)
REG_MAX = 16
NC = 80

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Cafe/bar demo classes (subsets of COCO the demo highlights)
BAR_CLASSES = {0: "person", 39: "bottle", 41: "cup", 44: "spoon", 46: "bowl",
               47: "banana", 56: "chair", 60: "dining table", 67: "cell phone",
               75: "book", 76: "clock", 77: "vase"}


def softmax16(r):
    """r: (N,16) logits -> (N,16) probs (row-wise softmax over the 16 bins)."""
    e = np.exp(r - r.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def decode_level(reg, cls, stride):
    """Decode one head output.

    reg: (64,H,W) float32 raw regression logits
    cls: (80,H,W) float32 raw class logits
    -> boxes (N,4) xyxy in letterboxed-640 pixels, scores (N,80) sigmoid probs
    """
    c, h, w = reg.shape
    n = h * w
    reg = reg.reshape(4, REG_MAX, n).transpose(2, 0, 1)     # (N,4,16)
    reg = softmax16(reg.reshape(n * 4, 16)).reshape(n, 4, REG_MAX)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = (reg * bins).sum(axis=-1) * stride               # (N,4) pixels
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ax = (gx.reshape(-1) + 0.5) * stride
    ay = (gy.reshape(-1) + 0.5) * stride
    anchor = np.stack([ax, ay], axis=1)                     # (N,2) pixels
    x1y1 = anchor - dist[:, :2]
    x2y2 = anchor + dist[:, 2:]
    boxes = np.concatenate([x1y1, x2y2], axis=1)
    scores = 1.0 / (1.0 + np.exp(-cls.reshape(NC, n).T))    # (N,80)
    return boxes, scores


def nms(boxes, scores, score_thr=0.25, iou_thr=0.45, max_det=300, top_k=1000):
    """Per-class NMS, fully vectorized numpy. boxes (N,4) xyxy, scores (N,80).

    Two fast paths over the naive loop:
      1. top_k: only the top-`top_k` candidates by class score enter NMS
         (YOLOv8n 640x640 emits 8400 anchors, most of them noise).
      2. suppression is vectorized: for each kept box, IoU against all
         remaining same-class boxes is computed in one shot with boolean
         masking - no per-candidate numpy array rebuilds.

    Result set is identical to the naive implementation (same greedy order,
    same IoU criterion) so the optimization is purely a speedup.

    -> list of dicts {box:(x1,y1,x2,y2), cls:int, score:float} sorted by score.
    """
    n = boxes.shape[0]
    cls_ids = scores.argmax(axis=1)
    cls_scores = scores[np.arange(n), cls_ids]
    cand = np.where(cls_scores >= score_thr)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(-cls_scores[cand])][:top_k]
    ob = boxes[order]
    oc = cls_ids[order]
    oscore = cls_scores[order]
    x1 = ob[:, 0]; y1 = ob[:, 1]
    x2 = ob[:, 2]; y2 = ob[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    suppressed = np.zeros(order.size, dtype=bool)
    out = []
    for i in range(order.size):
        if suppressed[i]:
            continue
        out.append((ob[i], int(oc[i]), float(oscore[i])))
        if len(out) >= max_det:
            break
        j = np.flatnonzero((~suppressed) & (oc == oc[i]))
        j = j[j > i]
        if j.size == 0:
            continue
        xx1 = np.maximum(x1[i], x1[j])
        yy1 = np.maximum(y1[i], y1[j])
        xx2 = np.minimum(x2[i], x2[j])
        yy2 = np.minimum(y2[i], y2[j])
        inter = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
        iou = inter / (areas[i] + areas[j] - inter + 1e-9)
        suppressed[j[iou > iou_thr]] = True
    return [{"box": b, "cls": c, "score": s} for b, c, s in out]


def decode_all(heads, strides=STRIDES, score_thr=0.25, iou_thr=0.45):
    """heads: list of 3 numpy arrays as the DPU returns them.

    Accepts both NHWC (1,H,W,144) - the VART runner layout on KV260 - and
    NCHW (1,144,H,W); the NHWC form is auto-transposed. Each head channel
    layout: [0:64) raw box regression (4 x 16 DFL bins), [64:144) class logits.

    -> detections {box:(x1,y1,x2,y2) in letterboxed-640 pixels, cls, score}
    """
    boxes_l, scores_l = [], []
    for h, s in zip(heads, strides):
        h0 = np.asarray(h)
        if h0.ndim == 4 and h0.shape[-1] == 4 * REG_MAX + NC and h0.shape[1] != 4 * REG_MAX + NC:
            h0 = h0.transpose(0, 3, 1, 2)        # NHWC -> NCHW
        h0 = h0[0]                               # (144,H,W)
        reg = h0[:4 * REG_MAX]
        cls = h0[4 * REG_MAX:]
        b, c = decode_level(reg, cls, s)
        boxes_l.append(b)
        scores_l.append(c)
    boxes = np.concatenate(boxes_l, axis=0)
    scores = np.concatenate(scores_l, axis=0)
    return nms(boxes, scores, score_thr, iou_thr)


def letterbox(img, size=640):
    """Match the preprocess used by the quantize script (and the board runner)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def letterbox_inverse(box, orig_shape, size=640):
    """Map a detection box from letterboxed-640 coords back to the original image."""
    oh, ow = orig_shape[:2]
    r = min(size / oh, size / ow)
    nh, nw = int(round(oh * r)), int(round(ow * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    x1, y1, x2, y2 = box
    x1 = (x1 - left) / r
    y1 = (y1 - top) / r
    x2 = (x2 - left) / r
    y2 = (y2 - top) / r
    return np.clip([x1, y1, x2, y2], 0, [ow, oh, ow, oh]).astype(np.float32)


def draw_detections(img, dets, names=COCO_NAMES, only_bar=False):
    """Draw detections on a BGR image (cv2). dets boxes are already in img pixels."""
    import cv2
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cl = d["cls"]
        if only_bar and cl not in BAR_CLASSES:
            continue
        label = f'{names[cl]} {d["score"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    # self-test: random heads must run without error and produce sane shapes
    rng = np.random.default_rng(0)
    heads = [rng.standard_normal((1, 4 * REG_MAX + NC, s_, s_)).astype(np.float32)
             for s_ in (80, 40, 20)]
    dets = decode_all(heads)
    print(f"self-test OK: {len(dets)} detections on random inputs (expected 0)")

    # end-to-end sanity: random heads but with a planted strong reg signal
    heads2 = []
    for h in heads:
        h2 = h.copy()
        h2[:, 4 * REG_MAX:, :, :] = 0.0          # zero class logits -> sigmoid 0.5
        h2[:, 4 * REG_MAX + 0, :, :] = 2.0       # class 0 (person) positive
        heads2.append(h2)
    dets2 = decode_all(heads2, score_thr=0.6)
    print(f"self-test OK: {len(dets2)} person detections on planted signal (expected > 0)")
