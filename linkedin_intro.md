# LinkedIn Intro — Kria Edge Vision

## Short version (LinkedIn "Projects" section)

**Kria Edge Vision — Real-time YOLOv8 object detection on AMD Kria KV260's on-board DPU**

Deployed a fully quantized YOLOv8n (COCO, 80 classes) end-to-end on the KV260's dedicated AI processor (DPU B4096), served through a live web dashboard with MJPEG streaming and per-box class/confidence overlays.

What made this non-trivial: the entire backbone + neck + Detect-head convolutions run on the DPU. I patched the Detect head to return its 3 raw `(1,144,80/40/20)` tensors, making them leaf nodes so post-training quantization (pytorch_nndct 3.5, COCO-128 calibration, SiLU → hard-swish) exports a clean 3-output xmodel — sidestepping a known NNDCT multi-output serialization bug. The parts the DPU cannot execute (DFL softmax-16 decode + per-class NMS + letterbox inverse) I implemented from scratch in pure numpy on the ARM CPU.

Measured on hardware: ~5.8–5.9 FPS (3-stage pipelined, +37% vs single-thread), DPU inference ~50 ms at 640×640 INT8, SoC ~42 °C under load. Deployed as systemd services that auto-load the DPU bitstream at boot so the pipeline survives power cycles. Stage 2 (planned): moving the CPU decode path into Vitis HLS accelerators on the programmable logic.

Repo: https://github.com/<your-user>/kria-edge-vision

## Longer version (LinkedIn post)

**Deploying YOLOv8n on an AMD Kria KV260 — DPU, quantization, and a numpy decode path.**

I wanted to see a modern object detector run on embedded programmable silicon — not on a GPU. So I took the AMD Kria KV260 (Zynq UltraScale+ MPSoC with a dedicated DPU B4096), and put YOLOv8n on it: a fully quantized model running backbone, neck, and Detect-head convolutions on the DPU, with a live web dashboard on top.

The interesting engineering was in the details:

- **3-output quantization trick.** The ultralytics Detect head applies DFL and NMS in its forward pass, which can't run on the DPU. I patched `Detect.forward` to return only the 3 raw head tensors `(1,144,80/40/20)`, making them clean leaf nodes. Result: post-training quantization (pytorch_nndct 3.5, COCO-128 calibration, SiLU → hard-swish) exports an xmodel with exactly 3 outputs, cleanly bypassing a known NNDCT multi-output serialization bug.
- **Pure-numpy decode on the ARM CPU.** The DPU can't do DFL softmax-16, per-class NMS, or letterbox inverse — so I wrote them from scratch in numpy (no OpenCV), with letterbox-inverse mapping back to the camera frame.
- **Pipeline + real numbers.** A 3-stage capture → detect → draw/encode pipeline took throughput from 4.3 to ~5.8–5.9 FPS; DPU inference is ~50 ms/frame at 640×640 INT8; the SoC stays at ~42 °C under continuous load. A live dashboard streams FPS, DPU latency, per-stage timings, and every detection as JSON.
- **Survives power cycles.** Two systemd units: one loads the DPU B4096 bitstream at boot (the FPGA resets to the starter shell otherwise), the other starts the vision service only once the DPU is present.

Next: moving the numpy decode stages (letterbox, DFL decode, NMS) into custom Vitis HLS accelerators on the programmable logic, to get the ARM CPU out of the per-frame critical path.

Full build log and deployment notes in the repo: https://github.com/<your-user>/kria-edge-vision

#EdgeAI #FPGA #YOLOv8 #AMD #KV260 #Kria #ComputerVision #EmbeddedSystems #VitisAI #DeepLearning
