# KV260 VART 3.5.0 Deployment Log

Date: 2026-08-15 (morning session)
Goal: Deploy Vitis AI Runtime 3.5.0 (VART + XIR) on the KV260, to enable the board-side runtime for the quantized/compiled YOLOv8 (DPUCZDXG8_ISA1_B4096).

## System Status

- Ubuntu 22.04 (Kria variant), kernel 5.15.0-1027-xilinx-zynqmp
- XRT 2.13.479-0ubuntu2 (Ubuntu package, corresponds to XRT 2020.2) — does not match the XRT 2023.1 that VART 3.5.0 officially expects; not upgraded yet (proceeding with "make it run" first)
- dfx-mgr 2023.1, xmutil, fpga-manager all installed
- Disk space: /dev/mmcblk1p2 29G total / 21G free

## Step 1: Download and unpack the VART runtime

```bash
cd ~
wget "https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-3.5.0.tar.gz" -O vitis-ai-runtime-3.5.0.tar.gz
tar xzf vitis-ai-runtime-3.5.0.tar.gz
ls ~/vitis-ai-runtime-3.5.0/2023.1/aarch64/centos/
```

Contents: 5 RPMs (**no XRT RPM**) — libunilog, libxir, libtarget-factory (DPU runner), libvart, libvitis_ai_library + setup.sh.
Note: setup.sh is just `rpm -ivh --force` and depends on the rpm package manager, so it does not work on Ubuntu → manual unpacking instead.

## Step 2: Unpack the RPMs and install into the system

```bash
mkdir -p ~/vart_extract && cd ~/vart_extract
for f in ~/vitis-ai-runtime-3.5.0/2023.1/aarch64/centos/*.rpm; do
  rpm2cpio "$f" | cpio -idmv
done

sudo cp -a usr/lib/. /usr/lib/
sudo cp -a usr/bin/. /usr/bin/
sudo mkdir -p /usr/lib/python3/dist-packages
sudo cp -a usr/lib/python3.10/site-packages/. /usr/lib/python3/dist-packages/
sudo ldconfig
```

Verify: `which xdputil` → `/usr/bin/xdputil` OK (RPM copy succeeded)

## Step 3: Missing Boost 1.80 library (first error)

```text
python3 -c "import vart, xir" → ImportError: libboost_filesystem.so.1.80.0: cannot open shared object file
```

Cause: VART 3.5.0 was built for PetaLinux 2023.1 (Boost 1.80), but Ubuntu 22.04 ships Boost 1.74, so the soname does not match. `ldd` confirmed this is the only missing library on the whole system.

## Step 4: Symlink approach fails (key lesson)

```bash
sudo ln -s /usr/lib/aarch64-linux-gnu/libboost_filesystem.so.1.74.0 /usr/lib/aarch64-linux-gnu/libboost_filesystem.so.1.80.0
sudo ldconfig
```

New error:

```text
/lib/libunilog.so.3: undefined symbol: _ZN5boost10filesystem4path9append_v3ERKS1_
```

Conclusion: a symlink only fixes the **soname name-lookup layer** (lets ld.so find a file named .so.1.80.0); it cannot fix the **symbol ABI layer** (`append_v3` is a v3 path API added in Boost 1.80 and is absent from the 1.74 symbol table). A real Boost 1.80 is required.

Symbol decode: `_ZN5boost10filesystem4path9append_v3ERKS1_` = `boost::filesystem::path::append_v3(const path&)` (Itanium ABI name mangling).

## Step 5: Handle the apt lock (stuck unattended-upgrades)

Symptom: `apt install` stuck on `Waiting for cache lock: held by process 7866 (unattended-upgr)`, not finishing after 39 minutes (--download-only mode stuck on network download).

Resolution:

```bash
sudo systemctl stop unattended-upgrades
sudo fuser -v /var/lib/dpkg/lock-frontend   # no output = lock released
sudo apt install -y build-essential
```

Permanently disable (recommended on dev machines):

```bash
sudo systemctl disable --now unattended-upgrades
```

## Step 6: Download Boost 1.80 source (mirror change after pitfall)

Pitfall: `https://boostorg.jfrog.io/...` link is dead — returns 302 to `landing.jfrog.com/reactivate-server/boostorg` (an 11KB HTML page), so `tar xjf` fails with "not a bzip2 file".

Correct source:

```bash
wget https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.bz2 -O boost_1_80_0.tar.bz2
# China mirrors (fallback):
# wget https://mirrors.tuna.tsinghua.edu.cn/boost/1.80.0/boost_1_80_0.tar.bz2
# wget https://mirrors.ustc.edu.cn/boost/1.80.0/boost_1_80_0.tar.bz2

file boost_1_80_0.tar.bz2   # must show "bzip2 compressed data" before extracting
tar xjf boost_1_80_0.tar.bz2
```

## Step 7: Build Boost 1.80 (filesystem + system only)

```bash
cd ~/boost_1_80_0
./bootstrap.sh --with-libraries=filesystem,system
sudo ./b2 install -j4 --with-filesystem --with-system --prefix=/usr/local
```

- Installed to: `/usr/local/lib` (libraries), `/usr/local/include` (headers) — does not pollute the package-managed `/usr`
- `/usr/local/lib` is in `/etc/ld.so.conf.d/libc.conf`, so ldconfig scans it
- Remove the fake symlink to avoid 1.74 being loaded first:

```bash
sudo rm -f /usr/lib/aarch64-linux-gnu/libboost_filesystem.so.1.80.0
sudo ldconfig
```

## Step 8: Verification OK

```text
ldconfig -p | grep libboost_filesystem
    libboost_filesystem.so.1.80.0 => /usr/local/lib/libboost_filesystem.so.1.80.0   <- real 1.80
    libboost_filesystem.so.1.74.0 => /lib/aarch64-linux-gnu/...                     <- system 1.74 coexists

python3 -c "import vart, xir; print('VART OK:', vart.__file__)"
    VART OK: /usr/lib/python3/dist-packages/vart.so   OK
```

VART 3.5.0 deployment complete.

## Next Steps (then pending)

- [ ] DPU firmware verification: LogicTronix B4096 firmware → `/lib/firmware/xilinx/` → `xmutil loadapp` → `xdputil query`
- [ ] If `xdputil query` reports an XRT version/parse error → upgrade XRT to 2023.1 (2.15.225)
- [ ] Quantize + compile YOLOv8n (pytorch_nndct) → compile DPUCZDXG8_ISA1_B4096 → on-board inference

## Key Technical Notes (quick reference)

1. **RPM → Ubuntu port**: `rpm2cpio <x.rpm> | cpio -idmv` unpacks and copies manually to system dirs; rpm is only a container.
2. **soname vs symbol ABI**: symlink/ldconfig only solves "library filename lookup"; a symbol-table mismatch (e.g. append_v3) requires building the matching library version.
3. **C++ name mangling**: the mangled symbol exactly encodes "namespace + class + method + argument types" — it is the identity card for binary compatibility.
4. **/usr/local convention**: manually compiled installs go to /usr/local, isolated from package-managed /usr.
5. **Dynamic-link load order**: ld.so searches by the soname in DT_NEEDED → cache /etc/ld.so.cache → default directories.

---

# Archived: Abandoned attempts before the Route B decision (2026-08-15)

Background: the cafe/bar general object detection (COCO 80 classes) was decided to follow Route B (full-model quantization: backbone+neck+Detect-head convolutions on the DPU, DFL decode + NMS on the board CPU in numpy). All of the following artifacts are unrelated to Route B and were moved into `workspace/backup_pre_routeB/` (files generated inside the container are root-owned; moving them requires `mv` + `chown -R 1000:1000` inside the wod container, or `sudo` on the host).

1. **Old full-model export `quantize_result/` (DetectionModel.py + DetectionModel_int.xmodel) — only 1 output + CPU subgraph, abandoned**
   - The auto-generated `DetectionModel.forward` returned 6 tensors at the end (decoded `(1,84,8400)` + 3 raw head tensors + 2 concat intermediates), of which 5 were consumed by the subsequent DFL decode ops, leaving only the decoded tensor as a leaf node.
   - xir evidence (2026-08-15 09:00, `g.get_tail_ops()`): only 1 output op, `DetectionModel__DetectionModel_Detect_model__Detect_22__ret`; the head_ops were all DFL reshape/transpose/slice/shape_attr/sink_transpose ops, and loading also warned `aten::silu_ undefined` → the graph contains aten CPU ops.
   - Conclusion: single output + a large CPU subgraph means poor performance and cannot showcase DPU acceleration.

2. **Backbone-only quantization `quantize_yolov8n.py` + `quantize_result_backbone/` + `compiled_backbone/` — performance insufficient, abandoned**
   - Only backbone+neck went to the DPU; the Detect head's 3 1x1 convs (~200M MACs) stayed on the board CPU, heavy per-frame CPU load, 20+ FPS not achievable.
   - The compile arch of this route (arch = DPUCZDXG8_ISA1_B4096) was reused by Route B.

3. **NNDCT 3.5 multi-output serialization experiments `multiout_test*/` + `test_multiout*.py` + `probe_devgraph*.py` + `patch_probe.py/patch_revert.py` — all failed**
   - The goal was to let the xmodel output multiple tensors (framework patches to make DFL tensors leaves, cat-concat multi-output, etc.), but NNDCT 3.5's multi-output serialization has a bug; all patches/probes failed.
   - Route B sidesteps this bug: monkeypatching `Detect.forward` to return only the 3 raw `(1,144,H,W)` head tensors → 3 leaf nodes → the xmodel has exactly 3 outputs, never triggering the multi-output serialization path.

4. **CrossFit/pose-specific `analyze_motion.py`, `motion_engine.py`, `test_pose*.py`, `yolov8n-pose.pt` — unrelated to cafe/bar**
   - The cafe/bar demo app is general object detection (COCO 80 classes), unrelated to CrossFit counting; no pose model or motion analysis needed.

5. **Old assets `output/`, `output_video/`, `videos/`, `bus.jpg` — calibration set changed**
   - Originally the fallback for the old calibration set; Route B switched the calibration set to COCO128 (auto-downloaded, covering person/cup/bottle/chair/table and other cafe/bar-scene classes).
   - If COCO128 download fails and a fallback is needed, copy back from `backup_pre_routeB/legacy_media/` (script fallback paths are `/workspace/output/*.jpg`, `/workspace/bus.jpg`, `/workspace/videos/*.mp4`).

**Kept (Route B related, not moved)**: `quantize_yolov8n_full.py`, `yolo_decode.py`, `yolov8n.pt`, `start_dev.sh`, `xcl_stub.c` (the board-side XRT 2.13 workaround for VART 3.5.0; still needed for deployment).

**Script fix note**: `quantize_yolov8n_full.py` `verify_xmodel()` originally called `g.get_input_ops()/get_output_ops()`, but the container's xir does not have these two methods (verified method list: get_head_ops/get_tail_ops/get_tensors/get_op etc.); changed to `get_head_ops()/get_tail_ops()`.

---

# Stage 2: Route B Quantization → Compile → Board Deployment (2026-08-15/16 sessions)

## Environment (board, verified)

- Login: `ssh ubuntu@10.0.0.20` (password `kv260dev`, headless / no display)
- Ubuntu 22.04.4 (Kria), kernel 5.15.0-1027-xilinx-zynqmp, 21G disk free; sudo password is also `kv260dev`
- **VART 3.5.0 / XRT 2.13 / xdputil pre-installed** (Stage-1 deployment done): `import vart, xir, target_factory` passes, python3.10 + numpy 1.21.5, gcc 11.4.0 can build aarch64
- **DPU overlay active**: `xmutil loadapp benchmark-b4096` (slot 0)
- `xdputil query` (with LD_PRELOAD stub) confirms: **DPU Arch = `DPUCZDX8G_ISA1_B4096_0101000016010407`**, 300 MHz, 1 core, cu 0x80010000
- **Hardware fingerprint = `0x101000016010407`** (short form 0x16) — this is the key baseline for deployment verification

## Step 9: Route B full-model quantization (host, wod container)

Goal: backbone + neck + Detect-head convolutions (cv2/cv3, pure conv) all go to the DPU; DFL decode + NMS stays on the board's CPU (numpy).

Approach: new wrapper `YoloV8DetectExport` (`quantize_yolov8n_full.py`) monkeypatches `Detect.forward` to return only the 3 raw head tensors `(1,144,80/40/20)` without running DFL → 3 leaf nodes → the xmodel has exactly 3 outputs, sidestepping the NNDCT 3.5 multi-output serialization bug.

```bash
# Inside the container (vitis-ai-pytorch environment)
conda activate vitis-ai-pytorch
cd /workspace
python quantize_yolov8n_full.py --resume   # or run directly (COCO128 calibration set auto-downloaded)
```

- Calibration set: COCO128 (covers person/cup/bottle/chair/table and other cafe/bar-scene classes)
- Artifact: `/workspace/quantize_result_full/YoloV8DetectExport_int.xmodel`
- Verification (xir): `OUTPUT COUNT: 3`, `[1,80,80,144]` / `[1,40,40,144]` / `[1,20,20,144]` → `ROUTE B OK`

## Step 10: Compile DPUCZDXG8_ISA1_B4096 (container)

```bash
vai_c_xir -x /workspace/quantize_result_full/YoloV8DetectExport_int.xmodel \
          -a /workspace/arch_kv260_fp16.json \
          -o /workspace/compiled_full_v2 \
          -n deploy
```

- v1 artifact: `compiled_full/YoloV8DetectExport.xmodel` (4479143 B)
- v2 artifact: `compiled_full_v2/deploy.xmodel` (4485383 B, `arch_kv260_fp16.json` has both `target` and `fingerprint` fields)
- **Fingerprint evidence (late 2026-08-15)**: both artifacts' DPU subgraph `dpu_fingerprint` = `72339070457545735` = `0x101000056010407` (0x56)
  - Conclusion: `target` and `fingerprint` in `arch.json` are mutually exclusive if/elif (`vai_c_xir.py` lines 85-115); when `target` exists, `fingerprint` is ignored → the built-in B4096 table still produces 0x56

## Step 11: Board DPU overlay + hardware fingerprint check (early 2026-08-16)

```bash
sudo xmutil loadapp benchmark-b4096        # load DPU overlay (slot 0)
LD_PRELOAD=/usr/local/lib/libxcl_stub.so xdputil query
# → DPU Arch = DPUCZDX8G_ISA1_B4096_0101000016010407 (0x16), 300 MHz
```

**Core conflict**: hardware fingerprint 0x16 vs model fingerprint 0x56 — **mismatch** (the only blocker at this point).

## Step 12: xcl_stub.c (XRT 2.13 missing-symbol workaround)

Cause: XRT 2.13 is missing the `xclIPSetReadRange` / `xclIPReadRange` / `xclIPWriteRange` symbols (BIND_NOW immediate binding → dlopen fails). The patch `workspace/xcl_stub.c` provides these symbols.

```bash
# Build and install on the board
gcc -shared -fPIC -o libxcl_stub.so xcl_stub.c -ldl
sudo cp libxcl_stub.so /usr/local/lib/
```

Usage: prefix every DPU program with `LD_PRELOAD=/usr/local/lib/libxcl_stub.so` (exported inside the inference scripts, not written to /etc/ld.so.preload). Verified dpu-runner / device-handle dlopen success and normal `xdputil query` output.

## Step 13: Upload model and inference assets (host → board)

```bash
# host scp
scp board_infer.py yolo_decode.py input_640.npy YoloV8DetectExport.xmodel ubuntu@10.0.0.20:~/kv260/yolov8-dpu/
```

- `input_640.npy`: preprocessed on host inside the container with `letterbox` (matching quantization) from COCO128 `000000000071.jpg`, [1,640,640,3] float32 [0,1] (board has no OpenCV, preprocessing stays on host)
- On-board models (md5 verified against host):
  - `YoloV8DetectExport.xmodel` = v2 (4485383 B, fp 0x56)
  - `YoloV8DetectExport_fp56_bad.xmodel` = v1 backup (4479143 B, fp 0x56)

## Step 14: On-board inference script board_infer.py (fixed)

```bash
LD_PRELOAD=/usr/local/lib/libxcl_stub.so python3 board_infer.py input_640.npy
```

Pitfall fixes (three):
1. After `xir.Graph.deserialize`, you must traverse `root.get_children()` to find the `device=='DPU'` subgraph to create the runner — `get_root_subgraph()` itself has no runner attribute
2. There is no `get_children_subgraphs()` method (verified API is `get_children()`, confirmed via `dir(root)` probing)
3. Outputs: `board_out.npz` (3 head tensors) + `board_dets.json` (decoded boxes) → for host-side PyTorch golden comparison

Board-side decode `yolo_decode.py` (pure numpy): DFL softmax16 + per-class NMS + letterbox inverse, auto-compatible with NHWC/NCHW.

## Step 15: Recompile with scheme 1 succeeds, fingerprint aligned (early 2026-08-16, milestone OK)

**Background**: `target` and `fingerprint` in `arch.json` are mutually exclusive if/elif (`vai_c_xir.py` source lines 85-115: `if 'target' in data` → target path; `elif 'fingerprint' in data` → fingerprint path). The earlier v2 had both fields, so `target` took priority → `fingerprint` ignored → model still 0x56.

**Action**: new `arch_kv260_fp_only.json` containing only the fingerprint field, recompile:

```json
{
    "fingerprint": "0x101000016010407"
}
```

```bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
vai_c_xir -x quantize_result_full/YoloV8DetectExport_int.xmodel \
          -a arch_kv260_fp_only.json -o compiled_full_v3 -n deploy
```

**Pitfall**: writing the JSON with `cat >` inside `docker exec bash -c "..."` gets the double quotes swallowed by the outer shell (`json.load` fails but the source's catch-all `except:` reports "Unable to open file"). Solved by writing it with an in-container `python3 -c "json.dump(...)"`.

**Verification**: compile log `Target architecture: DPUCZDX8G_ISA1_B4096_0101000016010407` (with 0x16 fingerprint); xir shows DPU subgraph `dpu_fingerprint = 72339069383803911 = 0x101000016010407`, **exactly matching the hardware** OK

## Step 16: On-board inference success + accuracy verification (2026-08-16 OK)

Upload the v3 model (paramiko SFTP; scp password auth unavailable), then run:

```bash
cd ~/kv260/yolov8-dpu && LD_PRELOAD=/usr/local/lib/libxcl_stub.so \
  python3 board_infer.py input_640.npy
```

Results:
- DPU execution **52.5 ms** (~19 FPS, B4096 single core 300 MHz)
- Input `[1,640,640,3] float32` → 3 head outputs `xint8`: `(1,80,80,144)/(1,40,40,144)/(1,20,20,144)`
- **DETECTIONS: 1**: cls=6, score=0.731, box=[71.5, 295.4, 528.1, 412.2]
- Saved `board_out.npz` + `board_dets.json`

**Accuracy comparison (test image COCO128 `000000000071.jpg`)**:

| Metric | PyTorch golden | Board DPU (int8) |
|--------|---------------|------------------|
| Detections | 1 | 1 |
| Class | cls=6 | cls=6 |
| Confidence | 0.718 | 0.731 |
| Box | [54, 292, 530, 412] | [71, 295, 528, 412] |

Quantization accuracy loss is minimal (box diff <20px, confidence diff 0.013) — the full deployment chain is wired up.

## Step 17: Web display service (2026-08-16 OK)

**Architecture**: board Flask service (systemd-managed) + MJPEG video stream + JSON API + web page. Camera hot-plug adaptive: USB camera plugged in → live stream automatically; otherwise falls back to a demo image loop (6 COCO128 test images).

**Files** (host `web/` synced with board `~/kv260/yolov8-dpu/web/`):
- `camera.py` — pure v4l2 ioctl (no OpenCV) reads YUYV 640x480 → numpy RGB; hot-plug + 2s retry throttle; `DemoSource` loops demo images
- `infer.py` — `DpuDetector` wrapper (xir → DPU subgraph → VART runner → 3 heads → yolo_decode → letterbox inverse)
- `draw.py` — PIL box drawing (replacing cv2), v2: English COCO labels + Apple-minimal style (2px thin boxes, rounded dark label chips, Apple system color 8-tone palette, DejaVu sans-serif font)
- `app.py` — Flask: `/` page, `/video_feed` MJPEG, `/api/status` JSON, `/api/source` input switch
- `templates/index.html` — v1 impressionist (Monet/Van Gogh tones, frame layout); v2 rebranded to **Apple design language** (all-English, deep-space black background, aurora 3-color gradient glows, glassmorphism cards, large tabular-nums metrics, Cormorant italic art accents, 11.7KB page); v3 rebranded for the job application to **"Kria Edge Vision — Real-Time Object Detection"** (title/wordmark/footer, subtitle `real-time object detection · YOLOv8n · Vitis AI DPU`, app.py comments updated); v4 enlarged layout (wrap 1120→1380px, metric values 27→37px, list 14→15px, overall font/spacing up) + FPS/ms units (innerHTML so textContent does not wipe them) + demo mode auto-hides Notes card; v5 undid demo hiding (Notes shows in both camera/demo) + added English `README.md` (architecture/quantization/deployment/startup/API, interview-grade) + board systemd `kv260-vision.service` Description updated to "Kria Edge Vision"; v6 README added **Roadmap** section (Stage 1 current/done, Stage 2 planned: Vitis HLS moves letterbox, DFL decode, NMS into PL so the ARM CPU drops out of the per-frame critical path, before/after comparison via dashboard metrics); next (tomorrow): first verify Stage 1 with a Logitech C920 real capture, then start Stage 2 HLS development

**systemd deployment** (key: ending the SSH session kills nohup background processes, so systemd is mandatory):
```bash
# host writes /etc/systemd/system/kv260-vision.service (sudo password kv260dev)
[Service]
Environment=LD_PRELOAD=/usr/local/lib/libxcl_stub.so
WorkingDirectory=/home/ubuntu/kv260/yolov8-dpu/web
ExecStart=/usr/bin/python3 /home/ubuntu/kv260/yolov8-dpu/web/app.py --port 8080
Restart=always
RestartSec=3
# systemctl enable + start
```

**Pitfalls (5 this round, all resolved)**:
1. **SEGV use-after-free (nastiest)**: the graph returned by `xir.Graph.deserialize()` was a `__init__` local variable, GC'd after the function returned → the VART runner held a dangling reference → the next `execute_async` crashed in `DpuKernel::get_fingerprint()` (gdb stack jumped to string "locatio..."). Fix: keep `self._graph = g` reference. `board_infer.py` does not crash because the graph is a module-level global
2. **v3 model subgraph structure**: root has no `runner` attr (fingerprint-only compile); the DPU subgraph is among the children; there is also a sibling CPU subgraph with a runner attr pointing to `libvart-cpu-runner.so` — must match `device=='DPU'` exactly, otherwise "cannot open library libvart-cpu-runner.so"
3. **MJPEG re-sending stale frames**: the generator re-sent the same JPEG while the worker had not updated (17.8MB/8s) → changed to push only when the `frame` sequence number changes (409KB/6s)
4. **Camera missing blocked the loop**: auto mode tried open every frame and `sleep(1s)` on failure (fps 0.7) → changed to 2s retry throttle (fps 3.1)
5. **pkill self-kill**: `pkill -f "python3 app.py"` matched the `sh -c` wrapper process of the command itself → use the `[a]pp.py` regex trick

**Measured results**:
- `GET /api/status` → JSON: `{fps, dpu_ms, det_count, detections[{cls,box,score}], source}`
- `GET /video_feed` → MJPEG 200 (only new frames pushed, ~240KB/s)
- `GET /` → web page 11.7KB
- Host access `http://10.0.0.20:8080` OK (10.0.0.10 client)
- demo mode fps≈3.1 (DPU 175ms is the main bottleneck: 52ms inference + letterbox + DFL/NMS decode)
- Single real-detection verification: dets=3 (train cls=6 + car cls=2), coordinates correctly inverse-mapped back to the original image

**Camera integration**: plug in USB camera (/dev/video0), the service auto-detects and switches to live stream within 2s; `POST /api/source {"mode":"cam"|"demo"|"auto"}` forces switching. Camera-mode frame rate is limited by DPU inference speed (~3-5fps, B4096 single core).

## Step 18: Logitech C920 camera integration fix (2026-08-16 OK)

**Symptom**: plugging in a Logitech C920 (USB camera) got "no reaction" — the page stayed in demo mode, source did not switch to camera.

**Debugging**: SSH to the board showed `lsusb` lists `046d:08e5 Logitech HD Pro Webcam C920`, `/dev/video0` exists, `ubuntu` is in the `video` group, the service runs fine — **the system layer fully recognizes the camera; the problem is the app layer reading no frames**.

**Root cause (three stacked, all in `web/camera.py`)**:
1. **Wrong ioctl number (primary)**: `VIDIOC_S_FMT = 0xC0D85605` is the **x86_64 value**. ioctl numbers encode the struct size; on aarch64 `struct v4l2_format` is 208 bytes (x86_64 is 216 bytes), so the correct value is `0xC0D05605`. The driver receives an unknown ioctl and returns ENOTTY (Errno 25 "Inappropriate ioctl for device") → `open()` always failed → forever in demo.
2. **uvcvideo does not support the `read()` method**: after fixing the ioctl number, STREAMON returned EINVAL — the uvcvideo driver only supports the standard mmap streaming path (`v4l2-ctl --stream-mmap` successfully grabbed full 614400-byte frames, proving the camera hardware is fully functional).
3. **`fcntl.ioctl` with `bytes()` loses write-back**: QUERYBUF/DQBUF are `_IOWR` ioctls requiring the kernel to fill back `length/offset/bytesused`, but passing immutable `bytes()` loses all write-back (returns all 0). Must pass mutable `bytearray`.

**Verification method**: compiled a C program on the board printing the real ioctl constants from system headers (`VIDIOC_S_FMT=0xC0D05605`, REQBUFS/QUERYBUF/QBUF/DQBUF/STREAMON etc.) and `struct v4l2_format`/`v4l2_buffer` field offsets (`type@0`, `fmt.pix@8` (with 4-byte padding), `width@8`, `height@12`, `pixelformat@16`, `field@20`, `bytesperline@24`, `sizeimage@28`, `colorspace@32`; `v4l2_buffer` `m.offset@64` is u64, `length@72`), locating each ioctl one by one.

**Fix**: rewrote `USBCamera` as the standard v4l2 mmap flow: `S_FMT(208B layout) → REQBUFS(4 buffers) → QUERYBUF+mmap → QBUF → STREAMON → select+DQBUF → QBUF loop`. All ioctls now pass mutable `bytearray`, constants updated to the aarch64 values confirmed on the board, with comments on cross-architecture differences.

**Result**: `/api/status` returns `"source":"camera"`, fps≈4.1, detecting the real scene (laptop cls=63, remote cls=66). Hot-plug rescans `/dev/video*` every 2s: unplug → back to demo, plug in → back to camera, no restart needed.

**File sync (three locations identical)**: board `/home/ubuntu/kv260/yolov8-dpu/web/camera.py`, GitHub repo `web/camera.py`, source dir `/home/luyuan/kv260/yolov8-dpu/web/camera.py`.

## Step 19: NMS vectorization + per-stage timing display (2026-08-16 OK)

**Changes**:
1. `web/yolo_decode.py` `nms()` fully vectorized: `top_k=1000` candidate trimming + boolean-mask batch IoU suppression, no more rebuilding numpy arrays per candidate. Results bit-identical to the old implementation (verified with 6 random board datasets, including strong same-class clusters; worst case 8400 anchors 254ms→161ms, 1.6x), real-scene gains are larger (top-K trims lots of low-score noise first, and only high-score candidates enter the loop).
2. `web/infer.py` `detect()` returns per-stage timings: `lb_ms` (letterbox+normalize, CPU) / `dpu_ms` (pure DPU inference) / `dec_ms` (DFL softmax16 + NMS + letterbox inverse, CPU).
3. `web/app.py`: state and `/api/status` gain `lb_ms/dpu_ms/dec_ms/cpu_ms` (EMA 0.9/0.1 smoothing, consistent with fps).
4. `web/templates/index.html`: new `CPU: prep X + decode Y ms` line under the DPU card.

**Measured (board camera mode)**: fps≈3.7-4.2, `dpu_ms=48.6` (only ~30% of the frame time), `lb_ms=8.6`, `dec_ms=109` — confirms the bottleneck is the ARM CPU Python decode, not the DPU, providing the baseline for moving decode to the PL with Vitis HLS.

**File sync (three locations identical)**: board, GitHub, source dir `yolo_decode.py / infer.py / app.py / templates/index.html`.

## Step 20: Web Latency panel + dropping the 480 downscale plan (2026-08-16 OK)

**Changes**:
1. `web/app.py`: `state` and `/api/status` gain `total_ms` (whole-frame latency = read frame + detect + draw + JPEG encode, reciprocal of fps, EMA 0.9/0.1 smoothed).
2. `web/templates/index.html`: new full-width **Latency** section (between metrics and detections), 4 columns: `Whole frame` (badge synced) / `DPU` / `CPU prep` (letterbox+normalize) / `CPU decode` (DFL+NMS+inverse); Whole frame column sub shows `CPU X + DPU Y ms`. Apple-style `.latgrid` 4-column grid, auto 2 columns on narrow screens.

**Decision (fps already 4.3, user approved)**:
- **Dropped the 480 downscale route** (former "next step") — requires re-quantize/recompile of the xmodel and small-target detection drops; no longer doing it.
- **CPU speed-up directions changed to (implement tomorrow, do not do early)**: ① multithreading — parallelize capture/DPU/decode pipeline, pure software, no xmodel change; ② Vitis HLS — move letterbox/DFL/NMS into PL, ARM CPU drops out of the per-frame critical path (Stage 2), compare before/after with this panel.

**Measured baseline (camera mode, for tomorrow's comparison)**: fps≈4.3, `total_ms≈230`, `dpu_ms≈48.6`, `lb_ms≈8.6`, `dec_ms≈109` (CPU decode ~118ms ≈ 51% of the frame, the main bottleneck).

**File sync (three locations identical)**: board, GitHub, source dir `app.py / templates/index.html`.

## Step 21: multithreading pipeline parallelism (2026-08-16 OK)

**Changes (only `web/app.py`)**: single-thread worker → 3-stage pipeline:
- `capture_loop` (capture thread): `cam.read()` + read_ms timing → `q_frames` (bounded `queue.Queue(maxsize=2)`)
- `detect_loop` (detect thread): `detector.detect()` (lb+DPU+dec) → `q_res` (bounded maxsize=2); DPU VART runner is not thread-safe, only this thread calls it
- main loop: draw + JPEG encode + state update; fps now measures "interval between adjacent completed frames" (pipeline throughput, not per-frame latency)

Bounded queues naturally back-pressure: the downstream being slow no longer buffers stale frames (prevents latency bloat). GIL impact is small: numpy/PIL/DPU heavy ops all release the GIL, so real parallelism works. `read_ms` is now timed by the capture thread and passed through the queue.

**Measured (board camera mode)**: fps≈5.9 (baseline 4.3, +37%). Bottleneck changed from "total 230ms" to "slowest stage detect≈168ms" (lb 16.7 + DPU 49.1 + dec 102.6); read_ms dropped from 174 (including waiting for the pipeline) to 74.6. CPU decode dec≈102.6ms is still the main critical-path item — exactly the Vitis HLS target for tomorrow (move to PL in Step 2).

**Semantic change (important, dashboard panel updated)**: total_ms (sum of six EMAs) = per-frame end-to-end work/latency (cam→screen), while wall-clock frame period = 1000/fps. Pipeline stages run in parallel on different frames, so period (≈172ms) < work (≈285ms); the difference is the overlap hidden by the pipeline (≈113ms) — fps is a throughput metric, total_ms is a latency metric. The "Whole frame" panel card now shows the wall-clock period (new API field `clock_ms`), sub-line shows "Σ work / pipeline overlaps", avoiding the 283ms vs 5.9fps confusion.

**Measured (board camera mode, Step 21 addendum)**: fps≈5.8, clock≈172.6ms, total≈284.6ms (read 74.2 + lb 17.0 + dpu 49.8 + dec 105.5 + draw 25.7 + enc 12.5, the six-stage sum matches). Bottleneck stage detect≈172ms (lb+DPU+dec); CPU decode dec≈105.5ms remains the main item.

**File sync (three locations identical)**: board, GitHub, source dir `web/app.py`.

## Step 21.1: Dashboard layout fix (2026-08-16 OK)

**Problem**: the right panel was originally 4 stacked sections (metrics / latency / detections / notes); next to the 4:3 video card (height≈613px) the total height was ≈950px, pushing the Detections list (max-height 360px) and Notes below the viewport — the page overflowed vertically.

**Solution** (keep the info, don't delete): Detections and Notes moved out of the right panel into a full-width horizontal band below the video+panel row (`.bottom{grid-template-columns:1fr auto}`, auto single column on narrow screens). The Detections list changed from vertical scroll to `flex-wrap` horizontal chips (colored dot + class + score + coordinates); no matter how many, they wrap downward instead of blowing up the viewport. The right panel now only holds metrics + latency, height≈460px < video height, flush with the video row.

**Changes**: only `web/templates/index.html` (CSS + DOM structure), zero JS/backend changes (all `#detlist`/`#detBadge`/`#notesBox` ids kept). Board verification: `bottom`/`panel` each 1, `detlist`/`notesBox` all inside the band.

**File sync (three locations identical)**: board, GitHub, source dir `web/templates/index.html`.

## Step 21.2: Remove Detections / Notes sections (2026-08-16 OK)

**Decision**: the page keeps only the video + metrics (metrics / latency). Rationale: the video overlay already draws class+confidence, so a text list is redundant for demos; Notes is static prose. No information lost: `/api/status` `detections` (cls/score/box) and `det_count` remain, for the future cafe/bar counting feature.

**Changes** (only `web/templates/index.html`): deleted the `.bottom` band HTML/CSS, `#detlist`/`.legend` styles, and the JS `COCO_EN`/`PALETTE` constants + detlist rendering logic (`detBadge` removed too). Backend `det_count`/`detections` fields kept.

**Verification**: board page grep `detlist|notesBox|detBadge|bottom` = 0; API normal (fps 5.9, clock 168.9ms, det_count=2).

**File sync (three locations identical)**: board, GitHub, source dir `web/templates/index.html`.

## Next Steps (optional optimization)

- [x] **NMS vectorization** (done → Step 19, 1.6x worst case)
- [x] **480p input downscale** (dropped: requires re-quantize/recompile of the xmodel, small-target detection drops; decision made 2026-08-16)
- [x] **USB camera real test** (done → Step 18: C920 hot-plug + mmap capture verified)
- [x] **multithreading to speed up CPU decode** (done → Step 21: capture/detect/draw 3-stage pipeline, fps 4.3→5.9; tomorrow HLS continues reducing the detect-stage time)
- [ ] **Vitis HLS decode to PL** (tomorrow direction, Stage 2: letterbox/DFL/NMS into PL, compare with Step 20 dashboard metrics; goal: move the dec 102.6ms + lb 16.7ms of the detect-stage 168ms out of the CPU critical path)
- [ ] Production-grade WSGI (gunicorn) + HTTPS/LAN deployment
- [ ] cafe/bar-scene-specific: show only BAR_CLASSES subset + counting
