#!/usr/bin/env python3
"""KV260 USB camera capture via raw v4l2 mmap (no OpenCV) + demo fallback source.

- Hot-plug aware: re-scans /dev/video* on every failure, so plugging in a
  USB camera mid-run just works.
- YUYV 640x480 -> RGB via numpy (no cv2).
- Uses the standard v4l2 mmap streaming path (REQBUFS/QUERYBUF/mmap/QBUF/
  STREAMON/DQBUF). The uvcvideo driver does NOT support the legacy read()
  method (S_FMT works, but STREAMON fails with EINVAL).
- If no camera is present, falls back to a looping demo image source so the
  web UI still shows live-looking results.

KV260 / aarch64 gotchas (2026-08):
- ioctl numbers encode the struct size, so VIDIOC_S_FMT differs from x86_64:
  0xC0D05605 here (struct v4l2_format = 208 B) vs 0xC0D85605 on x86_64.
- fcntl.ioctl() MUST receive a mutable bytearray for _IOWR ioctls (QUERYBUF,
  DQBUF, S_FMT write-back); passing bytes() silently drops the kernel's
  returned values (length/offset/bytesused all come back 0).
"""
import os
import glob
import time
import struct
import fcntl
import select
import mmap as mmap_mod

import numpy as np
from PIL import Image

# ---- v4l2 constants (aarch64, confirmed on board) ----
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_PIX_FMT_YUYV = 0x56595559  # fourcc 'YUYV'
VIDIOC_S_FMT = 0xC0D05605       # _IOWR('V',5, struct v4l2_format=208B)
VIDIOC_REQBUFS = 0xC0145608     # _IOWR('V',8, struct v4l2_requestbuffers=20B)
VIDIOC_QUERYBUF = 0xC0585609    # _IOWR('V',9, struct v4l2_buffer=88B)
VIDIOC_QBUF = 0xC058560F        # _IOWR('V',15, struct v4l2_buffer)
VIDIOC_DQBUF = 0xC0585611       # _IOWR('V',17, struct v4l2_buffer)
VIDIOC_STREAMON = 0x40045612    # _IOW('V',18, __u32)
VIDIOC_STREAMOFF = 0x40045613   # _IOW('V',19, __u32)

# struct v4l2_format field offsets (aarch64, 208-byte struct)
_V4L2_FMT_OFF = {'width': 8, 'height': 12, 'pixelformat': 16,
                 'field': 20, 'bytesperline': 24, 'sizeimage': 28,
                 'colorspace': 32}
# struct v4l2_buffer field offsets (aarch64, 88-byte struct)
_V4L2_BUF_OFF = {'index': 0, 'type': 4, 'bytesused': 8, 'flags': 12,
                 'memory': 60, 'm_offset': 64, 'length': 72}


def find_camera():
    cams = sorted(glob.glob('/dev/video*'))
    return cams[0] if cams else None


def yuyv_to_rgb(buf, w, h):
    """YUYV (2 bytes/pixel) -> RGB uint8 (h, w, 3)."""
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w * 2).astype(np.float32)
    y = a[:, 0::2]
    u = a[:, 1::4].repeat(2, axis=1)[:, :w]
    v = a[:, 3::4].repeat(2, axis=1)[:, :w]
    r = y + 1.402 * (v - 128.0)
    g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
    b = y + 1.772 * (u - 128.0)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


class USBCamera:
    """v4l2 mmap streaming capture from /dev/videoN. Returns None when no frame."""

    def __init__(self, width=640, height=480, nbufs=4):
        self.width = width
        self.height = height
        self.nbufs = nbufs
        self.fd = None
        self.dev = None
        self._maps = []          # (mmap object, length)
        self._last_error = 'not opened yet'
        self._retry_at = 0.0     # throttle re-open attempts when no camera

    @property
    def last_error(self):
        return self._last_error

    def _set_fmt(self, fd):
        """S_FMT to YUYV WxH. bytearray so the driver can write back."""
        fmt = bytearray(208)
        struct.pack_into('I', fmt, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)   # type
        struct.pack_into('I', fmt, _V4L2_FMT_OFF['width'], self.width)
        struct.pack_into('I', fmt, _V4L2_FMT_OFF['height'], self.height)
        struct.pack_into('I', fmt, _V4L2_FMT_OFF['pixelformat'], V4L2_PIX_FMT_YUYV)
        struct.pack_into('I', fmt, _V4L2_FMT_OFF['field'], 0)
        struct.pack_into('I', fmt, _V4L2_FMT_OFF['colorspace'], 1)   # SMPTE170M
        fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
        # self.width/height may be adjusted by the driver; keep ours for now.

    def _reqbufs(self, fd):
        """REQBUFS: request N mmap buffers."""
        rb = bytearray(20)  # struct v4l2_requestbuffers
        struct.pack_into('I', rb, 0, self.nbufs)                     # count
        struct.pack_into('I', rb, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)    # type
        struct.pack_into('I', rb, 8, V4L2_MEMORY_MMAP)               # memory
        fcntl.ioctl(fd, VIDIOC_REQBUFS, rb)

    def _querybuf(self, fd, index):
        """QUERYBUF for one buffer, return (offset, length)."""
        b = bytearray(88)  # struct v4l2_buffer
        struct.pack_into('I', b, _V4L2_BUF_OFF['index'], index)
        struct.pack_into('I', b, _V4L2_BUF_OFF['type'], V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, b)
        offset = struct.unpack_from('Q', b, _V4L2_BUF_OFF['m_offset'])[0]
        length = struct.unpack_from('I', b, _V4L2_BUF_OFF['length'])[0]
        return offset, length

    def _qbuf(self, fd, index):
        """QBUF: enqueue a buffer for capture."""
        b = bytearray(88)
        struct.pack_into('I', b, _V4L2_BUF_OFF['index'], index)
        struct.pack_into('I', b, _V4L2_BUF_OFF['type'], V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into('I', b, _V4L2_BUF_OFF['memory'], V4L2_MEMORY_MMAP)
        fcntl.ioctl(fd, VIDIOC_QBUF, b)

    def open(self):
        self.close()
        dev = find_camera()
        if not dev:
            self._last_error = 'no /dev/video* device'
            return False
        try:
            fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
            self._set_fmt(fd)
            self._reqbufs(fd)
            for i in range(self.nbufs):
                off, length = self._querybuf(fd, i)
                m = mmap_mod.mmap(fd, length, mmap_mod.MAP_SHARED,
                                  mmap_mod.PROT_READ | mmap_mod.PROT_WRITE,
                                  offset=off)
                self._maps.append((m, length))
                self._qbuf(fd, i)
            fcntl.ioctl(fd, VIDIOC_STREAMON,
                        struct.pack('I', V4L2_BUF_TYPE_VIDEO_CAPTURE))
            self.fd = fd
            self.dev = dev
            self._last_error = ''
            return True
        except Exception as e:
            self._last_error = 'open {}: {}'.format(dev, e)
            try:
                os.close(fd)
            except Exception:
                pass
            self._maps = []
            return False

    def read(self):
        if self.fd is None:
            # no camera: retry at most every 2s (don't stall the loop)
            if time.time() < self._retry_at:
                return None
            if not self.open():
                self._retry_at = time.time() + 2.0
                return None
        want = self.width * self.height * 2
        try:
            r, _, _ = select.select([self.fd], [], [], 1.0)
            if not r:
                return None
            b = bytearray(88)
            struct.pack_into('I', b, _V4L2_BUF_OFF['type'], V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into('I', b, _V4L2_BUF_OFF['memory'], V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
            index = struct.unpack_from('I', b, _V4L2_BUF_OFF['index'])[0]
            bytesused = struct.unpack_from('I', b, _V4L2_BUF_OFF['bytesused'])[0]
            mm, _ = self._maps[index]
            mm.seek(0)
            buf = mm.read(bytesused if bytesused > 0 else want)
            self._qbuf(self.fd, index)  # re-enqueue immediately
            if len(buf) < want:
                return None
            return yuyv_to_rgb(buf[:want], self.width, self.height)
        except BlockingIOError:
            return None
        except Exception as e:
            self._last_error = 'read: {}'.format(e)
            self.close()
            return None

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF,
                            struct.pack('I', V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except Exception:
                pass
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None
            self.dev = None
        for m, _ in self._maps:
            try:
                m.close()
            except Exception:
                pass
        self._maps = []


class DemoSource:
    """Loops demo images from a directory, mimicking a live ~8fps source."""

    def __init__(self, img_dir):
        self.files = sorted(
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.png')))
        self.idx = 0
        self.last_error = 'no demo images in {}'.format(img_dir) if not self.files else ''

    def read(self):
        if not self.files:
            time.sleep(0.5)
            return None
        f = self.files[self.idx % len(self.files)]
        self.idx += 1
        try:
            img = Image.open(f).convert('RGB')
            img.thumbnail((640, 480))
            time.sleep(0.1)  # ~10fps pacing
            return np.asarray(img)
        except Exception as e:
            self.last_error = 'demo read {}: {}'.format(f, e)
            time.sleep(0.5)
            return None


class CameraManager:
    """auto = camera if present else demo; 'cam'/'demo' force a mode."""

    def __init__(self, demo_dir):
        self.cam = USBCamera()
        self.demo = DemoSource(demo_dir)
        self.mode = 'auto'
        self.mode_src = 'starting'

    def set_mode(self, mode):
        if mode not in ('auto', 'cam', 'demo'):
            mode = 'auto'
        self.mode = mode

    def read(self):
        if self.mode in ('auto', 'cam'):
            frame = self.cam.read()
            if frame is not None:
                self.mode_src = 'camera'
                return frame
            if self.mode == 'cam':
                time.sleep(0.1)
                return None
            if not self.cam.last_error:
                self.cam.last_error = 'no frame yet'
        self.mode_src = 'demo'
        frame = self.demo.read()
        if frame is None:
            time.sleep(0.1)
        return frame
