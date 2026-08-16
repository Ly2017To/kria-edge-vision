/*
 * xcl_stub.c - KV260 board-side XRT 2.13 workaround for VART 3.5.0
 *
 * KV260's trimmed libxrt_core.so.2 (XRT 2.13) does not export the
 * xclIP* register-range family (xclIPSetReadRange / xclIPReadRange /
 * xclIPWriteRange) that VART 3.5.0 was linked against (built for XRT 2.15).
 *
 * The DPU inference hot path uses ERT commands (xrt::run start/wait),
 * NOT these symbols, so a zero-returning stub only needs to make dlopen
 * succeed. If a stub IS called it prints a log line so we can detect it.
 *
 * Build:
 *   gcc -shared -fPIC -o libxcl_stub.so xcl_stub.c
 *   sudo cp libxcl_stub.so /usr/local/lib/ && sudo ldconfig
 * Run:
 *   LD_PRELOAD=/usr/local/lib/libxcl_stub.so xdputil query
 */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static void stub_log(const char *fn, uint32_t ip, uint64_t a, uint64_t b) {
    fprintf(stderr, "[stub] %s ip=%u a=0x%lx b=0x%lx\n",
            fn, ip, (unsigned long)a, (unsigned long)b);
}

int xclIPSetReadRange(void *handle, uint32_t ipIndex, uint64_t start, uint64_t size) {
    stub_log("xclIPSetReadRange", ipIndex, start, size);
    return 0;
}

int xclIPReadRange(void *handle, uint32_t ipIndex, uint64_t start, uint64_t size, void *dst) {
    stub_log("xclIPReadRange", ipIndex, start, size);
    if (dst) memset(dst, 0, (size_t)size);
    return 0;
}

int xclIPWriteRange(void *handle, uint32_t ipIndex, uint64_t start, uint64_t size, const void *src) {
    stub_log("xclIPWriteRange", ipIndex, start, size);
    return 0;
}
