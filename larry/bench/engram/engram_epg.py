#!/usr/bin/env python3
"""engram_epg.py convert SRC.safetensors LAYER OUT.epg | verify SRC.safetensors LAYER OUT.epg [--sample N]

larry: an Engram table (layers.LAYER.engram.embed.weight F8_E4M3 [rows, 256] + .scale F8_E8M0 [rows, 8] in a HF
safetensors shard: the scales first, then the weights - two random reads a row) rewritten as 4 KiB pages of 15 rows,
each row its 256 weight bytes, its 8 scale bytes and 8 zero bytes (272 B; 16 zero bytes end a page), so a row is ONE
aligned read.  Page 0 is the header: magic "EPG1\\0\\0\\0\\0", u64 rows, u32 row bytes (272), u32 rows a page (15),
u32 weight bytes (256), u32 scale bytes (8), u32 page bytes (4096), u32 layer; row r is at 4096 * (1 + r // 15) +
272 * (r % 15).  The source is only read (it is the downloaded repo's file).  verify: every row (or --sample N random
rows) compared with the source, byte for byte."""
import json
import os
import struct
import sys
import time

import numpy as np

RPP, ROW, PAGE, WB, SB = 15, 272, 4096, 256, 8


def tensors(src, layer):
    with open(src, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    base = 8 + n
    sc = h['layers.%d.engram.embed.scale' % layer]
    wt = h['layers.%d.engram.embed.weight' % layer]
    rows = sc['shape'][0]
    assert sc['dtype'] == 'F8_E8M0' and sc['shape'] == [rows, SB], sc
    assert wt['dtype'] == 'F8_E4M3' and wt['shape'] == [rows, WB], wt
    return rows, base + sc['data_offsets'][0], base + wt['data_offsets'][0]


def pread_all(fd, n, off):
    b = bytearray(n)
    mv = memoryview(b)
    got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError('short read at %d' % (off + got))
        got += k
    return np.frombuffer(b, np.uint8)


def convert(src, layer, out):
    rows, s_off, w_off = tensors(src, layer)
    npages = (rows + RPP - 1) // RPP
    fd = os.open(src, os.O_RDONLY)
    tmp = out + '.part'
    od = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    hdr = bytearray(PAGE)
    struct.pack_into('<8sQIIIIII', hdr, 0, b'EPG1\0\0\0\0', rows, ROW, RPP, WB, SB, PAGE, layer)
    os.write(od, bytes(hdr))
    CH = 32768   # pages a chunk (~134 MB out)
    t0 = time.time()
    done_out = PAGE
    for p0 in range(0, npages, CH):
        p1 = min(npages, p0 + CH)
        r0, r1 = p0 * RPP, min(rows, p1 * RPP)
        nr = r1 - r0
        w = pread_all(fd, nr * WB, w_off + r0 * WB).reshape(nr, WB)
        s = pread_all(fd, nr * SB, s_off + r0 * SB).reshape(nr, SB)
        buf = np.zeros((p1 - p0, PAGE), np.uint8)
        rowsv = np.zeros(((p1 - p0) * RPP, ROW), np.uint8)
        rowsv[:nr, :WB] = w
        rowsv[:nr, WB:WB + SB] = s
        buf[:, :RPP * ROW] = rowsv.reshape(p1 - p0, RPP * ROW)
        mv = memoryview(buf).cast('B')
        wrote = 0
        while wrote < len(mv):
            wrote += os.write(od, mv[wrote:])
        os.posix_fadvise(fd, w_off + r0 * WB, nr * WB, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, s_off + r0 * SB, nr * SB, os.POSIX_FADV_DONTNEED)
        done_out += len(mv)
        if (p0 // CH) % 16 == 15:
            os.fdatasync(od)
            os.posix_fadvise(od, 0, done_out, os.POSIX_FADV_DONTNEED)
            el = time.time() - t0
            print('%5.1f%%  %.1f GB out  %.0f MB/s' % (100.0 * p1 / npages, done_out / 1e9, done_out / el / 1e6), flush=True)
    os.fdatasync(od)
    os.posix_fadvise(od, 0, done_out, os.POSIX_FADV_DONTNEED)
    os.close(od)
    os.close(fd)
    os.rename(tmp, out)
    print('wrote %s: %d rows, %d pages, %.2f GB in %.0f s' % (out, rows, npages + 1, done_out / 1e9, time.time() - t0))


def verify(src, layer, out, sample):
    rows, s_off, w_off = tensors(src, layer)
    fd = os.open(src, os.O_RDONLY)
    od = os.open(out, os.O_RDONLY)
    h = os.pread(od, 40, 0)
    magic, r2, row, rpp, wb, sb, page, lay = struct.unpack('<8sQIIIIII', h)
    assert (magic, r2, row, rpp, wb, sb, page, lay) == (b'EPG1\0\0\0\0', rows, ROW, RPP, WB, SB, PAGE, layer), h
    t0 = time.time()
    bad = 0
    if sample:
        rng = np.random.default_rng(12345)
        for r in rng.integers(0, rows, sample):
            r = int(r)
            e = os.pread(od, WB + SB, PAGE * (1 + r // RPP) + ROW * (r % RPP))
            if e[:WB] != os.pread(fd, WB, w_off + r * WB) or e[WB:] != os.pread(fd, SB, s_off + r * SB):
                bad += 1
        print('verify %s: %d random rows, %d differ (%.0f s)' % (out, sample, bad, time.time() - t0))
        return bad == 0
    npages = (rows + RPP - 1) // RPP
    CH = 32768
    for p0 in range(0, npages, CH):
        p1 = min(npages, p0 + CH)
        r0, r1 = p0 * RPP, min(rows, p1 * RPP)
        nr = r1 - r0
        w = pread_all(fd, nr * WB, w_off + r0 * WB).reshape(nr, WB)
        s = pread_all(fd, nr * SB, s_off + r0 * SB).reshape(nr, SB)
        e = pread_all(od, (p1 - p0) * PAGE, PAGE * (1 + p0)).reshape(p1 - p0, PAGE)
        er = e[:, :RPP * ROW].reshape((p1 - p0) * RPP, ROW)
        bad += int(np.count_nonzero(er[:nr, :WB] != w)) + int(np.count_nonzero(er[:nr, WB:WB + SB] != s))
        bad += int(np.count_nonzero(er[:, WB + SB:])) + int(np.count_nonzero(er[nr:])) + int(np.count_nonzero(e[:, RPP * ROW:]))
        os.posix_fadvise(fd, w_off + r0 * WB, nr * WB, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, s_off + r0 * SB, nr * SB, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(od, PAGE * (1 + p0), (p1 - p0) * PAGE, os.POSIX_FADV_DONTNEED)
        if (p0 // CH) % 64 == 63:
            print('%5.1f%% verified, %d bad bytes' % (100.0 * p1 / npages, bad), flush=True)
    print('verify %s: all %d rows, %d bad bytes (%.0f s)' % (out, rows, bad, time.time() - t0))
    return bad == 0


if __name__ == '__main__':
    mode, src, layer, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if mode == 'convert':
        convert(src, layer, out)
    else:
        n = int(sys.argv[sys.argv.index('--sample') + 1]) if '--sample' in sys.argv else 0
        sys.exit(0 if verify(src, layer, out, n) else 1)
