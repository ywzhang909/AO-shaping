"""对比 WFS_GetSpotfieldImage 和 WFS_GetSpotfieldImageCopy 的硬件测试脚本。

在已连接的 WFS 设备上测试：
1. 新函数 GetSpotfieldImageCopy (已绑定, 正确用法)
2. 旧函数 GetSpotfieldImage (未绑定, 原始用法)
3. exposure_time getter (byref 修复)
4. handle_error (WfsError 修复)
"""
from __future__ import annotations

import ctypes
from ctypes import byref, c_double, c_int32, c_uint8

import numpy as np

from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS, MlaRes, Mla_pix


def main():
    wfs = ThorlabWFS()
    print("[INFO] Opening WFS...")
    wfs.open()
    print(f"[OK] Connected: {wfs.device_name}, SN: {wfs.serial_num}")
    print(f"[OK] MLA: {wfs.mla_index.name}, image_pix: {wfs.image_pix}")

    # ── Test 1: exposure_time getter (byref fix) ──
    print("\n=== Test 1: exposure_time getter (byref fix) ===")
    try:
        t = wfs.exposure_time
        print(f"[PASS] exposure_time = {t} (type={type(t).__name__})")
        assert isinstance(t, float), f"Expected float, got {type(t)}"
    except Exception as e:
        print(f"[FAIL] exposure_time getter: {e}")

    # ── Test 2: take image ──
    print("\n=== Test 2: take_image ===")
    try:
        wfs.take_image()
        print("[PASS] take_image succeeded")
    except Exception as e:
        print(f"[FAIL] take_image: {e}")
        wfs.close()
        return

    # ── Test 3: NEW GetSpotfieldImageCopy (bound, correct) ──
    print("\n=== Test 3: WFS_GetSpotfieldImageCopy (NEW, bound) ===")
    try:
        max_h, max_w = 1024, 1280
        image_buf = np.empty((max_h, max_w), dtype=np.uint8)
        rows = c_int32()
        cols = c_int32()
        err = wfs._lib.WFS_GetSpotfieldImageCopy(
            wfs._instrument_handle,
            image_buf.ctypes.data_as(ctypes.POINTER(c_uint8)),
            byref(rows),
            byref(cols),
        )
        if err:
            wfs.handle_error(err)
            print(f"[FAIL] GetSpotfieldImageCopy returned error: {err}")
        else:
            img = image_buf[: rows.value, : cols.value]
            print(f"[PASS] GetSpotfieldImageCopy: shape={img.shape}, "
                  f"dtype={img.dtype}, min={img.min()}, max={img.max()}, "
                  f"mean={img.mean():.1f}, nonzero={np.count_nonzero(img)}")
    except Exception as e:
        print(f"[FAIL] GetSpotfieldImageCopy: {e}")

    # ── Test 4: OLD GetSpotfieldImage (unbound, original usage) ──
    print("\n=== Test 4: WFS_GetSpotfieldImage (OLD, unbound) ===")
    try:
        old_buf = np.empty([80, 80], dtype=np.uint8)
        rows_old = byref(c_int32(80))
        cols_old = byref(c_int32(80))
        err = wfs._lib.WFS_GetSpotfieldImage(
            wfs._instrument_handle,
            old_buf.ctypes.data_as(ctypes.POINTER(c_uint8)),
            rows_old,
            cols_old,
        )
        if err:
            print(f"[FAIL] GetSpotfieldImage returned error: {err}")
        else:
            actual_rows = ctypes.cast(rows_old, ctypes.POINTER(c_int32)).contents.value
            actual_cols = ctypes.cast(cols_old, ctypes.POINTER(c_int32)).contents.value
            print(f"[PASS] GetSpotfieldImage: shape={old_buf.shape}, "
                  f"reported_rows={actual_rows}, reported_cols={actual_cols}, "
                  f"min={old_buf.min()}, max={old_buf.max()}, "
                  f"mean={old_buf.mean():.1f}, nonzero={np.count_nonzero(old_buf)}")
    except Exception as e:
        print(f"[FAIL] GetSpotfieldImage: {type(e).__name__}: {e}")

    # ── Test 5: Compare both results ──
    print("\n=== Test 5: Compare both functions ===")
    try:
        # Re-run both to get fresh data
        # NEW
        img_new = np.empty((1024, 1280), dtype=np.uint8)
        r, c = c_int32(), c_int32()
        wfs._lib.WFS_GetSpotfieldImageCopy(
            wfs._instrument_handle,
            img_new.ctypes.data_as(ctypes.POINTER(c_uint8)),
            byref(r), byref(c),
        )
        img_new = img_new[: r.value, : c.value]

        # OLD
        img_old = np.empty([80, 80], dtype=np.uint8)
        ro, co = byref(c_int32(80)), byref(c_int32(80))
        wfs._lib.WFS_GetSpotfieldImage(
            wfs._instrument_handle,
            img_old.ctypes.data_as(ctypes.POINTER(c_uint8)),
            ro, co,
        )

        print(f"  NEW shape: {img_new.shape}, OLD shape: {img_old.shape}")
        print(f"  NEW nonzero: {np.count_nonzero(img_new)}, OLD nonzero: {np.count_nonzero(img_old)}")

        # Check if OLD result is a subset of NEW result
        if img_new.shape[0] >= 80 and img_new.shape[1] >= 80:
            overlap = img_new[:80, :80]
            if np.array_equal(overlap, img_old):
                print("  [MATCH] OLD result == NEW[0:80, 0:80] (byte-identical)")
            else:
                diff = np.abs(overlap.astype(int) - img_old.astype(int))
                print(f"  [DIFF] max_diff={diff.max()}, mean_diff={diff.mean():.2f}")
        else:
            print("  [SKIP] NEW image too small for overlap comparison")

    except Exception as e:
        print(f"[FAIL] Comparison: {e}")

    # ── Test 6: handle_error with WfsError ──
    print("\n=== Test 6: handle_error raises WfsError ===")
    try:
        from ao_shaping.drivers.wfs._thorlab_wfs import WfsError
        wfs.handle_error(-120)
        print("[FAIL] Should have raised WfsError")
    except WfsError as e:
        print(f"[PASS] WfsError raised: {e}")
    except Exception as e:
        print(f"[FAIL] Wrong exception type: {type(e).__name__}: {e}")

    # ── Cleanup ──
    print("\n=== Cleanup ===")
    wfs.close()
    print("[OK] WFS closed")


if __name__ == "__main__":
    main()
