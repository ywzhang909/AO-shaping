"""Hardware test for ThorlabWFS driver — reproduces suspected bugs on real device.

Run with the project venv:
    .venv\\Scripts\\python.exe scripts\\wfs_driver_hw_test.py

Each suspect is wrapped in try/except so one failure doesn't abort the rest.
Prints PASS/FAIL per check with exact exception text.
"""
from __future__ import annotations

import sys
from loguru import logger
import traceback

sys.path.insert(0, "src")
sys.path.insert(0, "libs")

from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS


def check(name: str, fn) -> None:
    try:
        result = fn()
        logger.info(f"[PASS] {name}: {result}")
    except Exception as e:  # intentional, we want all failures
        logger.info(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def main() -> None:
    wfs = ThorlabWFS(mla_index="512", high_speed=False, stable_sample_enable=False)
    logger.info("=== WFS open() ===")
    try:
        wfs.open()
        logger.info(f"[PASS] open(): state={wfs._state}, handle={wfs._instrument_handle.value}")
        logger.info(f"  mla_index={wfs.mla_index}, exp={wfs._explosure_time}, gain={wfs._gain}")
        logger.info(f"  pupil=({wfs.c_x},{wfs.c_y},{wfs.d_x},{wfs.d_y})")
    except Exception as e:  # 保留宽泛捕获: 设备 API 是多态实现 (真机/仿真/mock), 抛出的类型不可枚举
        logger.info(f"[FAIL] open(): {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return

    logger.info("\n=== Bug #1: exposure_time property GETTER (missing byref) ===")
    check("exposure_time getter", lambda: wfs.exposure_time)

    logger.info("\n=== Bug #2: high_speed property GETTER (double .value) ===")
    check("high_speed getter", lambda: wfs.high_speed)

    logger.info("\n=== Bug #3: get_spotfiled_image() (unbound fn + buffer overflow) ===")
    check("get_spotfiled_image", wfs.get_spotfiled_image)

    logger.info("\n=== P1 #4: handle_error with a real error code ===")
    check("handle_error(-120)", lambda: wfs.handle_error(-120))

    logger.info("\n=== P1 #6: open() already done; verify take_image + get_wavefront work ===")
    check("take_image", lambda: wfs.take_image())
    check("get_wavefront", lambda: wfs.get_wavefront().shape)

    logger.info("\n=== P1 #5: get_zernike(5) (unbound ZernikeLsf) ===")
    check("get_zernike(5)", lambda: wfs.get_zernike(5).shape)

    logger.info("\n=== get_spots_statics (regression sanity) ===")
    check("get_spots_statics", lambda: (wfs.get_spots_statics()[0].shape, len(wfs.get_spots_statics()[1])))

    logger.info("\n=== high_speed SETTER round-trip (Bug #8 silent fail) ===")
    check("high_speed setter True", lambda: setattr(wfs, "high_speed", True))

    logger.info("\n=== cleanup ===")
    wfs.close()
    logger.info("[PASS] close()")


if __name__ == "__main__":
    main()