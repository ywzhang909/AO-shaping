# coding: utf-8
"""
读取 Santec SLM 在指定波长下的实际相位调制量

功能：
1. 打开 SLM USB 控制接口
2. 设置 wavelength = 532 nm, target phase = 2.00 pi
3. 等待 SLM 从 BUSY 变为 READY
4. 读回 SLM 实际采用的 wavelength 和 phase
5. 计算 GS_max = int(2.0 / actual_phase_pi * 1023)

注意：
- 这里读到的是 SLM 控制器内部返回的 phase 参数；
- 它不是干涉仪实测相位，而是 SDK/控制板认为实际展开后的相位调制量。
"""

import ctypes
import time
import _slm_win as slm


def check_ret(ret, func_name):
    """检查 SLM 函数返回值"""
    if ret != slm.SLM_OK:
        raise RuntimeError(f"{func_name} failed, return code = {ret}")


def wait_slm_ready(slm_number=1, timeout_sec=80, interval_sec=1.0):
    """
    等待 SLM Ready.
    SLM_Ctrl_ReadSU 返回：
    SLM_OK = ready
    SLM_BS = busy
    """
    t0 = time.time()

    while True:
        ret = slm.SLM_Ctrl_ReadSU(slm_number)

        if ret == slm.SLM_OK:
            print("SLM status: READY")
            return True

        elif ret == slm.SLM_BS:
            elapsed = time.time() - t0
            print(f"SLM status: BUSY, waited {elapsed:.1f} s")
            if elapsed > timeout_sec:
                raise TimeoutError("SLM is still BUSY after timeout")
            time.sleep(interval_sec)

        else:
            raise RuntimeError(f"SLM_Ctrl_ReadSU failed, return code = {ret}")


def read_wavelength_phase(slm_number=1):
    """读取当前 wavelength 和 phase"""
    wavelength = ctypes.c_uint32(0)
    phase = ctypes.c_uint32(0)

    ret = slm.SLM_Ctrl_ReadWL(
        slm_number,
        ctypes.byref(wavelength),
        ctypes.byref(phase)
    )
    check_ret(ret, "SLM_Ctrl_ReadWL")

    wavelength_nm = wavelength.value
    phase_pi = phase.value / 100.0

    return wavelength_nm, phase.value, phase_pi


def set_and_read_actual_phase(
    slm_number=1,
    wavelength_nm=520,
    target_phase_pi=2.00,
    save_to_slm=False
):
    """
    设置波长和目标相位，然后读回实际相位调制量。

    参数：
    slm_number: SLM 编号，通常是 1
    wavelength_nm: 目标波长，例如 532
    target_phase_pi: 目标相位调制量，单位是 pi，例如 2.00
    save_to_slm:
        False: 只临时设置，不写入掉电保存
        True : 调用 SLM_Ctrl_WriteAW 保存到 SLM
    """

    target_phase_code = int(round(target_phase_pi * 100))

    print("Opening SLM USB interface...")
    ret = slm.SLM_Ctrl_Open(slm_number)
    check_ret(ret, "SLM_Ctrl_Open")

    try:
        print("Waiting SLM ready before setting...")
        wait_slm_ready(slm_number)

        print(f"Setting wavelength = {wavelength_nm} nm, target phase = {target_phase_pi:.2f} pi")
        ret = slm.SLM_Ctrl_WriteWL(
            slm_number,
            int(wavelength_nm),
            target_phase_code
        )
        check_ret(ret, "SLM_Ctrl_WriteWL")

        print("Waiting SLM ready after setting wavelength/phase...")
        wait_slm_ready(slm_number)

        if save_to_slm:
            print("Saving wavelength/phase settings to SLM...")
            ret = slm.SLM_Ctrl_WriteAW(slm_number)
            check_ret(ret, "SLM_Ctrl_WriteAW")

            print("Waiting SLM ready after saving...")
            wait_slm_ready(slm_number)

        read_wl_nm, read_phase_code, actual_phase_pi = read_wavelength_phase(slm_number)

        gs_max = int(2.0 / actual_phase_pi * 1023)

        print("\n========== Result ==========")
        print(f"Read wavelength        = {read_wl_nm} nm")
        print(f"Read phase code        = {read_phase_code}")
        print(f"Actual phase modulation= {actual_phase_pi:.4f} pi")
        print(f"GS_max                 = int(2.0 / {actual_phase_pi:.4f} * 1023) = {gs_max}")
        print("============================\n")

        return {
            "wavelength_nm": read_wl_nm,
            "phase_code": read_phase_code,
            "actual_phase_pi": actual_phase_pi,
            "GS_max": gs_max,
        }

    finally:
        print("Closing SLM USB interface...")
        ret = slm.SLM_Ctrl_Close(slm_number)
        if ret != slm.SLM_OK:
            print(f"Warning: SLM_Ctrl_Close returned {ret}")


if __name__ == "__main__":
    result = set_and_read_actual_phase(
        slm_number=1,
        wavelength_nm=520,
        target_phase_pi=2.00,
        save_to_slm=False
    )