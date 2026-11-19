"""带锁的精确相位重放外壳: 只负责串行化占用与记录证据。

``python -m ao_shaping.tools.slm.transfer.replay_phases`` 能按归档重放
``<shape>_seed`` / ``<shape>_final`` 相位并在 Common native 位深与曝光下重新
测量, 但它自己不拿共享设备锁、也不等待别的会话 (见
``report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md``
§2: "该模块本身仍无锁, 本轮未提供包装器")。本脚本补上那层外壳:

1. **重放本身一定会开设备**, 所以这里没有 model-only 分支: 没有 ``--execute``
   时不转发、不等待、不加锁、不导入任何设备模块, 只打印拒绝原因和两份 help。
   ``--execute`` 由**本脚本**消费 (``replay_phases`` 自己没有这个开关, 转发过去
   会被 argparse 拒绝), 因此它不会出现在子进程的 argv 里 — 这是本脚本唯一不从
   转发列表里原样传下去的参数, 与验证器外壳语义一致;
2. 有 ``--execute`` 时懒导入 ``ao_shaping.tools.kernels.device_mutex``, 按
   ``--camera / --cam-id / --slm-number`` 调用 ``wait_until_idle``, 等不到就
   **不启动重放**;
3. 保留等待返回的锁; 没返回就自己拿同一条 bench 共享锁, 拿不到同样不启动;
4. 锁**一直握到重放进程返回**, 在 ``finally`` 里释放 (释放失败只记录, 不掩盖
   重放的退出码);
5. 之后把 ``device_lock.json`` 写进本次 ``--out``; 若重放在创建目录之前就失败,
   改写同级的 ``<out>.device_lock.json`` — 绝不创建运行目录 (重放本身用
   ``exist_ok=False``, 空目录会让下一次干净重跑也失败)。

两条**显式**前置条件 (同样在 ``--execute`` 之前拒绝, 不进入等待):

- ``--out`` 必填: 证据要有落脚点, 且重放目录不允许复用;
- ``--exposures`` 必须显式给出: 默认的 50/100 ms 是历史扫档, 验收计划要求现场
  曝光只能来自当次相机读回范围与安全上限, 不能靠默认值沿用。

``--wait-until-idle 0`` 是仓库约定: 既不等待也不加锁 (无 shadow lock), 证据里
明确记为 disabled。本脚本不新增任何验收指标, 也不改重放算法。
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Shared lock envelope (wait record, release, evidence write). Reusing the
#: validator wrapper's helpers keeps one definition of the lock protocol instead
#: of a drifted copy; that module is import-side-effect free.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import validate_transfer_shapes_hardware as _envelope

REPLAY_MODULE = "ao_shaping.tools.slm.transfer.replay_phases"
DEFAULT_WAIT_S = 120.0
LOCK_EVIDENCE_NAME = "device_lock.json"
SCHEMA = "replay_transfer_shapes_hardware/device_lock/1"


def build_parser() -> argparse.ArgumentParser:
    """Wrapper options only; every other token is forwarded verbatim.

    ``allow_abbrev=False`` keeps a replay option from being consumed as an
    abbreviation of ``--wait-until-idle`` (and vice versa).
    """
    parser = argparse.ArgumentParser(
        prog="replay_transfer_shapes_hardware.py",
        description=__doc__.splitlines()[0],
        add_help=False,
        allow_abbrev=False,
        epilog=(
            "All other options are forwarded verbatim to "
            f"{REPLAY_MODULE} (see its --help below). Unlike the validator "
            "wrapper there is no device-free mode: replay always opens the SLM "
            "and camera, so --execute is required and --help is the only "
            "device-free invocation."
        ),
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help plus the replay module's options, then exit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Required device gate: without it this wrapper refuses and no "
            "device module is imported. Consumed here (the replay module has "
            "no such flag), so the child never sees --execute."
        ),
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_S,
        metavar="SECONDS",
        help=(
            "Before the replay opens hardware, wait up to SECONDS for the "
            "device lock / other hardware processes to clear, then hold the "
            "device_mutex lock over the whole replay "
            f"(default: {DEFAULT_WAIT_S:g}). Pass 0 to bypass both the wait "
            "and the lock (repo convention: 0 = off)."
        ),
    )
    return parser


def _replay_help_text() -> str:
    """The replay module's own ``--help`` (all of its options, verbatim)."""
    env = _child_env()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", REPLAY_MODULE, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"could not read {REPLAY_MODULE} --help: {type(exc).__name__}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"{REPLAY_MODULE} --help exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    return proc.stdout


def _print_help(parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    print("\nReplay options (from replay_phases.py, forwarded verbatim):\n")
    try:
        print(_replay_help_text())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _child_env() -> dict[str, str]:
    """Environment for the child: ``src`` first on ``PYTHONPATH`` (``-m`` needs it)."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return env


def _has_option(argv: Sequence[str], name: str) -> bool:
    """True when ``--name`` or ``--name=...`` appears in ``argv``."""
    return any(token == name or token.startswith(f"{name}=") for token in argv)


def _evidence_doc(wait_s: float, argv: Sequence[str]) -> dict[str, Any]:
    """Evidence skeleton; filled in as the wait/lock/replay proceed."""
    return {
        "schema": SCHEMA,
        "script": Path(__file__).name,
        "replay_module": REPLAY_MODULE,
        "replay_argv": list(argv),
        "exposures_explicit": _has_option(argv, "--exposures"),
        "wait_until_idle_s": float(wait_s),
        "lock_mode": "wait+acquire" if wait_s > 0 else "disabled",
        "device_wait": None,
        "lock": {
            "path": None,
            "held": False,
            "acquired_utc": None,
            "released_utc": None,
            "held_s": None,
            "released": None,
            "release_error": None,
        },
        "replay_invoked": False,
        "replay_exit_code": None,
        "written_utc": None,
    }


def _invoke_replay(argv: Sequence[str]) -> int:
    """Run the replay module with the forwarded argv; return its exit code."""
    proc = subprocess.run(
        [sys.executable, "-m", REPLAY_MODULE, *argv],
        check=False,
        env=_child_env(),
    )
    return int(proc.returncode)


def _run_execute(wait_s: float, forwarded: Sequence[str]) -> int:
    """Wait, hold the bench lock over the replay, and record evidence."""
    # The ONLY import of device-side modules in this script: --help and the
    # refusal paths never reach it.
    from ao_shaping.tools.kernels import device_mutex

    out = _envelope._out_path(forwarded)
    doc = _evidence_doc(wait_s, forwarded)
    lock = None
    try:
        if wait_s > 0:
            try:
                verdict = device_mutex.wait_until_idle(
                    wait_s,
                    slm_number=_envelope._int_option(forwarded, "--slm-number", 1),
                    cam_id=_envelope._int_option(forwarded, "--cam-id", 0),
                    cam_type=str(_envelope._option(forwarded, "--camera", "daheng")),
                    own_paths=[Path(__file__).resolve()],
                )
            except Exception as exc:
                doc["device_wait"] = {
                    "budget_s": float(wait_s),
                    "status": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                print(
                    "ERROR: device idle wait failed, replay not started: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1

            doc["device_wait"] = _envelope._wait_record(verdict, wait_s)
            if not bool(getattr(verdict, "ok", False)):
                print(
                    "ERROR: device unavailable "
                    f"({getattr(verdict, 'kind', None) or getattr(verdict, 'status', None)}): "
                    f"{_envelope._verdict_headline(verdict)} -- "
                    f"{getattr(verdict, 'remedy', '')}; replay not started",
                    file=sys.stderr,
                )
                return 1
            lock = getattr(verdict, "lock", None)
        else:
            doc["device_wait"] = _envelope._disabled_wait_record()

        if wait_s > 0 and lock is None:
            try:
                lock = device_mutex.DeviceLock.default()
                acquired = bool(lock.acquire())
            except Exception as exc:
                doc["lock"]["acquire_error"] = f"{type(exc).__name__}: {exc}"
                print(
                    "ERROR: bench lock acquire failed, replay not started: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
            if not acquired:
                doc["lock"]["path"] = str(getattr(lock, "path", ""))
                print(
                    "ERROR: bench lock is held, replay not started "
                    f"({getattr(lock, 'path', '')})",
                    file=sys.stderr,
                )
                return 1

        if lock is not None:
            doc["lock"]["path"] = str(getattr(lock, "path", ""))
            doc["lock"]["acquired_utc"] = _envelope._utc_now()
            doc["lock"]["held"] = True

        doc["replay_invoked"] = True
        code = _invoke_replay(forwarded)
        doc["replay_exit_code"] = code
        return code
    finally:
        _envelope._release_lock(lock, doc["lock"])
        _envelope._write_evidence(out, doc)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )

    if not math.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        print(
            "ERROR: --wait-until-idle must be a finite non-negative number",
            file=sys.stderr,
        )
        return 2
    if args.help:
        return _print_help(parser)
    if not args.execute:
        print(
            "ERROR: replay always opens the SLM and camera, so --execute is "
            "required; --help is the only device-free invocation. Nothing was "
            "waited on, locked or imported.",
            file=sys.stderr,
        )
        return 2
    if _envelope._out_path(forwarded) is None:
        print(
            "ERROR: --out is required: the device-lock evidence needs a run "
            "directory that does not exist yet (replay never reuses one).",
            file=sys.stderr,
        )
        return 2
    if not _has_option(forwarded, "--exposures"):
        print(
            "ERROR: --exposures must be given explicitly: the module default "
            "(50/100 ms) is a historical sweep, and the acceptance plan allows "
            "only exposures taken from the current camera range on site.",
            file=sys.stderr,
        )
        return 2
    return _run_execute(float(args.wait_until_idle), forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
