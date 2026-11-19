"""目标形状 CCD 验收的带锁外壳: 只负责串行化占用与记录证据。

``scripts/slm/validate_transfer_shapes.py`` (实现 ``ao_shaping.tools.slm.
transfer.validate``) 已经能合成目标形状、可选用 CCD 验收; 但它自己不拿共享
设备锁、也不等待别的会话。本脚本**原样转发**它的全部参数, 只多一个
``--wait-until-idle``; 没有 ``--execute`` 时完全不碰设备 — 不等待、不加锁、
不导入任何设备模块。

有 ``--execute`` 时:

1. 懒导入 ``ao_shaping.tools.kernels.device_mutex`` (只有这一个入口会导入它);
2. ``wait_until_idle`` 按 ``--camera / --cam-id / --slm-number`` 等到设备空出,
   并把持有者 pid / 命令行 / 超时细节记下来; 等不到就**不启动验证脚本**;
3. 保留等待返回的锁; 没返回就立刻自己拿同一条 bench 共享锁, 拿不到同样不启动;
4. 锁**一直握到验证脚本返回**, 在 ``finally`` 里释放 (释放失败也只记录, 不掩盖
   验证脚本的退出码);
5. 之后把 ``device_lock.json`` 写进本次 ``--out``。若验证脚本在创建目录之前就
   失败, 改写同级的 ``<out>.device_lock.json`` — 绝不创建运行目录, 也不让证据
   写入异常盖掉原始错误。

``--wait-until-idle 0`` 是仓库约定: 既不等待也不加锁 (无 shadow lock), 证据里
明确记为 disabled。本脚本不新增任何验收指标, 也不改验收算法。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VALIDATOR_SCRIPT = Path(__file__).resolve().with_name("validate_transfer_shapes.py")
DEFAULT_WAIT_S = 120.0
LOCK_EVIDENCE_NAME = "device_lock.json"
SCHEMA = "validate_transfer_shapes_hardware/device_lock/1"


def _utc_now() -> str:
    """Current UTC time in the same form the other hardware scripts record."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def build_parser() -> argparse.ArgumentParser:
    """Wrapper options only; every other token is forwarded verbatim.

    ``allow_abbrev=False`` keeps a validator option from being consumed as an
    abbreviation of ``--wait-until-idle`` (and vice versa).
    """
    parser = argparse.ArgumentParser(
        prog="validate_transfer_shapes_hardware.py",
        description=__doc__.splitlines()[0],
        add_help=False,
        allow_abbrev=False,
        epilog=(
            "All other options are forwarded verbatim to "
            "validate_transfer_shapes.py (see its --help below). Without "
            "--execute the validator runs plan/model-only and this wrapper "
            "never waits, locks or imports device modules."
        ),
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help plus the validator's options, then exit",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_S,
        metavar="SECONDS",
        help=(
            "Before the validator opens hardware (--execute only), wait up to "
            "SECONDS for the device lock / other hardware processes to clear, "
            "then hold the device_mutex lock over the whole validation "
            f"(default: {DEFAULT_WAIT_S:g}). Pass 0 to bypass both the wait "
            "and the lock (repo convention: 0 = off)."
        ),
    )
    return parser


def _validator_help_text() -> str:
    """The existing validator's own ``--help`` (all of its options, verbatim)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"could not run {VALIDATOR_SCRIPT.name} --help: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"{VALIDATOR_SCRIPT.name} --help exited {proc.returncode}: {detail}"
        )
    return proc.stdout.rstrip()


def _print_help(parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    print("\nValidator options (from validate_transfer_shapes.py, forwarded verbatim):\n")
    try:
        print(_validator_help_text())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _has_execute(argv: Sequence[str]) -> bool:
    """True when the forwarded validator argv carries ``--execute``."""
    return "--execute" in argv


def _option(argv: Sequence[str], name: str, default: Any) -> Any:
    """Last ``--name VALUE`` / ``--name=VALUE`` in ``argv`` (validator default if absent)."""
    value = default
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 < len(argv):
                value = argv[index + 1]
        elif token.startswith(f"{name}="):
            value = token.partition("=")[2]
    return value


def _int_option(argv: Sequence[str], name: str, default: int) -> int:
    """Integer form of :func:`_option`; an invalid value falls back to the default.

    The validator's own argparse rejects the same token, so a typo still fails
    loudly there; this only keeps the idle-wait probe from crashing first.
    """
    try:
        return int(_option(argv, name, default))
    except (TypeError, ValueError):
        return int(default)


def _out_path(argv: Sequence[str]) -> Path | None:
    """The validator's ``--out`` (a run directory); ``None`` when not given."""
    raw = _option(argv, "--out", None)
    return Path(raw) if raw else None


def _wait_record(verdict: Any, wait_s: float) -> dict[str, Any]:
    """Timeout/holder detail of a ``wait_until_idle`` verdict, JSON-safe."""
    holder = getattr(verdict, "holder", None)
    return {
        "budget_s": float(wait_s),
        "status": getattr(verdict, "status", None),
        "kind": getattr(verdict, "kind", None),
        "waited_s": round(float(getattr(verdict, "waited_s", 0.0) or 0.0), 3),
        "own_process_tree": bool(getattr(verdict, "holder_is_me", False)),
        "holder_pid": getattr(holder, "pid", None) if holder is not None else None,
        "holder_cmdline": (
            getattr(holder, "cmdline", None) if holder is not None else None
        ),
        "detail": getattr(verdict, "detail", ""),
        "remedy": getattr(verdict, "remedy", ""),
        "lock_returned_by_wait": getattr(verdict, "lock", None) is not None,
    }


def _verdict_headline(verdict: Any) -> str:
    headline = getattr(verdict, "headline", None)
    if callable(headline):
        return str(headline())
    return str(
        getattr(verdict, "detail", "")
        or getattr(verdict, "status", "")
        or "device unavailable"
    )


def _disabled_wait_record() -> dict[str, Any]:
    return {
        "budget_s": 0.0,
        "status": "skipped",
        "kind": None,
        "waited_s": 0.0,
        "own_process_tree": False,
        "holder_pid": None,
        "holder_cmdline": None,
        "detail": (
            "waiting disabled (--wait-until-idle 0); device lock not acquired"
        ),
        "remedy": "",
        "lock_returned_by_wait": False,
    }


def _evidence_doc(wait_s: float, argv: Sequence[str]) -> dict[str, Any]:
    """Evidence skeleton; filled in as the wait/lock/validator proceed."""
    return {
        "schema": SCHEMA,
        "script": Path(__file__).name,
        "validator": str(VALIDATOR_SCRIPT),
        "validator_argv": list(argv),
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
        "validator_invoked": False,
        "validator_exit_code": None,
        "written_utc": None,
    }


def _release_lock(lock: Any, record: dict[str, Any]) -> None:
    """Release in ``finally`` and record timestamps/duration/error; never raises."""
    acquired = record.get("acquired_utc")
    if acquired is not None:
        record["held_s"] = round(
            (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(str(acquired))
            ).total_seconds(),
            3,
        )
    if lock is None:
        # Nothing held: disabled wait (--wait-until-idle 0) or a failed acquire.
        record["released"] = None
        return
    try:
        lock.release()
        record["released"] = True
    except Exception as exc:  # a release fault must not mask the validator's code
        record["released"] = False
        record["release_error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"WARNING: device lock release failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    finally:
        record["released_utc"] = _utc_now()
        try:
            record["held"] = bool(lock.is_locked())
        except Exception:
            record["held"] = None


def _evidence_target(out: Path) -> Path:
    """``<out>/device_lock.json`` when the run directory exists, else a sibling.

    A sibling is deliberate: a validator that failed before ``mkdir`` must not
    leave an empty run directory behind (it would block a clean re-run, which
    uses ``exist_ok=False``).
    """
    if out.is_dir():
        return out / LOCK_EVIDENCE_NAME
    return out.parent / f"{out.name}.{LOCK_EVIDENCE_NAME}"


def _write_evidence(out: Path | None, doc: dict[str, Any]) -> Path | None:
    """Best-effort evidence write; a failure is reported, never raised.

    ``Exception`` (not just ``OSError``) is deliberate: a malformed field must
    not replace the validator's own exception or exit code.
    """
    if out is None:
        print(
            "WARNING: --out missing from validator args; "
            "device-lock evidence not written",
            file=sys.stderr,
        )
        return None
    target = _evidence_target(out)
    doc["written_utc"] = _utc_now()
    try:
        target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except Exception as exc:
        print(
            f"WARNING: could not write device-lock evidence {target}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    print(f"device lock evidence: {target}", file=sys.stderr)
    return target


def _invoke_validator(argv: Sequence[str]) -> int:
    """Run the existing validator with the forwarded argv; return its exit code."""
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), *argv],
        check=False,
    )
    return int(proc.returncode)


def _run_execute(wait_s: float, forwarded: Sequence[str]) -> int:
    """Wait, hold the bench lock over the validator, and record evidence."""
    # The ONLY import of device-side modules in this script: plan/model-only and
    # --help paths never reach it.
    from ao_shaping.tools.kernels import device_mutex

    out = _out_path(forwarded)
    doc = _evidence_doc(wait_s, forwarded)
    lock = None
    try:
        if wait_s > 0:
            try:
                verdict = device_mutex.wait_until_idle(
                    wait_s,
                    slm_number=_int_option(forwarded, "--slm-number", 1),
                    cam_id=_int_option(forwarded, "--cam-id", 0),
                    cam_type=str(_option(forwarded, "--camera", "daheng")),
                    own_paths=[Path(__file__).resolve(), VALIDATOR_SCRIPT],
                )
            except Exception as exc:
                doc["device_wait"] = {
                    "budget_s": float(wait_s),
                    "status": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                print(
                    "ERROR: device idle wait failed, validator not started: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1

            doc["device_wait"] = _wait_record(verdict, wait_s)
            if not bool(getattr(verdict, "ok", False)):
                print(
                    "ERROR: device unavailable "
                    f"({getattr(verdict, 'kind', None) or getattr(verdict, 'status', None)}): "
                    f"{_verdict_headline(verdict)} -- "
                    f"{getattr(verdict, 'remedy', '')}; validator not started",
                    file=sys.stderr,
                )
                return 1

            # ``wait_until_idle`` hands back the lock it already took when it
            # cleared us; otherwise take the shared bench lock ourselves.
            if getattr(verdict, "lock", None) is not None:
                lock = verdict.lock
            else:
                try:
                    lock = device_mutex.DeviceLock.default()
                    acquired = lock.acquire(timeout=0.0)
                except Exception as exc:
                    doc["lock"]["path"] = (
                        str(lock.path) if lock is not None else None
                    )
                    print(
                        "ERROR: device lock acquisition failed, validator not "
                        f"started: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    lock = None
                    return 1
                if not acquired:
                    doc["lock"]["path"] = str(lock.path)
                    print(
                        "ERROR: device lock could not be acquired right after "
                        "the idle wait; another session took it -- rerun (or "
                        "wait for it to finish). Validator not started",
                        file=sys.stderr,
                    )
                    lock = None
                    return 1
            doc["lock"]["path"] = str(lock.path)
            doc["lock"]["held"] = True
            doc["lock"]["acquired_utc"] = _utc_now()
        else:
            # No shadow lock: 0 means the operator explicitly disables both.
            doc["device_wait"] = _disabled_wait_record()

        doc["validator_invoked"] = True
        exit_code = _invoke_validator(forwarded)
        doc["validator_exit_code"] = exit_code
        return exit_code
    finally:
        _release_lock(lock, doc["lock"])
        _write_evidence(out, doc)


def main(argv: list[str] | None = None) -> int:
    """Forward to the validator, serializing hardware occupancy when needed."""
    parser = build_parser()
    ns, forwarded = parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    if not math.isfinite(ns.wait_until_idle) or ns.wait_until_idle < 0:
        parser.error("--wait-until-idle must be a finite non-negative number")
    if ns.help:
        return _print_help(parser)
    if not _has_execute(forwarded):
        print(
            "validate_transfer_shapes_hardware: no --execute "
            "(plan/model-only); device lock not used",
            file=sys.stderr,
        )
        return _invoke_validator(forwarded)
    return _run_execute(float(ns.wait_until_idle), forwarded)


if __name__ == "__main__":
    sys.exit(main())
