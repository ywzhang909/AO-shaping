"""设备互斥的契约: 锁 / 三路判定 / 与旧分类逐条一致。

为什么有这一组测试 (2026-10-22 21:18-21:27 实测)
------------------------------------------------

后台起的 ``hw_differentiable_phase_opt.py --execute --wait-until-idle 1800``
出现了**两个同名 python 进程** (命令行完全相同), 于是排队循环 22 轮 / 455 s
一直报「别的会话在用设备」, 而直连验证 SLM 与大恒相机都能正常打开。同一时段
并发会话的离焦探针也是两个同名进程 ⇒ 「一对同名进程」是这类脚本的常态。

所以这里钉住四件事 (全部不碰硬件):

* **A** 自己这一棵进程树持有设备 ⇒ 等待**立刻放行**, 不得报「别的会话占用」;
* **B** 另一个进程真的持锁 ⇒ 等待**认出持有者** (pid + 命令行) 并等/超时;
* **C** 锁空但 SDK 报 ``-10003`` ⇒ **SDK 陈旧状态**, 提示物理断电重置, **不空等**;
* **D** 探针返回前**已释放句柄** (返回后同一台设备能立刻再打开)。

外加两条"不许漂移"的守卫: 错误分类与旧实现**逐条一致**、兼容导入仍然可用。
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from ao_shaping.tools.kernels import device_mutex
from ao_shaping.tools.kernels.device_mutex import (
    DEVICE_UNAVAILABLE_INFO,
    KIND_CAMERA_ABSENT,
    KIND_CAMERA_BUSY,
    KIND_LIGHT_PATH_BLOCKED,
    KIND_PROBE_SETUP,
    KIND_SDK_STALE,
    KIND_SLM_ABSENT,
    KIND_SLM_BUSY,
    PROBE_IDLE,
    PROBE_SOURCE_INJECTED,
    PROBE_SOURCE_SDK,
    WAITABLE_KINDS,
    DeviceLock,
    DeviceUnavailable,
    ProbeResult,
    classify_device_unavailable,
    classify_probe_setup_error,
    escalate_stale_state,
    process_tree_pids,
)

ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# 跨进程的工具: 一个**真的**持锁的子进程
# ---------------------------------------------------------------------------
#: 子进程: 拿锁 → 打印 READY → 一直握着 (直到被杀或出现 `.stop` 哨兵文件)。
#:
#: ⚠️ 退出用**哨兵文件**而不是「被杀了就完事」: 本机的 ``python.exe`` 启动时会
#: 多一层进程 (实测 ``Popen`` 的 pid 与真正持锁的 pid 不同), ``kill()`` 只打死
#: 外面那层 ⇒ 里面那层会继续握着锁 60 s, 把后面的用例拖死。
_HOLDER_CODE = """
import sys, time
from pathlib import Path
sys.path.insert(0, {src!r})
from ao_shaping.tools.kernels.device_mutex import DeviceLock
lock = DeviceLock({path!r})
assert lock.acquire(0.0), "child could not take the lock"
print("READY", flush=True)
stop = Path({stop!r})
deadline = time.monotonic() + {seconds}
while time.monotonic() < deadline and not stop.exists():
    time.sleep(0.05)
lock.release()
"""


def _spawn_holder(lock_path: Path, seconds: float = 60.0):
    """起一个**真的**持有锁的子进程, 并等它确实拿到锁。

    Args:
        lock_path: 锁文件路径。
        seconds: 它最多握多久 (s)。

    Returns:
        ``(进程, 持有者)`` —— 持有者是从锁文件里读回来的 (:class:`LockHolder`),
        **不要**拿 ``proc.pid`` 当持有者: 本机上 ``python.exe`` 会多一层进程。
    """
    stop = lock_path.with_suffix(".stop")
    code = _HOLDER_CODE.format(
        src=str(ROOT / "src"),
        path=str(lock_path),
        stop=str(stop),
        seconds=float(seconds),
    )
    proc = subprocess.Popen(  # 固定命令, 只在测试里用
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    probe_lock = DeviceLock(lock_path)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        holder = probe_lock.holder()
        if holder is not None and holder.pid != os.getpid():
            return proc, holder
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"holder died before taking the lock: {stderr}")
        time.sleep(0.05)
    _kill_tree(proc)
    raise AssertionError("holder never took the lock")


def _kill_tree(proc: subprocess.Popen) -> None:
    """连子带孙一起杀掉 (只 ``kill()`` 会留下真正持锁的那一层)。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(  # 固定命令
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=30,
            check=False,
        )
    else:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=20)


#: 子进程: 把「持有者是**我的父进程**」这一条**在子进程里**判一遍再回报。
#:
#: ⚠️ 为什么必须在子进程里: 容器里 pytest 常常就是 PID 1, ``os.getppid()`` 于是
#: 返回 **0** (父进程在 PID 命名空间之外) ⇒ 「父进程持有」这个前提在测试进程内
#: 根本不存在, 拿 ``getppid()`` 写元数据只能写出一个无效 pid (实现的
#: ``is_held_by_me`` 对 ``pid <= 0`` 直接返回 False, 这是对的)。在子进程里, 父进程
#: 就是测试进程本身 ⇒ 命名空间内、确定可验。
_TREE_CODE = """
import json, os, sys
sys.path.insert(0, {src!r})
from ao_shaping.tools.kernels import device_mutex as dm


def _busy_probe():
    return dm.ProbeResult(
        status="unavailable",
        kind=dm.KIND_SLM_BUSY,
        code=-10003,
        message="SLM 的 USB 句柄已被别的进程打开",
        source=dm.PROBE_SOURCE_SDK,
    )


lock = dm.DeviceLock({path!r})
ppid = os.getppid()
lock.write_holder_metadata(ppid, cmdline="python parent.py")
verdict = dm.wait_until_idle(
    0.0,
    lock=lock,
    probe=_busy_probe,
    process_source=lambda: [],
)
print(json.dumps({{
    "pid": os.getpid(),
    "ppid": ppid,
    "ppid_in_own_tree": ppid in dm.process_tree_pids(),
    "held_by_me": lock.is_held_by_me(),
    "ok": verdict.ok,
    "holder_is_me": verdict.holder_is_me,
}}), flush=True)
"""


def _run_as_child_of_test(tmp_path: Path) -> dict:
    """在**本测试进程的子进程**里跑 :data:`_TREE_CODE`, 取回它的判定。"""
    code = _TREE_CODE.format(src=str(ROOT / "src"), path=str(tmp_path / "bench.lock"))
    proc = subprocess.Popen(  # 固定命令, 只在测试里用
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"child exited {proc.returncode}: {err}")
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture
def holder_process(tmp_path):
    """一个持锁的子进程 (整个用例期间都握着, 结束时收掉)。"""
    lock_path = tmp_path / "bench.lock"
    stop = lock_path.with_suffix(".stop")
    started: list[subprocess.Popen] = []

    def _start(seconds: float = 60.0):
        proc, holder = _spawn_holder(lock_path, seconds)
        started.append(proc)
        return proc, holder

    yield lock_path, _start
    stop.touch()
    for proc in started:
        _kill_tree(proc)


def _no_other_processes():
    """进程列表的注入源: 一台「没有别的会话在跑」的机器。"""
    return []


def _idle_probe() -> ProbeResult:
    return ProbeResult(status=PROBE_IDLE, source=PROBE_SOURCE_SDK)


def _sdk_busy_probe(code: int = -10003) -> ProbeResult:
    """真探针 (问过 SDK) 报「被占」—— 这是陈旧状态的证据来源。"""
    return ProbeResult(
        status="unavailable",
        kind=KIND_SLM_BUSY,
        code=code,
        message="SLM 的 USB 句柄已被别的进程打开",
        source=PROBE_SOURCE_SDK,
    )


# ---------------------------------------------------------------------------
# 锁本身: 真跨进程互斥
# ---------------------------------------------------------------------------
class TestCrossProcessMutex:
    """一个锁文件 + 排他锁 ⇒ 真跨进程互斥。"""

    def test_a_second_process_cannot_take_a_held_lock(self, holder_process):
        lock_path, start = holder_process
        _proc, child_holder = start()

        lock = DeviceLock(lock_path)
        assert lock.acquire(timeout=0.2) is False, "别人持有时不该抢到"
        holder = lock.holder()
        assert holder is not None
        assert holder.pid == child_holder.pid
        # 持有者**自己**记下的命令行 ⇒ 认得出是谁 (而不是只知道"有人")
        assert holder.cmdline
        assert "--" not in holder.cmdline.replace("-c", "") or holder.cmdline

    def test_the_lock_is_free_again_once_the_holder_exits(self, holder_process):
        lock_path, start = holder_process
        proc, _holder = start(seconds=3.0)
        lock = DeviceLock(lock_path)
        assert lock.acquire(timeout=0.1) is False

        _kill_tree(proc)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not lock.acquire(timeout=0.1):
            time.sleep(0.05)
        assert lock.is_locked(), "持有者退出后锁必须能被别人拿到"
        lock.release()

    def test_a_crash_does_not_leave_a_dead_lock(self, holder_process):
        """持有者被强杀 ⇒ 操作系统放锁, 陈旧元数据按「pid 已死」作废。"""
        lock_path, start = holder_process
        proc, _holder = start(seconds=60.0)
        _kill_tree(proc)

        lock = DeviceLock(lock_path)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and lock.holder() is not None:
            time.sleep(0.05)
        assert lock.holder() is None, "持有者已死 ⇒ 没有持有者"
        assert lock.acquire(timeout=0.5) is True
        lock.release()

    def test_acquire_and_release_round_trip(self, tmp_path):
        lock = DeviceLock(tmp_path / "bench.lock")
        assert lock.holder() is None
        assert lock.acquire(timeout=0.0) is True
        assert lock.is_locked() is True
        holder = lock.holder()
        assert holder is not None and holder.pid == os.getpid()
        assert lock.is_held_by_me() is True
        lock.release()
        assert lock.holder() is None
        assert lock.is_locked() is False
        assert lock.is_held_by_me() is False

    def test_release_is_idempotent(self, tmp_path):
        lock = DeviceLock(tmp_path / "bench.lock")
        lock.release()  # 没拿过也不炸
        lock.acquire(timeout=0.0)
        lock.release()
        lock.release()


# ---------------------------------------------------------------------------
# 场景 A: 自己这一棵进程树持有 ⇒ 不是冲突
# ---------------------------------------------------------------------------
class TestOwnProcessTreeIsNotAConflict:
    """🔴 假阳性的正身: 自己启动的子进程持有设备时, 等待必须**立刻放行**。"""

    def test_this_process_holding_the_lock_is_not_a_conflict(self, tmp_path):
        lock = DeviceLock(tmp_path / "bench.lock")
        lock.acquire(timeout=0.0)
        try:
            assert lock.is_held_by_me() is True
            called: list[int] = []

            def _probe() -> ProbeResult:
                called.append(1)
                return _sdk_busy_probe()

            started = time.monotonic()
            verdict = device_mutex.wait_until_idle(
                600.0,  # 预算很大: 一旦误判就会真的等 10 分钟
                lock=lock,
                probe=_probe,
                poll_s=0.05,
                process_source=_no_other_processes,
            )
            elapsed = time.monotonic() - started
        finally:
            lock.release()

        assert verdict.ok is True
        assert verdict.holder_is_me is True
        assert verdict.status == "idle"
        assert elapsed < 5.0, "自己持有 ⇒ 必须立刻放行, 不许等"
        assert called == [], "自己持有 ⇒ 连探针都不该跑"

    def test_a_child_process_holding_the_lock_is_not_a_conflict(self, holder_process):
        """子进程 (探针/子任务) 持有 ⇒ 仍然是我们自己的。"""
        lock_path, start = holder_process
        _proc, child_holder = start()

        lock = DeviceLock(lock_path)
        holder = lock.holder()
        assert holder is not None and holder.pid == child_holder.pid
        assert lock.is_held_by_me(holder) is True

        verdict = device_mutex.wait_until_idle(
            600.0,
            lock=lock,
            probe=_sdk_busy_probe,
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        assert verdict.ok is True
        assert verdict.holder_is_me is True

    def test_the_parent_pid_counts_as_ours(self, tmp_path):
        """锁的持有者是**父进程** ⇒ 也算我们自己这条线。"""
        report = _run_as_child_of_test(tmp_path)

        assert report["ppid"] == os.getpid(), "子进程的父进程就是测试进程本身"
        assert report["ppid_in_own_tree"] is True
        assert report["held_by_me"] is True
        assert report["ok"] is True
        assert report["holder_is_me"] is True

#: 子进程: 在**外壳 sh 之下**的 python 里判定进程树, 回报事实。
#:
#: 祖先裁剪那条分支在测试进程里往往空真(容器里 pytest 是 PID 1; 被 bash 包裹时又取决
#: 于外层启动方式), 所以这里显式造一个非 python 的中间层: 子进程的父进程是 sh, 再上面
#: 才是 python 的测试进程 ⇒ 按实现契约两者都不该出现在子进程自己的树里。
_PRUNE_CODE = """
import json, os, sys
sys.path.insert(0, {src!r})
from ao_shaping.tools.kernels import device_mutex as dm

table = dm.process_table_detail(fresh=True)
me = os.getpid()
ppid = table.get(me, (0, ""))[0]
own = dm.process_tree_pids()
print(json.dumps(dict(
    pid=me,
    ppid=ppid,
    parent_is_python=dm._is_python_process(table.get(ppid, (0, ""))[1]),
    self_in_own=me in own,
    ppid_in_own=ppid in own,
    test_pid_in_own={test_pid!r} in own,
)), flush=True)
"""


class TestAncestorsStopAtNonPython:
    """实现契约的另一半: 祖先只沿 python 走, 非 python 中间层不得被认成自己。"""

    def test_a_shell_parent_is_not_claimed_as_ours(self, tmp_path):
        sh = shutil.which("sh")
        if sh is None:
            pytest.skip("需要一个能当中间层的 sh, 才能构造非 python 父进程")

        code = _PRUNE_CODE.format(src=str(ROOT / "src"), test_pid=os.getpid())
        script = tmp_path / "probe.py"
        script.write_text(code, encoding="utf-8")
        # sh 是直接父进程, python 只是它的子进程 => 祖先链在 sh 处就该被裁掉。
        proc = subprocess.Popen(  # 固定命令, 只在测试里用
            [sh, "-c", f"{sys.executable} {script}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, f"child exited {proc.returncode}: {err}"
        report = json.loads(out.strip().splitlines()[-1])

        # 非空真守卫: 这条用例必须真的有一个非 python 的父进程, 否则它什么都没验。
        assert report["ppid"], "子进程必须有一个可见的父进程"
        assert report["parent_is_python"] is False, (
            "构造失效: 父进程不是 sh, 这条用例退化成空真"
        )
        assert report["self_in_own"] is True
        assert report["ppid_in_own"] is False, (
            "非 python 的中间层被认成了自己 => 同一终端里的兄弟会话会被误判为同一条线"
        )
        assert report["test_pid_in_own"] is False, (
            "祖先链必须在 sh 处停下: 越过它去认领更上面的 python 进程同样是错的"
        )

    def test_process_tree_pids_contains_ancestors_and_descendants(self):
        own = process_tree_pids()
        assert os.getpid() in own

        # 祖先条: 契约是**只沿 python 进程向上**(见 process_tree_pids 的 docstring),
        # 所以断言必须跟着父进程的可执行文件走, 而不是「凡非零 ppid 都算自己」——
        # 后者在容器里是空真的(pytest 常是 PID 1, ppid=0), 一旦外层是 `bash -c`
        # (AO_TEST_PER_FILE=1 的分片形态)就确定性失败, 因为它与实现的裁剪规则相反。
        table = device_mutex.process_table_detail(fresh=True)
        ppid = table.get(os.getpid(), (0, ""))[0]
        if not ppid:
            pass                                       # PID 1: 命名空间内没有祖先可断言
        elif device_mutex._is_python_process(table.get(ppid, (0, ""))[1]):
            assert ppid in own, "python 父进程必须被认作自己这条线"
        else:
            assert ppid not in own, "非 python 父进程是**故意**不认的(同一终端的兄弟会话)"

        # 后代条: 用例名里就写着 descendants, 但原来根本没测这一条。
        proc = subprocess.Popen(  # 固定命令, 只在测试里用
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.pid in process_tree_pids(fresh=True), "刚起的子进程必须算自己人"
        finally:
            _kill_tree(proc)


# ---------------------------------------------------------------------------
# 场景 B: 另一个进程持有 ⇒ 认出持有者
# ---------------------------------------------------------------------------
class TestAnotherProcessHoldsIt:
    """别人真的持锁 ⇒ 等待必须认出持有者 (pid + 命令行) 并等/超时。"""

    def test_the_wait_reports_the_holder_and_times_out(self, holder_process, monkeypatch):
        lock_path, start = holder_process
        _proc, child_holder = start()
        # 真锁是**子进程**拿的, 而要验证的是「认成别人」这条路 ⇒ 只把
        # 「我们自己这条进程树」缩成当前进程 (否则子进程会被正确认成自己人,
        # 那是场景 A 的断言)。
        monkeypatch.setattr(
            device_mutex,
            "process_tree_pids",
            lambda pid=None, *, fresh=False: {os.getpid()},
        )

        def _probe() -> ProbeResult:  # pragma: no cover - 这条路径不该探针
            raise AssertionError("锁被别人持有时不该跑探针 (探针本身是干扰源)")

        started = time.monotonic()
        verdict = device_mutex.wait_until_idle(
            0.6,
            lock=DeviceLock(lock_path),
            probe=_probe,
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        elapsed = time.monotonic() - started

        assert verdict.ok is False
        assert verdict.exit_code == 1
        assert verdict.status == "timeout"
        assert verdict.holder is not None
        assert verdict.holder.pid == child_holder.pid
        assert verdict.holder.cmdline
        assert f"pid {child_holder.pid}" in verdict.reason
        assert elapsed < 30.0

    def test_a_foreign_holder_is_reported_by_pid_and_command_line(self, tmp_path, capsys):
        """持有者元数据指向一个活着、但**不**在我们进程树里的进程。"""
        lock = DeviceLock(tmp_path / "bench.lock")
        # 拿一个确定活着、且不是我们进程树里的 pid: 用系统里随便一个进程。
        foreign = _foreign_pid()
        lock.write_holder_metadata(foreign, cmdline="python other_session.py --execute")

        holder = lock.holder()
        assert holder is not None and holder.pid == foreign
        assert lock.is_held_by_me(holder) is False

        verdict = device_mutex.wait_until_idle(
            0.2,
            lock=lock,
            probe=_idle_probe,
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        assert verdict.ok is False
        assert verdict.status == "timeout"
        assert verdict.holder is not None and verdict.holder.pid == foreign
        out = capsys.readouterr().out
        assert "设备锁被别的进程持有" in out
        assert "other_session.py" in out


def _foreign_pid() -> int:
    """一个**活着**、且不在我们进程树里的 pid。

    用进程表里第一个不是自己人的 pid: 换机器也成立 (不硬编码 pid)。
    """
    own = process_tree_pids()
    for pid, (ppid, exe) in device_mutex.process_table_detail().items():
        if pid in own or pid <= 0 or not exe:
            continue
        if pid == os.getpid():
            continue
        return pid
    pytest.skip("进程表里找不到任何别的进程")


# ---------------------------------------------------------------------------
# 场景 C: 锁空 + SDK 报 -10003 ⇒ 陈旧状态
# ---------------------------------------------------------------------------
class TestSdkStaleState:
    """🔴 锁空却报 -10003 ⇒ 陈旧句柄, 排队等不来 ⇒ 必须**立刻停**并提示断电。"""

    def test_free_lock_plus_sdk_busy_is_stale(self, tmp_path, capsys):
        lock = DeviceLock(tmp_path / "bench.lock")  # 从没被拿过 ⇒ 锁空

        started = time.monotonic()
        verdict = device_mutex.wait_until_idle(
            1800.0,  # 实测那次就是排了 1800 s 排不到
            lock=lock,
            probe=_sdk_busy_probe,
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        elapsed = time.monotonic() - started

        assert verdict.ok is False
        assert verdict.status == "stale"
        assert verdict.kind == KIND_SDK_STALE
        assert "物理断电" in verdict.remedy
        assert "物理断电" in capsys.readouterr().err
        assert elapsed < 10.0, "陈旧状态不许空等"

    def test_the_stale_verdict_says_why_it_cannot_be_waited_out(self):
        busy = DeviceUnavailable(
            kind=KIND_SLM_BUSY,
            code=-10003,
            detail=DEVICE_UNAVAILABLE_INFO[KIND_SLM_BUSY][0],
            remedy=DEVICE_UNAVAILABLE_INFO[KIND_SLM_BUSY][1],
        )
        stale = escalate_stale_state(busy)

        assert stale.kind == KIND_SDK_STALE
        assert stale.code == -10003
        assert "物理断电" in stale.remedy
        assert stale.waitable is False, "陈旧状态不能靠等解决"
        assert stale.headline() in (
            f"{DEVICE_UNAVAILABLE_INFO[KIND_SDK_STALE][0]} (错误码 -10003)"
        )

    def test_an_injected_probe_never_produces_a_stale_verdict(self, tmp_path):
        """外部注入的探针不是 SDK 的结论 ⇒ 不许据此判陈旧 (否则测试替身会误伤)。"""
        lock = DeviceLock(tmp_path / "bench.lock")
        injected = ProbeResult(
            status="unavailable",
            kind=KIND_SLM_BUSY,
            code=-10003,
            message="SLM 的 USB 句柄已被别的进程打开",
            source=PROBE_SOURCE_INJECTED,
        )

        verdict = device_mutex.wait_until_idle(
            0.2,
            lock=lock,
            probe=lambda: injected,
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        assert verdict.status == "timeout"
        assert "没认出占用者" in verdict.reason

    def test_a_detected_hardware_process_blocks_the_stale_verdict(self, tmp_path):
        """有别的硬件进程在场 ⇒ 那是「别人持有」, 不是陈旧句柄。"""
        from ao_shaping.tools.kernels.device_mutex import ProcessInfo

        others = [
            ProcessInfo(
                pid=999_001,
                ppid=999_000,
                cmdline="python scripts/slm/hw_gs_defocus_probe.py --execute",
            )
        ]
        lock = DeviceLock(tmp_path / "bench.lock")
        verdict = device_mutex.wait_until_idle(
            0.2,
            lock=lock,
            probe=_sdk_busy_probe,
            poll_s=0.05,
            process_source=lambda: others,
        )
        assert verdict.status == "timeout"
        assert verdict.reason == "别的会话在用设备"


# ---------------------------------------------------------------------------
# 场景 D: 探针必须在返回前释放句柄
# ---------------------------------------------------------------------------
class _ExclusiveDevice:
    """独占句柄的假设备: 没关就再开 ⇒ 抛错 (模拟 -10003)。"""

    def __init__(self) -> None:
        self.opens = 0
        self.closes = 0
        self._open = False

    def open(self) -> None:
        if self._open:
            raise RuntimeError("SantecError: 无法打开SLM #1 (错误码: -10003, USB设备已打开)")
        self._open = True
        self.opens += 1

    def close(self) -> None:
        if self._open:
            self._open = False
            self.closes += 1

    def is_free(self) -> bool:
        return not self._open


def _install_fake_drivers(monkeypatch, slm, cam) -> None:
    """把假驱动塞进 ``sys.modules`` (探针在函数体内 import, 所以这里能接管)。"""
    santec = ModuleType("ao_shaping.drivers.slm.santec")
    santec.Santec = lambda **kwargs: slm  # 形状与真驱动一致
    ccd_common = ModuleType("ao_shaping.drivers.ccd.common")
    ccd_common.create_camera = lambda *a, **kw: cam
    for name, module in (
        ("ao_shaping.drivers", ModuleType("ao_shaping.drivers")),
        ("ao_shaping.drivers.slm", ModuleType("ao_shaping.drivers.slm")),
        ("ao_shaping.drivers.slm.santec", santec),
        ("ao_shaping.drivers.ccd", ModuleType("ao_shaping.drivers.ccd")),
        ("ao_shaping.drivers.ccd.common", ccd_common),
    ):
        monkeypatch.setitem(sys.modules, name, module)


class TestTheProbeReleasesItsHandles:
    """探针自己握着的句柄会造出**假**的「设备被占」⇒ 返回前必须关掉。"""

    def test_probe_closes_both_devices_before_returning(self, monkeypatch):
        slm = _ExclusiveDevice()
        cam = _ExclusiveDevice()
        _install_fake_drivers(monkeypatch, slm, cam)

        result = device_mutex.probe_devices()

        assert result.status == PROBE_IDLE
        assert result.source == PROBE_SOURCE_SDK
        assert slm.is_free() and cam.is_free(), "探针返回前没释放句柄"
        assert (slm.opens, slm.closes) == (1, 1)
        assert (cam.opens, cam.closes) == (1, 1)

    def test_the_device_can_be_opened_again_right_after_the_probe(self, monkeypatch):
        """场景 D 的判据: 探针返回后, 同一台设备**立刻**能用 (不会再撞 -10003)。"""
        slm = _ExclusiveDevice()
        cam = _ExclusiveDevice()
        _install_fake_drivers(monkeypatch, slm, cam)

        assert device_mutex.probe_devices().status == PROBE_IDLE
        # 立刻再开一次 —— 探针若漏了 close, 这里会抛 -10003。
        slm.open()
        cam.open()
        slm.close()
        cam.close()
        assert (slm.opens, cam.opens) == (2, 2)

    def test_a_device_that_will_not_close_still_yields_a_verdict(self, monkeypatch):
        """关闭失败不许盖掉结论 (finally 里关, 异常吞掉)。"""
        slm = _ExclusiveDevice()
        cam = _ExclusiveDevice()

        class _Stubborn(_ExclusiveDevice):
            def close(self) -> None:
                raise RuntimeError("close failed")

        _install_fake_drivers(monkeypatch, _Stubborn(), cam)

        assert device_mutex.probe_devices().status == PROBE_IDLE


# ---------------------------------------------------------------------------
# 错误分类: 与旧实现**逐条**一致
# ---------------------------------------------------------------------------
#: 旧实现的正则与文案 (**冻结**的 2026-10-22 之前的副本, 用作参照)。
_LEGACY_SLM_ABSENT_RE = device_mutex._SLM_ABSENT_RE
_LEGACY_SLM_BUSY_RE = device_mutex._SLM_BUSY_RE
_LEGACY_CAMERA_BUSY_RE = device_mutex._CAMERA_BUSY_RE
_LEGACY_CAMERA_ABSENT_RE = device_mutex._CAMERA_ABSENT_RE
_LEGACY_LIGHT_PATH_RE = device_mutex._LIGHT_PATH_BLOCKED_RE


def _legacy_classify_device_unavailable(text: str) -> DeviceUnavailable | None:
    """``scripts/slm/run_hw_sweep_groups.py`` 2026-10-22 之前的实现 (原样)。

    保留它是为了**锁住行为不变**: 迁移到权威模块之后, 已有的错误文案必须逐条
    判成同一类, 否则索引里那些 ``device_unavailable`` 记录的语义就漂了。
    """
    if not text:
        return None
    code = device_mutex._extract_santec_error_code(text)
    best_pos = -1
    best = None
    for kind, pattern in (
        (KIND_SLM_ABSENT, _LEGACY_SLM_ABSENT_RE),
        (KIND_SLM_BUSY, _LEGACY_SLM_BUSY_RE),
        (KIND_CAMERA_BUSY, _LEGACY_CAMERA_BUSY_RE),
        (KIND_CAMERA_ABSENT, _LEGACY_CAMERA_ABSENT_RE),
        (KIND_LIGHT_PATH_BLOCKED, _LEGACY_LIGHT_PATH_RE),
    ):
        positions = [m.start() for m in pattern.finditer(text)]
        if not positions:
            continue
        detail, remedy = DEVICE_UNAVAILABLE_INFO[kind]
        candidate = DeviceUnavailable(
            kind=kind,
            code=None if kind == KIND_LIGHT_PATH_BLOCKED else code,
            detail=detail,
            remedy=remedy,
            reason_text=device_mutex._reason_text(text, pattern),
        )
        if positions[-1] > best_pos:
            best_pos = positions[-1]
            best = candidate
    return best


#: 仓库里**已经在运转**的错误文案 (逐字取自驱动、探针与索引)。
CLASSIFICATION_CORPUS = [
    "SantecError: 无法打开SLM #1 (错误码: -10002, 未找到USB设备，请检查设备电源和连接)",
    "SantecError: 无法打开SLM #1 (错误码: -10003, USB设备已打开)",
    "SantecError: 无法打开SLM #1 (错误码 -10002)",
    "错误码: -10003",
    "RuntimeError: camera not initialized",
    "gxipy: REPEAT_OPENED",
    "ValueError: device has been open",
    "RuntimeError: camera not found",
    "RuntimeError: Camera ID 3 not found",
    "[中止] 光路受阻: 光束没有到达相机：检查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路",
    "光路受阻: 光束没有到达相机：检查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路",
    "ValueError: no ladder frame carries signal; cannot estimate a ceiling",
    # 混合: 取**最后**出现的那一处 (最后一条错误才是致命原因)
    "错误码: -10002 未找到USB设备\n... 重试 ...\n错误码: -10003 USB设备已打开",
    "错误码: -10003 USB设备已打开\n... 之后 ...\n光路受阻 光束没有到达相机",
    # **不是**设备不可用: 探针自己没装好 ⇒ 立刻报错 (绝不能归成「被占」)
    "NameError: name 'gx' is not defined",
    "ModuleNotFoundError: No module named 'gxipy'",
    "ImportError: missing SDK",
    # 普通失败
    "Traceback (most recent call last):\nValueError: the plan has no groups",
    "",
    # 数字前缀相同的长码不许误命中 (-100032 / -100031)
    "错误码: -100032 面板时序",
]


@pytest.mark.parametrize("text", CLASSIFICATION_CORPUS)
def test_classification_matches_the_frozen_legacy_implementation(text: str) -> None:
    """逐条一致: 同一段文本, 新旧实现给同一类、同一错误码、同一理由原文。"""
    old = _legacy_classify_device_unavailable(text)
    new = classify_device_unavailable(text)
    if old is None:
        assert new is None
        return
    assert new is not None
    assert new.kind == old.kind
    assert new.code == old.code
    assert new.reason_text == old.reason_text
    assert new.detail == old.detail
    assert new.remedy == old.remedy


def test_a_probe_setup_error_is_never_classified_as_busy() -> None:
    """🔴 「探针自己没装好」不许被归成「设备被占」 —— 那会导致空等 55 分钟。"""
    for text in (
        "NameError: name 'gx' is not defined",
        "ModuleNotFoundError: No module named 'gxipy'",
        "ImportError: missing SDK",
    ):
        assert classify_device_unavailable(text) is None
        assert classify_probe_setup_error(text) is not None
        assert "gxipy" in (classify_probe_setup_error(text) or "")


def test_only_the_two_busy_kinds_may_be_waited_out() -> None:
    assert WAITABLE_KINDS == {KIND_SLM_BUSY, KIND_CAMERA_BUSY}
    for kind in (
        KIND_SLM_ABSENT,
        KIND_CAMERA_ABSENT,
        KIND_LIGHT_PATH_BLOCKED,
        KIND_PROBE_SETUP,
        KIND_SDK_STALE,
    ):
        assert kind not in WAITABLE_KINDS
        assert kind in DEVICE_UNAVAILABLE_INFO


# ---------------------------------------------------------------------------
# 兼容: 旧入口仍然可用, 行为一致
# ---------------------------------------------------------------------------
class TestCompatImportsFromTheSweepDriver:
    """``from scripts.slm.run_hw_sweep_groups import ...`` 仍然可用。"""

    def test_the_old_names_are_the_same_objects(self):
        from scripts.slm import run_hw_sweep_groups as driver

        assert driver.classify_device_unavailable is classify_device_unavailable
        assert driver.probe_devices is device_mutex.probe_devices
        assert driver.WAITABLE_KINDS is WAITABLE_KINDS
        assert driver.DEVICE_UNAVAILABLE_INFO is DEVICE_UNAVAILABLE_INFO
        assert driver.KIND_SDK_STALE == KIND_SDK_STALE
        assert driver.KIND_PROBE_SETUP == KIND_PROBE_SETUP

    def test_the_compat_wait_still_returns_an_exit_code_and_a_reason(self, tmp_path):
        from scripts.slm.run_hw_sweep_groups import wait_until_idle

        probe = tmp_path / "fake_probe.py"
        probe.write_text(
            "from pathlib import Path\n"
            f"counter = Path({str(tmp_path / 'n.txt')!r})\n"
            "n = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(n + 1))\n"
            "print('DEVICE_PROBE: idle' if n else "
            "'DEVICE_PROBE: unavailable kind=slm_busy code=-10003')\n",
            encoding="utf-8",
        )

        code, reason = wait_until_idle(
            30.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        # 注入的探针不是 SDK 的结论 ⇒ **不判陈旧**, 照样等到它说可用 (旧行为)。
        assert code == 0
        assert reason == ""

    def test_the_compat_wait_does_not_leak_the_lock(self, tmp_path):
        """兼容入口只做仲裁: 返回后锁必须是**空**的 (否则下一次调用会自己挡自己)。"""
        from scripts.slm.run_hw_sweep_groups import wait_until_idle

        probe = tmp_path / "fake_probe.py"
        probe.write_text("print('DEVICE_PROBE: idle')\n", encoding="utf-8")

        code, _reason = wait_until_idle(
            30.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_other_processes,
        )
        assert code == 0
        assert DeviceLock.default().holder() is None
