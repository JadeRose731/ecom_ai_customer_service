"""作业运行器:注册表边界 / make 口径 / idle 初态 / 同名冲突 / 停止语义。
ch10 扩展(needs/FORCE/stopped)与 ch03 老边界共用这一份。"""
import threading

import pytest

from app.core import jobs


@pytest.fixture
def clean_runs():
    jobs._procs.clear()
    jobs._meta.clear()
    yield
    jobs._procs.clear()
    jobs._meta.clear()


class _HeldProc:
    """wait() 挂住不放,模拟长作业;release() 后以给定退出码收尾。"""

    def __init__(self, rc: int = 0):
        self.pid = 424242
        self._rc = rc
        self._ev = threading.Event()

    def release(self):
        self._ev.set()

    def wait(self):
        self._ev.wait(timeout=5)
        return self._rc

    def poll(self):
        return None if not self._ev.is_set() else self._rc


def _drain(name: str) -> None:
    for _ in range(100):                 # 等工作线程把 rc 记完
        if jobs._meta.get(name) and jobs._meta[name].returncode is not None:
            return
        threading.Event().wait(0.05)


def _await_proc(name: str) -> None:
    for _ in range(100):                 # 等工作线程把进程登记进 _procs
        if jobs._procs.get(name) is not None:
            return
        threading.Event().wait(0.02)


def test_jobs_outside_whitelist_absent():
    assert "rm -rf /" not in jobs.JOBS
    assert "ch10-nonexistent" not in jobs.JOBS


def test_registry_is_make_typed_and_targets_exist():
    # 页面只传作业名;argv 一律 make 口径,目标必须真在 Makefile 里(按目标行匹配)
    text = jobs.MAKEFILE.read_text(encoding="utf-8")
    targets = set()
    for line in text.splitlines():
        if line and not line[0].isspace() and ":" in line and not line.startswith("#"):
            targets.add(line.split(":", 1)[0].strip())
    assert len(jobs.JOBS) == 21          # ch03/ch04/ch09 十个 + ch10 十一个
    for spec in jobs.JOBS.values():
        assert spec.argv[0] == "make", spec.name
        assert spec.argv[1] in targets, spec.name


def test_status_idle_for_unrun(clean_runs):
    assert jobs.status("ch10-train")["status"] == "idle"
    assert jobs.status("kb-build")["running"] is False


def test_start_twice_raises(clean_runs, monkeypatch):
    held = _HeldProc()
    monkeypatch.setattr(jobs, "_popen", lambda step, log: held)
    monkeypatch.setattr(jobs, "_stop_proc", lambda pid: held.release())
    jobs.start("ch10-eval")
    assert jobs.status("ch10-eval")["status"] == "running"
    with pytest.raises(jobs.JobAlreadyRunning):
        jobs.start("ch10-eval")
    jobs.stop("ch10-eval")
    assert jobs.status("ch10-eval")["status"] == "stopped"


def test_stopped_not_rewritten_to_failed(clean_runs, monkeypatch):
    # 人为 kill 的退出码也是非 0:stopped 就是 stopped,不冤枉成 failed
    held = _HeldProc(rc=1)
    monkeypatch.setattr(jobs, "_popen", lambda step, log: held)
    monkeypatch.setattr(jobs, "_stop_proc", lambda pid: held.release())
    jobs.start("ch10-export")
    _await_proc("ch10-export")           # 进程登记后再停,才能杀到 wait 里的真退出码
    jobs.stop("ch10-export")             # 先置 stopped 标记再放行 wait
    _drain("ch10-export")
    assert jobs._meta["ch10-export"].returncode == 1
    assert jobs.status("ch10-export")["status"] == "stopped"


def test_pass_fail_by_exit_code(clean_runs, monkeypatch):
    held = _HeldProc(rc=0)
    monkeypatch.setattr(jobs, "_popen", lambda step, log: held)
    monkeypatch.setattr(jobs, "_stop_proc", lambda pid: held.release())
    jobs.start("ch10-export")
    held.release()
    _drain("ch10-export")
    assert jobs.status("ch10-export")["status"] == "pass"


def test_stop_after_finish_keeps_terminal_state(clean_runs, monkeypatch):
    # 跑完的作业再点「停止」不该把它从 pass 改写成 stopped
    held = _HeldProc(rc=0)
    monkeypatch.setattr(jobs, "_popen", lambda step, log: held)
    monkeypatch.setattr(jobs, "_stop_proc", lambda pid: held.release())
    jobs.start("ch10-export")
    held.release()
    _drain("ch10-export")
    jobs.stop("ch10-export")
    assert jobs.status("ch10-export")["status"] == "pass"


def test_force_variant_registered():
    # 「不足一批强跑」走同一条 make 配方:FORCE=1 只是参数,不复制命令
    spec = jobs.JOBS["classify-pool-force"]
    assert spec.argv == ("make", "classify-pool", "FORCE=1")
    assert jobs.JOBS["classify-pool"].argv == ("make", "classify-pool")


def test_resolve_translates_force_and_env_fallback(monkeypatch):
    # 无 make 环境的直译:FORCE 展开、$(or $(X,d)) 落 env 缺省、PYTHONPATH= 转 env
    monkeypatch.setattr(jobs.shutil, "which", lambda name: None)
    monkeypatch.delenv("DAYS", raising=False)
    monkeypatch.delenv("TRIGGER", raising=False)
    steps = jobs._resolve_steps(jobs.JOBS["classify-pool-force"])
    assert steps[0]["cmd"] == ["uv", "run", "python",
                               "scripts/ch10/classify_pool.py", "--force"]
    assert steps[0]["env"] == {"PYTHONPATH": "."}
    cost = jobs._resolve_steps(jobs.JOBS["cost-report"])
    assert cost[0]["cmd"][-2:] == ["--days", "7"]
    monkeypatch.setenv("DAYS", "30")
    assert jobs._resolve_steps(jobs.JOBS["cost-report"])[0]["cmd"][-2:] == ["--days", "30"]


def test_resolve_walks_dependencies(monkeypatch):
    # ch10-corpus 依赖 ch10-golden:直译时依赖的配方排在前面
    monkeypatch.setattr(jobs.shutil, "which", lambda name: None)
    steps = jobs._resolve_steps(jobs.JOBS["ch10-corpus"])
    scripts = [s["cmd"][-1] for s in steps]
    assert scripts == ["scripts/ch10/validate_golden.py", "scripts/ch10/build_corpus.py"]
