# ch03 Task 17:作业运行器的安全边界;ch10 追加:FORCE 变体、stopped 语义、注册表口径
import threading

from app.core import jobs


async def test_unknown_job_is_rejected(client):
    """作业名不在白名单 → 404;命令片段没有任何地方能传进来。"""
    assert (await client.post("/api/jobs/nope")).status_code == 404
    # 路径穿越变体:框架归一化后到不了作业路由,4xx 即可(关键是到不了运行器)
    r = await client.post("/api/jobs/..%2Fevil")
    assert 400 <= r.status_code < 500
    assert (await client.get("/api/jobs/nope")).status_code == 404


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业还在跑 → 409,别让两份进程抢同一批产物。"""

    class FakeProc:
        def poll(self):
            return None  # 还在跑

    monkeypatch.setitem(jobs._procs, "kb-build", FakeProc())
    r = await client.post("/api/jobs/kb-build")
    assert r.status_code == 409


def test_every_job_target_exists_in_makefile():
    """注册表里每个作业都得是 Makefile 里真有的目标,免得按钮指向不存在的配方。"""
    makefile = open("Makefile", encoding="utf-8").read()
    targets = {ln.split(":", 1)[0].strip() for ln in makefile.splitlines()
               if ln and not ln.startswith(("\t", " ", "#")) and ":" in ln}
    for spec in jobs.JOBS.values():
        assert spec.target in targets, f"Makefile 缺目标 {spec.target}"


# ---------- ch10 追加 ----------
async def test_stop_running_job_marks_stopped(client, monkeypatch):
    """stop 后状态是 stopped:人为 kill 的非零退出码不改写成 failed。"""

    class HeldProc:
        def __init__(self):
            self.pid = 7
            self._ev = threading.Event()

        def poll(self):
            return None if not self._ev.is_set() else 1

        def wait(self):
            self._ev.wait(timeout=5)
            return 1

    held = HeldProc()
    monkeypatch.setattr(jobs, "_stop_proc", lambda pid: held._ev.set())
    monkeypatch.setitem(jobs._procs, "ch10-eval", held)
    r = await client.post("/api/jobs/ch10-eval/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"
    assert r.json()["running"] is False


async def test_list_covers_full_registry(client):
    body = (await client.get("/api/jobs")).json()
    assert set(body["jobs"]) == set(jobs.JOBS)
    assert body["jobs"]["classify-pool-force"]["needs"]
    assert body["jobs"]["ch10-train"]["heavy"] is True


def test_force_variant_same_recipe():
    # 「不足一批强跑」走同一条 make 配方:FORCE=1 只是参数,不复制命令
    assert jobs.JOBS["classify-pool-force"].target == "classify-pool"
    assert jobs.JOBS["classify-pool-force"].force is True
    assert jobs.JOBS["classify-pool"].force is False
