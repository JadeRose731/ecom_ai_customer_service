# ch03 Task 17:作业运行器的安全边界
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
