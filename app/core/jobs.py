# app/core/jobs.py
"""作业运行器:只能跑注册表里的 make 目标,argv 写死,前端只传作业名——没有 shell。
命令一律走 make:页面按的与终端敲的是同一条配方。日志覆盖写 log/acceptance/<job>.log。
平台差异:POSIX 用进程组收整棵树(make → uv → python);Windows 无进程组,降级 proc.terminate()。"""
import os
import pathlib
import signal
import subprocess
from dataclasses import dataclass

_LOG_DIR = pathlib.Path("log/acceptance")
_TAIL_LINES = 60


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str       # Makefile 目标
    heavy: bool = False   # heavy 作业页面二次确认才发起


JOBS: dict[str, JobSpec] = {
    spec.name: spec
    for spec in (
        JobSpec("kb-preview", "kb-preview"),
        JobSpec("kb-build", "kb-build"),
        JobSpec("kb-vectorize", "kb-vectorize"),
        JobSpec("kb-mine", "kb-mine", heavy=True),
        JobSpec("kb-reset", "kb-reset", heavy=True),
        JobSpec("seed-conv", "seed-conv"),
        JobSpec("eval-rag", "eval-rag", heavy=True),   # ch04:四策略评估,分钟级重活
    )
}

# 存活进程表(按作业名;同名没跑完拒绝重入)
_procs: dict[str, subprocess.Popen] = {}


class JobNotFound(KeyError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


def _log_path(name: str) -> pathlib.Path:
    return _LOG_DIR / f"{name}.log"


def start(name: str) -> None:
    spec = JOBS.get(name)
    if spec is None:
        raise JobNotFound(name)
    proc = _procs.get(name)
    if proc is not None and proc.poll() is None:
        raise JobAlreadyRunning(name)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(_log_path(name), "w", encoding="utf-8")  # 覆盖写:每次重跑看最新
    _procs[name] = subprocess.Popen(
        ["make", spec.target],
        stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # 独立会话,停止时 killpg 连 make→uv→python 一起收
    )


def stop(name: str) -> None:
    if name not in JOBS:
        raise JobNotFound(name)
    proc = _procs.get(name)
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, AttributeError):
        proc.terminate()  # Windows 或进程组已不在


def _tail(name: str) -> str:
    p = _log_path(name)
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-_TAIL_LINES:])


def status(name: str) -> dict:
    spec = JOBS.get(name)
    if spec is None:
        raise JobNotFound(name)
    proc = _procs.get(name)
    running = proc is not None and proc.poll() is None
    return {
        "name": name,
        "target": spec.target,
        "heavy": spec.heavy,
        "running": running,
        "log_tail": _tail(name),
    }


def all_status() -> dict[str, dict]:
    return {name: status(name) for name in JOBS}
