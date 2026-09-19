# app/core/jobs.py
"""作业运行器:只能跑注册表里的 make 目标,argv 写死,前端只传作业名——没有 shell。
命令一律走 make:页面按的与终端敲的是同一条配方。日志覆盖写 log/acceptance/<job>.log。
平台差异:POSIX 用进程组收整棵树(make → uv → python);Windows 无 make 也无进程组,
两件事都降级——start() 代读 Makefile 把配方直译成本地命令(环境前缀转 env、
$(or $(X,d)) 读环境变量、$(if $(FORCE),--force,) 看 spec.force、`< 文件` 转 stdin),
stop() 用 taskkill /T 杀树;直译不了的配方(裸 shell 语法)明着报错,不硬猜。
ch10 起:JobSpec 带 needs(按钮上直接提示前置条件);人为 kill 的非零退出码记
stopped,不冤枉成 failed;classifier-up/down 的配方是壳脚本(nohup/sleep/curl/kill),
由运行器原生托管(起 uv serve 轮 healthz / 按 pid 文件杀树)。
内存态重启归零;日志 mtime 与产物 ran_at 在盘上,页面照样说得出「上次什么时候跑的」。"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

_LOG_DIR = pathlib.Path("log/acceptance")
_TAIL_LINES = 60
ROOT = pathlib.Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
PID_FILE = ROOT / "data" / "classifier.pid"
HEALTHZ = "http://127.0.0.1:8110/healthz"
WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str       # Makefile 目标
    heavy: bool = False   # heavy 作业页面二次确认才发起
    needs: str = ""       # 前置条件,页面按钮上直接提示(ch10 起)
    force: bool = False   # 带 FORCE=1 跑同一条配方(ch10:classify-pool-force)

    @property
    def argv(self) -> tuple[str, ...]:
        return ("make", self.target) + (("FORCE=1",) if self.force else ())


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
        # ch09:观测与成本页三作业(成本账需 Langfuse 在跑;后两个 minutes 级重活)
        JobSpec("cost-report", "cost-report"),
        JobSpec("eval-flywheel", "eval-flywheel", heavy=True),
        JobSpec("calibrate-confidence", "calibrate-confidence", heavy=True),
        # ch10:主题分类器全链(产物落 data/ch10/,验收页就地重跑的就是这批)
        JobSpec("ch10-golden", "ch10-golden", needs="需聊天上游可用(.env CHAT_*)"),
        JobSpec("ch10-corpus", "ch10-corpus", heavy=True,
                needs="需上游可用;自动先过黄金闸"),
        JobSpec("ch10-dataset", "ch10-dataset", needs="需上游可用与语料产物"),
        JobSpec("ch10-train", "ch10-train", heavy=True,
                needs="需考卷产物;CPU 上是分钟级重活"),
        JobSpec("ch10-eval", "ch10-eval", needs="需已训练模型与测试考卷"),
        JobSpec("ch10-export", "ch10-export", needs="需已训练模型"),
        JobSpec("ch10-threshold-scan", "ch10-threshold-scan",
                needs="需分类器服务 :8110 在线"),
        JobSpec("classifier-up", "classifier-up", needs="需 ONNX 产物"),
        JobSpec("classifier-down", "classifier-down", needs="服务在跑才有得停"),
        JobSpec("classify-pool", "classify-pool",
                needs="需 :8110 在线;池子攒够一批(min-batch)才跑"),
        JobSpec("classify-pool-force", "classify-pool", force=True,
                needs="需 :8110 在线;不足一批也强跑"),
    )
}

# 存活进程表(按作业名;同名没跑完拒绝重入)——老接缝,测试在此打桩
_procs: dict[str, object] = {}


@dataclass
class _Meta:
    stopped: bool = False
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None


# 簿记表:人为停止标记 / 退出码 / 起止时间(内存态,重启归零)
_meta: dict[str, _Meta] = {}


class JobNotFound(KeyError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


class RecipeNotTranslatable(RuntimeError):
    pass


def _log_path(name: str) -> pathlib.Path:
    return _LOG_DIR / f"{name}.log"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def start(name: str) -> None:
    spec = JOBS.get(name)
    if spec is None:
        raise JobNotFound(name)
    if status(name)["status"] == "running":
        raise JobAlreadyRunning(name)
    if name not in ("classifier-up", "classifier-down"):
        steps = _resolve_steps(spec)   # 直译不了就在记账前抛,不留假 running
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    meta = _Meta(started_at=_now())
    _meta[name] = meta
    log = open(_log_path(name), "w", encoding="utf-8")  # 覆盖写:每次重跑看最新
    log.write(f"# argv: {' '.join(spec.argv)}\n# 发起: {meta.started_at}\n")
    log.flush()

    if name == "classifier-up":          # 壳脚本配方,运行器原生托管
        _finish(meta, _serve_up(log))
        return
    if name == "classifier-down":
        _finish(meta, _serve_down(log))
        return

    def work() -> None:
        rc = 0
        try:
            for step in steps:
                if meta.stopped:
                    break
                log.write(f"\n$ {' '.join(step['cmd'])}"
                          + (f"  < {step['stdin']}" if step["stdin"] else "") + "\n")
                log.flush()
                proc = _popen(step, log)
                _procs[name] = proc
                rc = proc.wait()
                if rc != 0:
                    break
        except Exception as e:           # 线程里炸了也要给终态,不能卡死 running
            log.write(f"\n运行器异常:{e!r}\n")
            rc = -1
        finally:
            _finish(meta, rc)

    threading.Thread(target=work, name=f"job-{name}", daemon=True).start()


def stop(name: str) -> None:
    if name not in JOBS:
        raise JobNotFound(name)
    proc = _procs.get(name)
    meta = _meta.get(name)
    if proc is None and meta is None:
        return                            # 从没跑过,无事可停
    meta = meta or _Meta()
    _meta[name] = meta
    meta.stopped = True   # 人为停的记 stopped,不按非零退出码冤枉成 failed
    if proc is not None and proc.poll() is None:
        pid = getattr(proc, "pid", None)
        if pid:
            _stop_proc(pid)


def _finish(meta: _Meta, rc: int) -> None:
    meta.returncode = rc
    meta.finished_at = _now()


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
    meta = _meta.get(name)
    running = ((proc is not None and proc.poll() is None)
               or (meta is not None and meta.started_at is not None
                   and meta.returncode is None))   # start 与 work 之间的窗口也算在跑
    if meta is not None and meta.stopped:
        st = "stopped"
    elif running:
        st = "running"
    elif meta is not None and meta.returncode is not None:
        st = "pass" if meta.returncode == 0 else "fail"
    else:
        st = "idle"
    return {
        "name": name,
        "target": spec.target,
        "heavy": spec.heavy,
        "needs": spec.needs,
        "running": running,
        "status": st,
        "started_at": meta.started_at if meta else None,
        "finished_at": meta.finished_at if meta else None,
        "log_tail": _tail(name),
    }


def all_status() -> dict[str, dict]:
    return {name: status(name) for name in JOBS}


# ---------- 进程接缝(测试在此打桩) ----------
def _popen(step: dict, log) -> subprocess.Popen:
    stdin = open(step["stdin"], "rb") if step["stdin"] else subprocess.DEVNULL
    return subprocess.Popen(
        step["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        stdin=stdin, env={**os.environ, **step["env"]},
        start_new_session=not WINDOWS,   # POSIX:独立会话,停止时 killpg 整树收
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
    )


def _stop_proc(pid: int) -> None:
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, AttributeError):
            pass


# ---------- Makefile 直译(无 make 环境的代敲;支持本仓库用到的三种构造) ----------
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_OR_RE = re.compile(r"\$\(or \$\((\w+)\),([^)]*)\)")


def _parse_target(target: str) -> tuple[list[str], list[str]]:
    deps: list[str] = []
    recipes: list[str] = []
    current: str | None = None
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("\t"):
            if current == target:
                recipes.append(line.strip())
            continue
        if not line or line.startswith("#"):
            current = None
            continue
        m = re.match(r"^([A-Za-z0-9_.%-]+)\s*:\s*(.*)$", line)
        if not m:
            current = None
            continue
        current = m.group(1)
        if current == target:
            deps = m.group(2).split("##", 1)[0].split()   # 目标行尾可能带注释
    if not deps and not recipes:
        raise JobNotFound(f"Makefile 里没有目标 {target}")
    return deps, recipes


def _resolve_steps(spec: JobSpec) -> list[dict]:
    """产出顺序步骤 [{env, cmd, stdin}]。有 make 用 make(原口径);没有则直译配方。"""
    if shutil.which("make"):
        return [{"env": {}, "cmd": list(spec.argv), "stdin": None}]
    steps: list[dict] = []

    def walk(target: str, chain: tuple[str, ...]) -> None:
        if target in chain:
            raise RecipeNotTranslatable(f"make 依赖成环:{'→'.join(chain + (target,))}")
        deps, recipes = _parse_target(target)
        for d in deps:
            walk(d, chain + (target,))
        for raw in recipes:
            line = raw.replace("$(if $(FORCE),--force,)", "--force" if spec.force else "")
            line = _OR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2)), line)
            if "$(" in line:
                raise RecipeNotTranslatable(
                    f"配方含未支持的 make 语法,需在装有 make 的环境跑:{target}")
            tokens = line.split()
            env: dict[str, str] = {}
            while tokens and _ENV_RE.match(tokens[0]):
                k, _, v = tokens.pop(0).partition("=")
                env[k] = v
            stdin = None
            if "<" in tokens:
                i = tokens.index("<")
                stdin = os.path.join(ROOT, tokens[i + 1])
                tokens = tokens[:i] + tokens[i + 2:]
            steps.append({"env": env, "cmd": tokens, "stdin": stdin})

    walk(spec.target, ())
    return steps


# ---------- classifier 起停原生托管(make 配方是壳脚本:nohup/sleep/curl/kill) ----------
def _serve_up(log) -> int:
    log.write("$ uv run --group ml python scripts/ch10/serve.py\n")
    log.flush()
    proc = subprocess.Popen(
        ["uv", "run", "--group", "ml", "python", "scripts/ch10/serve.py"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": "."},
        start_new_session=not WINDOWS,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    for _ in range(60):                       # 最多 ~15s 等 healthz
        try:
            with urllib.request.urlopen(HEALTHZ, timeout=2) as r:
                if r.status == 200:
                    log.write(f"healthz ok,pid {proc.pid}\n")
                    return 0
        except Exception:
            time.sleep(0.25)
    log.write("healthz 超时,服务没起来(看本日志上方输出)\n")
    return 1


def _serve_down(log) -> int:
    if not PID_FILE.exists():
        log.write("没有 pid 文件,服务本来就没起\n")
        return 0
    pid = int(PID_FILE.read_text().strip())
    log.write(f"$ 停止分类器(pid {pid},杀树)\n")
    _stop_proc(pid)
    PID_FILE.unlink(missing_ok=True)
    log.write("已停止\n")
    return 0
