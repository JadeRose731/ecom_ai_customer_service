"""ch10 验收 API:验收页的唯一数据源。
口径铁律:页面上的数与终端 make 跑出来的是同一份产物——这里只读 data/ch10/ 下的
落盘 json 与文件 stat,不做任何重算。字节数一律 1024 进制,与前端 fmtBytes 同口径。
闸门三态 pass / fail / missing:「没跑过」≠「没过线」,产物缺失时 note 不写失败态文案;
mysql 读不到时主题分布那一项单独降级,不连坐整页。"""
import json
import pathlib
from collections import Counter
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException

from app.core import jobs
from app.core.taxonomy import TOPIC_NAMES
from app.db import repository

router = APIRouter()

DATA_DIR = pathlib.Path("data/ch10")
REPORTS = DATA_DIR / "reports"
DATASET = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "model"
ONNX = DATA_DIR / "onnx"
SERVICE = "http://127.0.0.1:8110"

RECIPES = {   # 判错方向的处置建议(与「效果不行先搞数据」的口径一致)
    "漏打": "预标漏打该类——回查 PRELABEL_PROMPT 该类的边界表述与示例",
    "多打": "预标脑补了近邻类——收紧 prompt 标注铁律,不改黄金样例凑分",
    "错位": "近邻类目摩擦——优先改术语表边界与示例,再看要不要补语料",
}


# ---------- 通用小件 ----------
def _load_report(fname: str, make: str) -> dict:
    p = REPORTS / fname
    if not p.exists():
        return {"present": False, "make": make,
                "hint": f"还没跑过——点按钮或终端 make {make} 生成产物"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # 作业被中途 stop 会留下残缺 json(脚本 write_text 非原子)——按没跑过处理,不炸页面
        return {"present": False, "make": make,
                "hint": f"产物损坏(可能被中途停止)——重跑 make {make} 即可覆盖"}
    data["present"] = True
    data["make"] = make
    return data


def _stat(p: pathlib.Path) -> dict | None:
    if not p.exists():
        return None
    st = p.stat()
    out = {"name": p.name, "bytes": st.st_size,
           "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if p.suffix in (".jsonl", ".md"):   # 条数只对这两类算:config/tokenizer 算行数是噪声
        with p.open(encoding="utf-8") as fh:
            out["lines"] = sum(1 for line in fh if line.strip())
    return out


def _dir_stats(d: pathlib.Path, names: list[str]) -> list[dict]:
    out = []
    for n in names:
        s = _stat(d / n)
        if s:
            out.append(s)
    return out


async def _probe_service() -> dict:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{SERVICE}/healthz")
        return {"online": r.status_code == 200}
    except Exception:
        return {"online": False}


def gate(present: bool, passed: bool | None) -> str:
    if not present:
        return "missing"
    return "pass" if passed else "fail"


def _asynb(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


# ---------- 各端点 ----------
@router.get("/api/acceptance/overview")
async def overview():
    golden = _load_report("golden_report.json", "ch10-golden")
    eval_r = _load_report("eval_report.json", "ch10-eval")
    scan = _load_report("threshold_scan.json", "ch10-threshold-scan")
    export_r = _load_report("export_report.json", "ch10-export")
    pool = _load_report("classify_run.json", "classify-pool")
    svc = await _probe_service()

    corpus_labeled = _stat(DATA_DIR / "corpus_labeled.jsonl")
    sample_review = _stat(DATA_DIR / "sample_review.md")
    splits = _split_stats()
    model_files = _dir_stats(MODEL_DIR, ["model.safetensors", "tokenizer.json", "threshold.json"])

    blocks = [
        {"key": "golden", "no": 1, "title": "预标质量闸", "page": "/acceptance/data",
         "status": gate(golden["present"], golden.get("passed")),
         "headline": (f"通过率 {golden['rate']:.0%},闸线 {golden['pass_line']:.0%}"
                      if golden["present"] else "黄金样例还没跑"),
         "note": "" if golden["present"] else golden["hint"],
         "jobs": ["ch10-golden"]},
        {"key": "corpus", "no": 2, "title": "语料流水线产物", "page": "/acceptance/data",
         "status": gate(corpus_labeled is not None and sample_review is not None, True),
         "headline": (f"语料 {corpus_labeled['lines']} 条,抽审文件 {sample_review['lines']} 行"
                      if corpus_labeled and sample_review else "语料产物还没生成"),
         "note": "" if corpus_labeled and sample_review else
                 f"还没跑过——点按钮或终端 make ch10-corpus 生成产物",
         "jobs": ["ch10-corpus"]},
        {"key": "dataset", "no": 3, "title": "考卷划分与泄漏", "page": "/acceptance/data",
         "status": (gate(splits["ready"], splits["clean"] == True)  # noqa: E712
                    if splits["ready"] else "missing"),
         "headline": (f"三卷共 {sum(s['lines'] for s in splits['splits'].values())} 条,"
                      f"两两重叠 {sum(splits['leaks'].values())} 条"
                      if splits["ready"] else "考卷还没生成"),
         "note": ("" if not splits["ready"] or splits["clean"] else
                  "考卷互相泄漏,回 build_dataset 查划分") if splits["ready"] else
                 "还没跑过——点按钮或终端 make ch10-dataset 生成产物",
         "jobs": ["ch10-dataset"]},
        {"key": "train", "no": 4, "title": "训练三件套", "page": "/acceptance/data",
         "status": gate(len(model_files) == 3, True),
         "headline": (f"权重+tokenizer+阈值齐({_asnb(sum(f['bytes'] for f in model_files))})"
                      if len(model_files) == 3 else "还没训出模型"),
         "note": "" if len(model_files) == 3 else "还没跑过——点按钮或终端 make ch10-train",
         "jobs": ["ch10-train"]},
        {"key": "eval", "no": 5, "title": "测试集红线", "page": "/acceptance/eval",
         "status": gate(eval_r["present"], eval_r.get("red_line_passed")),
         "headline": (f"micro F1 {eval_r['micro']['f1']:.3f},严中档红线"
                      f"{'全过' if eval_r.get('red_line_passed') else '有未达标'}"
                      if eval_r["present"] else "评测还没跑"),
         "note": "" if eval_r["present"] else eval_r["hint"],
         "jobs": ["ch10-eval"]},
        {"key": "export", "no": 6, "title": "ONNX 导出一致", "page": "/acceptance/eval",
         "status": gate(export_r["present"], export_r.get("passed")),
         "headline": (f"{export_r['checked']} 条校验,不一致 {export_r['mismatch']} 条,"
                      f"文件 {_asnb(export_r.get('onnx_bytes', 0))}"
                      if export_r["present"] else "还没导出"),
         "note": "" if export_r["present"] else export_r["hint"],
         "jobs": ["ch10-export"]},
        {"key": "classifier", "no": 7, "title": "推理服务 :8110", "page": "/acceptance/eval",
         "status": "pass" if svc["online"] else "missing",
         "headline": "在线,可单句试分类" if svc["online"] else "服务未拉起(未拉起 ≠ 服务坏)",
         "note": "" if svc["online"] else "点「拉起推理服务」再试",
         "jobs": ["classifier-up", "classifier-down"]},
        {"key": "scan", "no": 8, "title": "阈值可复算", "page": "/acceptance/eval",
         "status": gate(scan["present"], scan.get("consistent")),
         "headline": (f"重演最优线 {scan['best_threshold']} == 在用线 {scan['in_use_threshold']}"
                      if scan["present"] else "重演还没跑"),
         "note": ("" if not scan["present"] or scan.get("consistent") else
                  "重演与在用线对不上——查训练扫描口径") if scan["present"] else
                 "还没跑过——需先拉起分类器",
         "jobs": ["ch10-threshold-scan"]},
        {"key": "pool", "no": 9, "title": "旁路归类落库", "page": "/topics",
         "status": gate(pool["present"], True),
         "headline": ({"done": f"本轮写入 {pool.get('written', 0)} 条",
                       "empty": "池里暂无待归类(幂等实证)",
                       "below_batch": f"待归类 {pool.get('pending', 0)} 条,攒批中"}.get(
                           pool.get("status"), "归类链路已跑") if pool["present"] else "还没跑过"),
         "note": "" if pool["present"] else pool["hint"],
         "jobs": ["classify-pool", "classify-pool-force"]},
    ]
    passed = sum(1 for b in blocks if b["status"] == "pass")
    return {"blocks": blocks, "passed": passed, "total": len(blocks),
            "all_pass": passed == len(blocks), "classifier": svc,
            "jobs": {name: {"title": s.target, "needs": s.needs, "heavy": s.heavy,
                            "status": jobs.status(name)["status"]}
                     for name, s in jobs.JOBS.items()}}


@router.get("/api/acceptance/eval")
async def eval_detail():
    eval_r = _load_report("eval_report.json", "ch10-eval")
    scan = _load_report("threshold_scan.json", "ch10-threshold-scan")
    threshold_in_use = None
    tf = ONNX / "threshold.json"
    if tf.exists():
        threshold_in_use = json.loads(tf.read_text(encoding="utf-8"))
    return {"eval": eval_r, "scan": scan, "threshold_in_use": threshold_in_use,
            "severity": eval_r.get("red_lines") if eval_r["present"] else None,
            "classifier": await _probe_service()}


@router.get("/api/acceptance/data")
async def data_detail():
    corpus = {n: _stat(DATA_DIR / f"corpus_{n}.jsonl") for n in ("raw", "clean", "labeled")}
    review = _stat(DATA_DIR / "sample_review.md")
    splits = _split_stats()
    model_threshold = _read_json(MODEL_DIR / "threshold.json")
    onnx_threshold = _read_json(ONNX / "threshold.json")
    model_present = len(_dir_stats(MODEL_DIR, ["model.safetensors", "tokenizer.json"])) == 2
    onnx_present = len(_dir_stats(ONNX, ["model.onnx", "tokenizer.json"])) == 2
    try:
        dist = await repository.topic_distribution()
        dist["present"] = True
    except Exception:
        dist = {"present": False, "hint": "mysql 不可达,主题分布暂缺(其他项不受连坐)"}
    return {"lineage": corpus, "sample_review": review, "dataset": splits,
            "model": {"present": model_present,
                      "files": _dir_stats(MODEL_DIR, ["model.safetensors", "tokenizer.json",
                                                      "threshold.json"]),
                      "threshold": model_threshold},
            "onnx": {"present": onnx_present,
                     "files": _dir_stats(ONNX, ["model.onnx", "tokenizer.json", "threshold.json"]),
                     "threshold": onnx_threshold},
            "topics": dist, "topic_names": list(TOPIC_NAMES)}


@router.get("/api/acceptance/errors")
async def errors_detail():
    eval_r = _load_report("eval_report.json", "ch10-eval")
    if not eval_r["present"]:
        return {"eval": eval_r, "errors": [], "kinds": {}, "pairs": {},
                "recipes": RECIPES, "matrix_entries": 0, "total_fp": 0, "total_fn": 0}
    errors = eval_r.get("errors", [])
    kinds = dict(Counter(e["kind"] for e in errors))
    pairs: dict[str, int] = Counter()
    for e in errors:
        for m in e["missed"]:
            for x in e["extra"]:
                pairs[f"{m}→{x}"] += 1
    return {"eval": {k: eval_r[k] for k in ("test_size", "threshold", "micro", "macro",
                                            "red_line_passed")},
            "errors": errors, "kinds": kinds, "pairs": dict(pairs), "recipes": RECIPES,
            "matrix_entries": eval_r.get("total_cells", 0),
            "total_fp": eval_r.get("total_fp", 0), "total_fn": eval_r.get("total_fn", 0)}


@router.get("/api/acceptance/service")
async def service_detail():
    return {"classifier": await _probe_service(), "onnx": _stat(ONNX / "model.onnx"),
            "threshold_in_use": _read_json(ONNX / "threshold.json")}


@router.post("/api/acceptance/classify")
async def classify_one(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{SERVICE}/classify", json={"texts": [text]})
            r.raise_for_status()
    except Exception:
        raise HTTPException(status_code=503, detail="分类器服务不在线,先点「拉起推理服务」")
    return {"result": r.json()["results"][0],
            "threshold_in_use": _read_json(ONNX / "threshold.json")}


# ---------- 内部 ----------
def _read_json(p: pathlib.Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _split_stats() -> dict:
    """三份考卷的行数/类目分布 + 两两文本重叠(泄漏硬闸)。只读产物,不重算划分。"""
    splits: dict[str, dict] = {}
    texts: dict[str, set[str]] = {}
    dists: dict[str, Counter] = {}
    for name in ("train", "val", "test"):
        p = DATASET / f"{name}.jsonl"
        if not p.exists():
            splits[name] = {"present": False, "lines": 0, "dist": {}}
            texts[name] = set()
            dists[name] = Counter()
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        texts[name] = {r["text"] for r in rows}
        dists[name] = Counter(lb for r in rows for lb in r["labels"])
        splits[name] = {"present": True, "lines": len(rows), "dist": dict(dists[name])}
    leaks = {"train×val": len(texts["train"] & texts["val"]),
             "train×test": len(texts["train"] & texts["test"]),
             "val×test": len(texts["val"] & texts["test"])}
    ready = all(splits[n]["present"] for n in ("train", "val", "test"))
    return {"splits": splits, "leaks": leaks, "ready": ready,
            "clean": None if not ready else sum(leaks.values()) == 0}
