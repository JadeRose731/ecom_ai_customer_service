"""ch10 数据集:分层划分 80/10/10 + 训练集增强(同义词替换/句式微调)。
增强只扩训练集——验证/测试是考题,不许照练习题变。运行:make ch10-dataset(需上游可用)。"""
import asyncio
import json
import os
import pathlib
import sys

from app.core.llm import get_chat_model
from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.corpus_lib import dedupe, split_dataset

SRC = pathlib.Path("data/ch10/corpus_labeled.jsonl")
OUT = pathlib.Path("data/ch10/dataset")

AUGMENT_PROMPT = """把下面这句电商客服用户问题改写一个变体:换同义词、微调句式(比如改成「我想问一下……」的口气),
不改原意、不增删诉求。只输出改写后的句子。

{text}"""


async def augment(samples: list[dict], concurrency: int = 8) -> list[dict]:
    model = get_chat_model()
    sem = asyncio.Semaphore(concurrency)

    async def one(s: dict) -> dict | None:
        async with sem:
            try:
                r = await model.ainvoke(AUGMENT_PROMPT.format(text=s["text"]))
                t = r.content.strip()
            except Exception:
                return None
        return {"text": t, "labels": s["labels"], "origin": "augmented"} if t else None

    outs = await asyncio.gather(*[one(s) for s in samples])
    return [o for o in outs if o]


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text("\n".join(
        json.dumps({"text": s["text"], "labels": s["labels"]}, ensure_ascii=False)
        for s in samples), encoding="utf-8")


def _dist(name: str, samples: list[dict]) -> None:
    counts = {n: 0 for n in TOPIC_NAMES}
    for s in samples:
        for lb in s["labels"]:
            counts[lb] += 1
    print(f"{name}({len(samples)} 条): " + " ".join(f"{k}={v}" for k, v in counts.items()))


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK 兜底
        sys.stdout.reconfigure(errors="replace")
    samples = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, val, test = split_dataset(samples)
    if os.environ.get("CH10_NO_AUG") == "1":
        aug = []   # 离线代跑(CH10_NO_AUG=1):同义句增强要 LLM,上游挂时逐条重试会卡几小时,直接跳过
    else:
        aug = await augment(train)
    aug = dedupe(aug)                        # 变体之间也去重
    seen = {s["text"] for s in samples}      # 变体撞上任何原句(含考题)就丢弃
    train = train + [a for a in aug if a["text"] not in seen]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ds in (("train", train), ("val", val), ("test", test)):
        _dump(OUT / f"{name}.jsonl", ds)
        _dist(name, ds)


if __name__ == "__main__":
    asyncio.run(main())
