# app/kb/dedup.py
import re

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded


def fingerprint(question: str, answer: str) -> str:
    """查重指纹 = normalize(问法) + "|" + normalize(正文)。
    只按问法会误杀同一节切出的多块(表格按行拆共用节标题)。"""
    return f"{normalize_question(question)}|{normalize_question(answer)}"
