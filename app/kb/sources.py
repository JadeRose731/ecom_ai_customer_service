# app/kb/sources.py
# 知识库源材料的唯一清单:build_kb / 预览 / 录入 API 都从这里读,不各留一份。
import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "kb"

# 文件 → content_type
SOURCE_TYPES = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
}

# 录入页可选的 content_type(mined 是挖知识内部产物,不对外)
CONTENT_TYPES = ("faq", "policy", "manual")
