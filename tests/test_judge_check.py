# tests/test_judge_check.py
"""裁判回归工具:证据拼回逐字一致 / 老快照缺 n 按序补 / 处置→标准答案映射 /
能考个案的筛选 / 调用失败不计入一致。"""
import pytest

from scripts.judge_check import build_evidence, eligible, expected_verdict, tally


def test_evidence_rebuild_matches_online_format():
    """拼回格式与线上 _evidence_text 逐字一致:'[n] 问题: 答案' 按行拼。"""
    citations = [
        {"n": 1, "question": "退货政策是什么", "answer": "支持 7 天无理由退货"},
        {"n": 2, "question": "退款时效", "answer": "3 个工作日内原路退回"},
    ]
    assert build_evidence(citations) == (
        "[1] 退货政策是什么: 支持 7 天无理由退货\n"
        "[2] 退款时效: 3 个工作日内原路退回")


def test_legacy_snapshot_without_n_gets_sequential_numbers():
    """老快照缺 n 字段:按数组序补 1,2,3…,格式不变形。"""
    citations = [
        {"question": "甲", "answer": "a"},
        {"question": "乙", "answer": "b"},
    ]
    assert build_evidence(citations) == "[1] 甲: a\n[2] 乙: b"


def test_resolution_to_expected_verdict_mapping():
    """处置→标准答案:已解决=该判编造(False),无需解决=该判忠实(True),未解决不考。"""
    assert expected_verdict("resolved") is False
    assert expected_verdict("wontfix") is True
    assert expected_verdict("unresolved") is None


def test_eligible_filter():
    """能考的个案 = 已处置 且 有 citations 快照;未解决或没快照的跳过。"""
    assert eligible({"status": "resolved", "citations": [{"n": 1}]})
    assert eligible({"status": "wontfix", "citations": [{"n": 1}]})
    assert not eligible({"status": "unresolved", "citations": [{"n": 1}]})
    assert not eligible({"status": "resolved", "citations": []})
    assert not eligible({"status": "resolved", "citations": None})


def test_failed_calls_excluded_from_agreement():
    """调用失败(None)既不算一致也不算不一致,单列;一致率只看考过的。"""
    rows = [
        ("B7", True, True),    # 一致
        ("C2", False, False),  # 一致
        ("A9", False, None),   # 调用失败
    ]
    t = tally(rows)
    assert t["agree"] == 2 and t["disagree"] == 0 and t["failed"] == 1
    assert t["graded"] == 2 and t["rate"] == pytest.approx(1.0)


def test_mismatch_listed_and_rate_reflects_it():
    t = tally([("B7", True, False), ("C2", False, False)])
    assert t["disagree"] == 1 and t["misses"] == ["B7"]
    assert t["rate"] == pytest.approx(0.5)
