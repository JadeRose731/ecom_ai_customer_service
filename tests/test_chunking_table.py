from app.kb import chunking


TABLE = "\n".join([
    "| 商品 | 价格 |",
    "| --- | --- |",
    "| A | 1 |",
    "| B | 2 |",
    "| C | 3 |",
])


def test_is_table_block():
    assert chunking.is_table_block(TABLE)
    assert not chunking.is_table_block("普通段落文字。")


def test_split_table_replicates_header():
    out = chunking.split_table_rows(TABLE, max_rows=2)
    assert len(out) == 2
    assert out[0] == "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    assert out[1] == "| 商品 | 价格 |\n| --- | --- |\n| C | 3 |"
    for block in out:
        assert block.startswith("| 商品 | 价格 |\n| --- | --- |")  # 每块都带表头


def test_split_table_small_stays_whole():
    assert chunking.split_table_rows(TABLE, max_rows=10) == [TABLE]
