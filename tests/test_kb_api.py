# ch03 Task 17:录入 API 四条口径
TABLE_MD = "\n".join(
    ["| 商品 | 价格 |", "| --- | --- |"]
    + [f"| 商品{i} | {i} |" for i in range(1, 16)]
)


async def test_preview_is_dry_run(client, db_session_factory, db_clean):
    """预览只切不写:按完预览再看库存,一块都不该多。"""
    r = await client.post("/api/kb/preview", json={
        "content_type": "policy", "content": "# 政策\n\n## 运费\n\n满99包邮。"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert body["blocks"][0]["questions"] == "运费"
    ov = (await client.get("/api/kb/overview")).json()
    assert ov["stats"] is None or ov["stats"]["total"] == 0  # 一块都没多


async def test_ingest_then_reingest_is_idempotent(client, db_session_factory, db_clean):
    """同一份正文重录:第二次 inserted=0、skipped=全部,库里总数不变。"""
    payload = {"content_type": "faq", "content": "# FAQ\n\n## 邮费多少\n\n满99包邮。"}
    r1 = await client.post("/api/kb/ingest", json=payload)
    assert r1.status_code == 200
    assert r1.json()["inserted"] >= 1
    r2 = await client.post("/api/kb/ingest", json=payload)
    assert r2.status_code == 200
    assert r2.json()["inserted"] == 0
    assert r2.json()["skipped"] == r1.json()["inserted"]
    ov = (await client.get("/api/kb/overview")).json()
    assert ov["stats"]["total"] == r1.json()["inserted"]


async def test_ingest_keeps_multiple_pieces_of_one_section(client, db_session_factory, db_clean):
    """大表格按行拆出的多块共用节标题——只按问法查重会误杀,这里锁住不许。"""
    r = await client.post("/api/kb/ingest", json={
        "content_type": "manual", "content": f"# 手册\n\n## 价目表\n\n{TABLE_MD}"})
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] >= 2, "同节多块都该入库"
    assert body["skipped"] == 0


async def test_preview_rejects_bad_input(client, db_session_factory, db_clean):
    """空正文 / 非法 content_type / 不在材料清单里的文件名(含路径穿越)一律 400。"""
    assert (await client.post("/api/kb/preview", json={
        "content_type": "faq", "content": "   "})).status_code == 400
    assert (await client.post("/api/kb/preview", json={
        "content_type": "no-such-type", "content": "正文"})).status_code == 400
    assert (await client.post("/api/kb/preview", json={
        "content_type": "faq", "file": "../../.env"})).status_code == 400
    assert (await client.post("/api/kb/preview", json={
        "content_type": "faq", "file": "not-in-list.md"})).status_code == 400
