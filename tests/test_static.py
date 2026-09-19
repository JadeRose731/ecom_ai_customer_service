# tests/test_static.py · 页面可达性断言(页面本体按 spec §10 在浏览器里验收)
async def test_kb_page_reachable_and_mounts_nav(client):
    r = await client.get("/kb")
    assert r.status_code == 200
    assert "/static/admin.js" in r.text and "/static/acceptance.js" in r.text


async def test_admin_page_reachable_and_mounts_nav(client):
    r = await client.get("/admin")
    assert r.status_code == 200
    assert "/static/admin.js" in r.text


async def test_observability_page_reachable_and_mounts_nav(client):
    r = await client.get("/observability")
    assert r.status_code == 200
    assert "/static/admin.js" in r.text and "/static/acceptance.js" in r.text


async def test_index_has_admin_entry(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert 'href="/admin"' in r.text or "location.href='/admin'" in r.text


async def test_feedback_bar_lock_declared(client):
    # 回归:ch09 曾把 `let locked = false;` 改丢——点击 👍/👎 抛 ReferenceError,fetch 根本不发起,
    # .catch 静默吞掉,反馈入口整条死且无任何可见症状(node --check 查不出未声明引用,只能字符串钉)
    r = await client.get("/")
    assert "let locked = false" in r.text


async def test_static_assets_served(client):
    for path in ("/static/admin.js", "/static/acceptance.js", "/static/acceptance.css"):
        assert (await client.get(path)).status_code == 200, path
