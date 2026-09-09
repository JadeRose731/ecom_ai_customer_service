# tests/conftest.py
import pathlib

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]


def _create_table_stmts() -> list[str]:
    stmts: list[str] = []
    for ddl in _DDL_FILES:
        raw = ddl.read_text(encoding="utf-8")
        sql = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("--"))
        stmts += [s.strip() for s in sql.split(";") if s.strip() and "CREATE TABLE" in s.upper()]
    return stmts

@pytest_asyncio.fixture(scope="session")
async def _test_engine():
    # 用 server-url(不带库名)建 mewhelp_test 库
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text("DROP DATABASE IF EXISTS mewhelp_test"))
        await conn.execute(text("CREATE DATABASE mewhelp_test CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(以 sql/ch0*-ddl.sql 为权威;按 ; 分割逐句执行)
    async with engine.begin() as conn:
        for s in _create_table_stmts():
            await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest.fixture()
def milvus():
    """真实向量库(Docker Milvus Standalone,Lite 无 Windows 包):每用例 drop/重建集合。"""
    from app.kb import milvus_client as mc

    c = mc.get_client()
    if c.has_collection(mc.COLLECTION):
        c.drop_collection(mc.COLLECTION)
    mc.ensure_collection(c)
    yield c
    if c.has_collection(mc.COLLECTION):
        c.drop_collection(mc.COLLECTION)


@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, monkeypatch):
    """把 repository 用到的 async_session 指向测试库。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory

@pytest_asyncio.fixture()
async def db_clean(_test_engine):
    """每个测试后清空 4 张表(关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))


@pytest_asyncio.fixture()
async def client():
    """直接打 FastAPI 应用(ASGI 传输,不起真端口)。"""
    import httpx

    from app.main import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
