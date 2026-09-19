import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.actions import router as actions_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.feedback import router as feedback_router
from app.api.acceptance import router as acceptance_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.admin import router as admin_router
from app.api.observability import router as observability_router
from app.api.rageval import router as rageval_router
from app.api.review import router as review_router
from app.api.topics import router as topics_router
from app.graph import runtime

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.tools import registry as tool_registry
    tool_registry.scan_builtin()   # ch08:内置工具服务启动时登记
    # ch05:起图(checkpointer 打开 + setup + 编译)。失败即启动失败,不带病服务。
    await runtime.init_graph()
    yield
    await runtime.close_graph()


app = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(actions_router)
app.include_router(conversations_router)
app.include_router(feedback_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(observability_router)
app.include_router(rageval_router)
app.include_router(review_router)
app.include_router(topics_router)
app.include_router(jobs_router)
app.include_router(acceptance_router)

# ch03 后台页面:各页保持原路径(路由必须先于根路径 StaticFiles 挂载注册,否则被吞)
_STATIC = pathlib.Path(__file__).resolve().parent / "static"


@app.get("/kb")
async def kb_page():
    return FileResponse(_STATIC / "kb.html")


@app.get("/admin")
async def admin_page():
    return FileResponse(_STATIC / "admin.html")


@app.get("/rag-eval")
async def rageval_page():
    return FileResponse(_STATIC / "rageval.html")


# ch09:审核后台页(页面本体 Task 12 建,先落路由避免占位期 404)
@app.get("/review")
async def review_page():
    return FileResponse(_STATIC / "review.html")


# ch09:观测与成本页(意图成本账/评估趋势/置信度校准三张报表)
@app.get("/observability")
async def observability_page():
    return FileResponse(_STATIC / "observability.html")


# ch10:主题分布页 + 类目问题列表页(类目与页码走地址栏,贴链接直达)
@app.get("/topics")
async def topics_page():
    return FileResponse(_STATIC / "topics.html")


@app.get("/topics/questions")
async def topic_questions_page():
    return FileResponse(_STATIC / "topic-questions.html")


# ch10:分类器验收四页(九闸总览/评测详情/数据产物/错例复核;页上按钮就地重跑作业)
@app.get("/acceptance")
async def acceptance_page():
    return FileResponse(_STATIC / "acceptance.html")


@app.get("/acceptance/eval")
async def acceptance_eval_page():
    return FileResponse(_STATIC / "acceptance-eval.html")


@app.get("/acceptance/data")
async def acceptance_data_page():
    return FileResponse(_STATIC / "acceptance-data.html")


@app.get("/acceptance/errors")
async def acceptance_errors_page():
    return FileResponse(_STATIC / "acceptance-errors.html")


# ch03:后台共用静态资源挂 /static(页面里引 /static/admin.js 等)。
# 注意挂载顺序:/static 先于根路径 catch-all,否则永远轮不到它。
app.mount("/static", StaticFiles(directory=_STATIC), name="static-files")
# 聊天页(原生 JS + SSE),挂根路径,/api/* 与上面的页面路由优先接管
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
