import pathlib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.admin import router as admin_router

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)

# 聊天页(原生 JS + SSE),挂根路径,/api/* 由上面的 router 优先接管
_STATIC = pathlib.Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
