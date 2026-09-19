"""ch09 可观测:Langfuse 挂载与 trace 标注,全部可选降级。

课程 README 姿势:设三个环境变量 + 图编译时挂一次回调,节点业务代码零侵入。
项目配置经 pydantic-settings(.env),不依赖 os.environ,故显式传参初始化单例。
所有对外函数在未配置 Langfuse 时必须是安全 no-op——观测是增强,不是依赖。

v4 偏差(dev-notes 有记录):README 的 get_client().update_current_trace() 在 SDK 4.x
已删除,服务端 v4 又是 events_only 模式(REST trace upsert 404、trace.get/list 退役)。
意图标注实测走双通道(trace 级属性直改 root span + propagate 进上下文),见 tag_intent。
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)
_client = None
_handler = None


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """初始化(或复用)Langfuse 单例。SDK 4.x host/base_url 两参名都收,取 README 口径 host。
    须先于 _make_handler:CallbackHandler 复用全局单例,而项目不靠 os.environ 传 key。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
    return _client


def _make_handler():
    global _handler
    from langfuse.langchain import CallbackHandler
    _handler = CallbackHandler()
    return _handler


def get_langfuse():
    """启用时返回已初始化 client(cost 脚本用 client.api 查询);未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调,全图自动 trace(README:「编译时挂一次,节点里一行不用动」)。
    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


def tag_intent(intent: str, confidence: float, session_id: str | None = None) -> None:
    """把意图打进本轮 trace 的 tags(Cost Control 按意图分堆的钩子)。

    v4 实测语义:tags 是 trace 级属性(langfuse.trace.tags),但 Metrics 的
    observations 视图按「各 span 自身携带的该属性」分组,故走双通道:
    1) 直改 root span 属性——handler 挂的 root observation 图终才导出,运行中
       set 即时生效,观测页 trace 详情得以显示 tags;
    2) propagate_attributes 进上下文后刻意不退出——span_processor 会把 propagated
       属性复制到此后本 task 内新建的每个 observation 上,LLM generation 由此
       按 intent:xxx 分组。LangGraph 每个节点跑在独立 task,attach 出不了本节点,
       产生模型调用的节点(Task 3/5 接线)需各自调一次本函数。
    session_id 给定时只标注匹配的那条 trace(并发轮次不串);未启用/未挂 handler
    静默跳过;任何异常只记 debug——观测失败不许影响业务流。confidence 留参数位
    不上 trace(以 DB 列为准,任务 4)。"""
    if not langfuse_enabled():
        return
    tags = [f"intent:{intent}"]
    try:
        if _handler is not None:
            runs = getattr(_handler, "_runs", {})
            for root_id in list(getattr(_handler, "_root_run_states", {})):
                span = getattr(runs.get(root_id), "_otel_span", None)
                if span is None:
                    continue
                if session_id is not None:
                    sid = (span.attributes or {}).get("session.id")
                    if sid is not None and str(sid) != str(session_id):
                        continue
                span.set_attribute("langfuse.trace.tags", tags)
    except Exception:
        logger.debug("langfuse tag_intent(trace tags)失败(已忽略)", exc_info=True)
    try:
        # 通道 2:直接 attach 进 OTel 上下文。不用 propagate_attributes().__enter__():
        # 丢弃 CM 临时对象会让 generator 被 GC、finally 立刻 detach(smoke-12/13 对照
        # 实证)。手动 attach 无 token 也无泄漏——LangGraph 节点跑在独立 task,上下文
        # 副本随 task 丢弃。键名 langfuse.propagated.tags 为 SDK 4.15.4 内部常量。
        from opentelemetry import context as otel_context_api
        otel_context_api.attach(otel_context_api.set_value(
            "langfuse.propagated.tags", tags, otel_context_api.get_current()))
    except Exception:
        logger.debug("langfuse tag_intent(propagate)失败(已忽略)", exc_info=True)
