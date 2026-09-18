# app/schemas/agent.py
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话,首轮留空")

class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict

class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str

class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
    suggested_actions: list = Field(default_factory=list)
    interrupt: dict | None = Field(default=None, description="待处理中断载荷(按 type 分派:select_order 带订单选择器 orders;confirm_ticket 带工单预览 preview);无则 None")
