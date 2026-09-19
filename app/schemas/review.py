from pydantic import BaseModel


class ApproveRequest(BaseModel):
    approved_answer: str


class ReviewItemOut(BaseModel):
    id: int
    normalized_question: str
    ai_suggested_answer: str | None
    occurrence_count: int
    review_status: str
    created_at: str
