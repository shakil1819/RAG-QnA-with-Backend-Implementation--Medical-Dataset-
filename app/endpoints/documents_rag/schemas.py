from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to be answered")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: list[str] = Field(default_factory=list, description="Source documents used")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
