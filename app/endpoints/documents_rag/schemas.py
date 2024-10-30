from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to be answered")

class AnswerResponse(BaseModel):
    answer: str

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
