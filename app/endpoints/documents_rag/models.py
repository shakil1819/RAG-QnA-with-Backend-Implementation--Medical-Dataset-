from pydantic import BaseModel

class QARequest(BaseModel):
    """Model to define the structure of the incoming request."""
    question: str

class QAResponse(BaseModel):
    """Model to define the structure of the response."""
    answer: str