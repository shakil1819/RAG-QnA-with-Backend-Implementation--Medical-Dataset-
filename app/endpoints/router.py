from pydantic import BaseModel
from fastapi import APIRouter
from typing import Dict, Union
from fastapi import HTTPException
from app.logging import logger
from app.rag.llm import get_retrieval_chain

qa_router = APIRouter(
    prefix="/qa",
    tags=["Question Answering"],
    responses={404: {"description": "Not found"}},
)

chain = get_retrieval_chain()

class Query(BaseModel):
    input: str
    detailed: bool

@qa_router.post("/question", response_model=Union[str, Dict])
def answer_question(query: Query):
    try:
        output = chain.invoke({"input": query.input})
        if query.detailed:
            return output
        else:
            return output["answer"]
    except Exception as e:
        logger.warning(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question"
        )