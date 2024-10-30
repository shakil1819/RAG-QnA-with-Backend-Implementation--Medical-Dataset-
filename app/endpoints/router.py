from pydantic import BaseModel
from fastapi import APIRouter
from typing import Dict, Union

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
    output = chain.invoke({"input": query.input})
    if query.detailed:
        return output
    else:
        return output["answer"]