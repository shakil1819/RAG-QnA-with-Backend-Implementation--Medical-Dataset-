from pydantic import BaseModel
from fastapi import FastAPI
from app.rag import get_Retrieval_chain

app = FastAPI()

chain = get_Retrieval_chain()

class Query(BaseModel):
    input: str
    detailed: bool = False

@app.post("/question")
def answer_question(query: Query):
    output = chain.invoke({"input": query.input})
    if query.detailed:
        return output
    else:
        return output["answer"]

