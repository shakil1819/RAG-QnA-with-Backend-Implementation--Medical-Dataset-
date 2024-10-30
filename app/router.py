import logging
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from app.rag import setup_question_answering_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["question-answering-with-CSV"])
class QueryRequest(BaseModel):
    query: str

@router.post("/answer_query")
def answer_query(request: QueryRequest):
    """
    Endpoint to handle user queries by calling the setup_question_answering_pipeline function.
    
    Args:
        request (QueryRequest): A JSON payload containing the directory path and user query.
    
    Returns:
        dict: A JSON response with the answer or an error message.
    """
    try:
        answer = setup_question_answering_pipeline(
            query=request.query
        )
        return {"answer": answer}

    except Exception as e:
        logger.warning(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")
