from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
from fastapi.security.api_key import APIKey, APIKeyHeader

from app.endpoints.documents_rag.schemas import QuestionRequest, AnswerResponse, ErrorResponse
from app.endpoints.documents_rag.exceptions import RAGException
from app.endpoints.documents_rag.logging import logger
from app.configs import settings
from app.endpoints.documents_rag.chains import process_documents_and_answer_question

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Annotated[APIKey, Depends(api_key_header)]):
    """Verify API key from header."""
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

@router.post(
    "/answer",
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_answer(
    request: QuestionRequest,
    api_key: Annotated[APIKey, Depends(verify_api_key)]
):
    """
    Generate an answer for the given question using RAG.
    
    Args:
        request: QuestionRequest containing the question
        api_key: API key for authentication
        
    Returns:
        AnswerResponse containing the generated answer
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(f"Processing question: {request.question}")
        answer = process_documents_and_answer_question(request.question)
        
        return AnswerResponse(
            answer=answer
        )
        
    except RAGException as e:
        logger.error(f"RAG error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )