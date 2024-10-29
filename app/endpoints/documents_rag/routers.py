from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
from fastapi.security.api_key import APIKey, APIKeyHeader

from ..schemas import QuestionRequest, AnswerResponse, ErrorResponse
from ..chains import rag_chain
from ..exceptions import RAGException
from ..logging import logger
from app.configs import settings

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
        AnswerResponse containing the generated answer and sources
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(f"Processing question: {request.question}")
        answer, sources = rag_chain.get_answer(request.question)
        
        return AnswerResponse(
            answer=answer,
            sources=sources
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

@router.post("/load-documents")
async def load_documents(
    api_key: Annotated[APIKey, Depends(verify_api_key)]
):
    """
    Load and index documents from the configured data path.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        dict: Status message
        
    Raises:
        HTTPException: If there's an error loading documents
    """
    try:
        rag_chain.load_documents(settings.DATA_PATH)
        return {"status": "Documents loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading documents: {str(e)}"
        )