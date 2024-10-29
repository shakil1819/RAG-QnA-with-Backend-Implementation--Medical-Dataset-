from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from endpoints.documents_rag.routers import qa_router
from endpoints.documents_rag.exceptions import RAGException
from endpoints.documents_rag.logging import logger

app = FastAPI(
    title="RAG QA System",
    description="A Retrieval-Augmented Generation Question-Answering System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(qa_router.router, prefix="/api", tags=["QA"])

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle RAG-specific exceptions."""
    logger.error(f"RAG error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request parameters", "details": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"},
    )

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
