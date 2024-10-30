from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import api_router

app = FastAPI(
    title="RAG API",
    description="API for Question Answering using RAG for Medical Dataset (CSV)",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "http://localhost:8000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


app.include_router(api_router, prefix="/api/v1")