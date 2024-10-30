from fastapi import APIRouter
from app.endpoints.router import qa_router

api_router = APIRouter()

api_router.include_router(qa_router)