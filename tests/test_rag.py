import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.endpoints.documents_rag.chains import RAGChain
from app.configs import settings

client = TestClient(app)

@pytest.fixture
def mock_rag_chain():
    with patch('app.endpoints.documents_rag.chains.RAGChain') as mock:
        yield mock

def test_answer_endpoint_success(mock_rag_chain):
    # Mock the RAG chain response
    mock_rag_chain.get_answer.return_value = (
        "Test answer",
        ["Source 1", "Source 2"]
    )
    
    response = client.post(
        "/api/answer",
        json={"question": "Test question"},
        headers={"X-API-Key": settings.API_KEY}
    )
    
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Test answer",
        "sources": ["Source 1", "Source 2"]
    }

def test_answer_endpoint_invalid_api_key():
    response = client.post(
        "/api/answer",
        json={"question": "Test question"},
        headers={"X-API-Key": "invalid_key"}
    )
    
    assert response.status_code == 401

def test_answer_endpoint_empty_question():
    response = client.post(
        "/api/answer",
        json={"question": ""},
        headers={"X-API-Key": settings.API_KEY}
    )
    
    assert response.status_code == 400

def test_load_documents_success(mock_rag_chain):
    response = client.post(
        "/api/load-documents",
        headers={"X-API-Key": settings.API_KEY}
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "Documents loaded successfully"}
