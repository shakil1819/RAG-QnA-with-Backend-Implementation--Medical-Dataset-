import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def sample_query():
    return {
        "input": "What are the symptoms of diabetes?",
        "detailed": False
    }

@pytest.fixture
def sample_query_detailed():
    return {
        "input": "What are the symptoms of diabetes?",
        "detailed": True
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_answer_question(sample_query):
    response = client.post("/api/v1/qa/question", json=sample_query)
    assert response.status_code == 200
    assert isinstance(response.json(), str)

def test_answer_question_detailed(sample_query_detailed):
    response = client.post("/api/v1/qa/question", json=sample_query_detailed)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_invalid_query():
    response = client.post("/api/v1/qa/question", json={"invalid": "query"})
    assert response.status_code == 422 