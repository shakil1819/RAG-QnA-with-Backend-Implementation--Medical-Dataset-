import pytest
from app.rag.llm import get_retrieval_chain

@pytest.fixture
def mock_chain():
    chain = get_retrieval_chain()
    return chain