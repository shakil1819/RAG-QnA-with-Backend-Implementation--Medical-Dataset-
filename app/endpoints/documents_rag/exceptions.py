class RAGException(Exception):
    """Base exception for RAG-related errors."""
    pass

class DocumentLoadError(RAGException):
    """Raised when there's an error loading or processing documents."""
    pass

class QuestionProcessingError(RAGException):
    """Raised when there's an error processing a question."""
    pass

class VectorStoreError(RAGException):
    """Raised when there's an error with the vector store operations."""
    pass
