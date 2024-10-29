from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
from typing import List, Optional

from endpoints.documents_rag.logging import logger
from app.configs import settings

class RAGChain:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_store: Optional[FAISS] = None
        
    def load_documents(self, data_path: str) -> None:
        """Load and index documents from the specified path."""
        try:
            data_dir = Path(data_path)
            texts = []
            
            for file_path in data_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.extend(self.text_splitter.split_text(f.read()))
            
            logger.info(f"Loaded {len(texts)} text chunks")
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def get_answer(self, question: str) -> tuple[str, List[str]]:
        """Generate an answer for the given question using RAG."""
        if not self.vector_store:
            raise ValueError("Documents not loaded. Call load_documents first.")

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            sources = [doc.page_content[:200] + "..." for doc in result["source_documents"]]
            
            return result["result"], sources
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

# Create singleton instance
rag_chain = RAGChain()
