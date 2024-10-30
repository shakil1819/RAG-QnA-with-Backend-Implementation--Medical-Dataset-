import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from pathlib import Path
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import uvicorn, logging
# from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter(tags=["question-answering"])

class QueryRequest(BaseModel):
    query: str

def setup_question_answering_pipeline(query: str) -> str:
    """
    Sets up an environment to load CSV documents, vectorize them, and create a question-answering 
    pipeline with retrieval-based augmented generation (RAG) capabilities.

    Args:
        directory_path (str): Path to the directory containing CSV files.
        query (str): The question to be answered using the pipeline.

    Returns:
        str: The answer to the provided query based on document retrieval and generation.

    Environment Variables:
        OPENAI_API_KEY (str): API key for accessing OpenAI services.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
    
    llm = ChatOpenAI(model=os.getenv("GPT_4_TEXT_MODEL"))

    directory_path = "."

    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            loader = CSVLoader(file_path=file_path)
            docs = loader.load_and_split()
            all_docs.extend(docs)

    embeddings = OpenAIEmbeddings()
    embedding_dim = 1536  # OpenAI embeddings dimension
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=all_docs)
    retriever = vector_store.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": query})
    return result['answer']

@router.post("/answer")
async def answer_query(request: QueryRequest):
    try:
        answer = setup_question_answering_pipeline(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Move the FastAPI app creation outside the __main__ block
app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    # Use the module path instead of the app instance directly
    uvicorn.run(
        "app.rag:app",  # Use the module path
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
    # To run this file directly:
    # 1. Make sure you have all required dependencies installed
    # 2. Set up your .env file with OPENAI_API_KEY and other required variables
    # 3. Run from command line: python -m app.rag
    # 4. The FastAPI server will start on http://0.0.0.0:8000
    # 5. Access the API docs at http://0.0.0.0:8000/docs
