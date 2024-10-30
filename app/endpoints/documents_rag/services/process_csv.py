import faiss
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
from app.configs import settings
from app.endpoints.documents_rag.utils import clean_text, chunk_text

load_dotenv()
openai_api_key = settings.OPENAI_API_KEY


llm = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL)

loader = CSVLoader(file_path=settings.DATA_PATH)
docs = loader.load_and_split()

# Clean and preprocess the documents
docs = [clean_text(doc) for doc in docs]

# Split documents into chunks
chunked_docs = []
for doc in docs:
    chunked_docs.extend(chunk_text(doc))

embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(documents=docs)

# Ensure documents are loaded into the vector store
if not vector_store.is_loaded():
    vector_store.add_documents(documents=chunked_docs)
