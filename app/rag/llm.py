import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from os import listdir, path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStore
from langchain_core.runnables.base import Runnable
import argparse
from app.configs import settings
from app.logging import logger
load_dotenv()

def load_api_key() -> None:
    """
    Get env var for API key, if not available try to get it from the .env file.
    """
    api_key_var = settings.OPENAI_API_KEY
    env_var = os.getenv(api_key_var)
    if env_var and env_var != "":
        os.environ[api_key_var] = env_var
    else:
        load_dotenv(override=True)


def load_documents(folder: str = settings.DATA_DIR) -> list[Document]:
    docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                loader = CSVLoader(file_path, encoding='utf-8')
                docs_ = loader.load_and_split()
                docs.extend(docs_)
    return docs


def populate_vector_db(docs: list[Document], db_path: str = settings.VECTOR_DIR) -> VectorStore:
    """

    :param docs:    Documents loaded from the disk with the TextLoader.
    :param db_path: Path where the vector database will be stored.
    :return:        An initialized vector store.
    """
    load_api_key()
    embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(db_path)
    return vector


def load_vector_db(db_path: str = settings.VECTOR_DIR) -> VectorStore:
    """
    :param db_path: Path to find the saved vector store.
    :return:        The vector store loaded from the disk.
    """
    load_api_key()
    embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return db


def get_retrieval_chain() -> Runnable:
    """
    Creates a pipeline for asking the LLM questions about our documents.
    The document chain creates a prompt from a given document.
    The retrieval chain first looks up the most relevant documents in the database
    and then produces queries using these documents as a context for the LLM.
    :return: The retrieval chain ready to be queried.
    """
    load_api_key()
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

        <context>
        {context}
        </context>
    
        Question: {input}"""
    )
    llm = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL, openai_api_key=settings.OPENAI_API_KEY)
    document_chain = create_stuff_documents_chain(llm, prompt)

    vector = load_vector_db()
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=settings.VECTOR_DIR)
    parser.add_argument("--document-folder", type=str, default=settings.DATA_DIR)
    parser.add_argument("--repopulate", action="store_true", default=False)

    args = parser.parse_args()
    if path.exists(args.db_path) and args.repopulate is False:
        logger.warning(
            f"The vector DB path '{args.db_path}' already exists. Run with option --repopulate to force repopulation."
        )
        exit()

    docs = load_documents(args.document_folder)
    populate_vector_db(docs, args.db_path)
