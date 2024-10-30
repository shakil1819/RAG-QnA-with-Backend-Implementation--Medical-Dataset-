from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from app.configs import settings
from app.endpoints.documents_rag.prompts import system_prompt
  

def process_documents_and_answer_question(question: str, directory_path: str = ".") -> str:
    load_dotenv()

    openai_api_key = settings.OPENAI_API_KEY
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set in environment variables.")
    os.environ['OPENAI_API_KEY'] = openai_api_key

    llm = ChatOpenAI(model=settings.GPT_4_TEXT_MODEL)

    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            loader = CSVLoader(file_path=file_path)
            docs = loader.load_and_split()
            all_docs.extend(docs)

    embeddings = OpenAIEmbeddings()
    embedding_dimension = 1536
    index = faiss.IndexFlatL2(embedding_dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=all_docs)

    retriever = vector_store.as_retriever()


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"question": question})
    return response["answer"]