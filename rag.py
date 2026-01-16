import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# LangChain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_state"
VECTORSTORE_DIR = str(Path(__file__).parent / "resources/vectorstore")

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # --- FIX 2: Force CPU usage to avoid 0xC0000409 error ---
            model_kwargs={'device': 'cpu'}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=VECTORSTORE_DIR,
            embedding_function=ef
        )


def process_urls(urls):
    yield "Initialize components"
    initialize_components()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    }

    yield "Resetting vector store ...."
    vector_store.reset_collection()

    yield "Loading data ..."
    loader = UnstructuredURLLoader(urls=urls , headers = headers)
    data = loader.load()

    yield "Splitting text into chunks ...."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(data)

    if docs:
        yield "Add chunks to vector db....."
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(docs, ids=uuids)

    else:
        print("No documents were found/split.")

    yield "Done adding docs to the vector database "

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector store is not initialized")
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm , retriever = vector_store.as_retriever())
    result = chain.invoke({"question": query} , return_only_outputs = True)
    sources = result.get("sources",)

    return result["answer"] , sources

