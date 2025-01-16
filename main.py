# Install dependencies before running:
# !pip install -q langchain-ollama langchain langchain-community langchain-chroma langchain-text-splitters langchain-huggingface unstructured[pdf] nltk fastapi uvicorn

# http://127.0.0.1:8000/docs

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

import nltk

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

# Global variable to hold our initialized QA chain
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    logger.info("Starting up the application...")

    # === Initialization code ===
    logger.info("Creating embeddings using HuggingFaceEmbeddings.")
    embedding = HuggingFaceEmbeddings()

    persist_directory = "doc_db"

    # Check if the persisted directory exists to decide whether to create a new vector store or load an existing one.
    if os.path.exists(persist_directory):
        logger.info("Persisted vector store found. Loading existing Chroma vector store.")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        logger.info("No persisted vector store found. Loading PDF documents and creating a new one.")
        logger.info("Loading PDF documents from the 'data/' directory.")
        loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=UnstructuredPDFLoader, show_progress=True)
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents. Splitting documents into smaller chunks.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40,
            length_function=len,
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(text_chunks)} text chunks.")

        logger.info("Initializing or loading vector store using Chroma.")
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding,
            persist_directory=persist_directory
        )

    logger.info("Creating a retriever from the vector store.")
    retriever = vectorstore.as_retriever()

    logger.info("Initializing the language model with OllamaLLM.")
    llm = OllamaLLM(
        model='gemma2:2b-instruct-q5_K_M',
        temperature=0.5
    )

    logger.info("Setting up the retrieval-based QA chain.")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    logger.info("Initialization complete. QA chain is ready.")
    # === End of initialization ===

    yield

    logger.info("Shutting down application...")

# Initialize FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    logger.info("Received request at '/' endpoint.")
    return "Hello World"

class QARequest(BaseModel):
    query: str

@app.post("/query")
async def query_qa(request: QARequest):
    logger.info(f"Received query: {request.query}")
    if qa_chain is None:
        logger.error("QA chain has not been initialized.")
        return {"error": "QA chain not initialized."}
        
    response = qa_chain.invoke({"query": request.query})
    logger.info("Query processed successfully.")
    return response
