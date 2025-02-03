# Install dependencies before running:
# http://127.0.0.1:8000/docs
# pip install fastapi uvicorn langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama chromadb pypdf
# pip install -r requirements.txt --no-index --find-links=""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from contextlib import asynccontextmanager

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from chromadb.config import Settings  

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a request model for incoming queries
class QueryRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Initializing document loading and chain creation...")
    
    # Load all PDF documents from the "data/" directory using PyPDFLoader.
    loader = DirectoryLoader(
        "data/", 
        glob="./*.pdf", 
        loader_cls=PyPDFLoader, 
        show_progress=True
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents.")

    # Split documents into smaller text chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(text_chunks)} text chunks.")

    # Create embeddings and build the vector store.
    embedding = HuggingFaceEmbeddings()
    persist_directory = "doc_db"
    client_settings = Settings(
        is_persistent=True,
        persist_directory=persist_directory,
        anonymized_telemetry=False  
    )
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=persist_directory,
        client_settings=client_settings
    )
    retriever = vectorstore.as_retriever()
    logger.info("Vectorstore and retriever created.")

    # Define the prompt template.
    system_prompt = (
        "Use the following context to answer the question. "
        "If you don't know the answer, please say so. "
        "Keep your answer concise. Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Initialize the LLM.
    llm = OllamaLLM(
        model='gemma2:2b-instruct-q5_K_M'
    )

    # Create the question-answer chain and combine it with the retriever.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("Retrieval chain created successfully.")

    # Store the RAG chain in the application state for later use.
    app.state.rag_chain = rag_chain

    yield

    logger.info("Shutdown: Cleaning up resources...")

# Create the FastAPI app using the lifespan context manager.
app = FastAPI(lifespan=lifespan, title="RAG FastAPI Service")

@app.get('/')
async def home():
    return 'Hello World'

@app.post("/ask")
async def ask_question(query_request: QueryRequest):
    logger.info(f"Received query: {query_request.query}")
    rag_chain = app.state.rag_chain
    if rag_chain is None:
        logger.error("Chain not initialized.")
        raise HTTPException(status_code=500, detail="Chain not initialized.")
    try:
        result = rag_chain.invoke({"input": query_request.query})
        logger.info("Query processed successfully.")
        return {"result": result}
    except Exception as e:
        logger.exception("Error processing query:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app)
