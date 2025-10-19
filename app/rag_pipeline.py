# app/rag_pipeline.py

import os
try:
    from langchain.globals import set_verbose, get_verbose
except ImportError:
    from langchain_core.globals import set_verbose, get_verbose
set_verbose(False)

set_verbose(True)  # Si quieres ver logs detallados

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
import mlflow

load_dotenv()

DATA_DIR = "data/pdfs"
PROMPT_DIR = "app/prompts"
VECTOR_DIR = "vectorstore"

def load_documents(path=DATA_DIR):
    docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def save_vectorstore(chunk_size=512, chunk_overlap=50, persist_path=VECTOR_DIR):
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.set_tag("vectorstore", persist_path)

def load_vectorstore(chunk_size=512, chunk_overlap=50):
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

def load_prompt(version="v1_asistente_rrhh"):
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    with open(prompt_path, "r") as f:
        prompt_text = f.read()
    return PromptTemplate(input_variables=["context", "question"], template=prompt_text)

def build_chain(vectordb, prompt_version="v1_asistente_rrhh"):
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(model="gpt-4o", temperature=0),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
