# src/kumulus_consultant/rag_system.py
import logging
import os
from typing import Any, Dict, List

# Corrected imports for LangChain components
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CORRECTED import for our settings file
from kumulus_consultant.config.settings import settings

logger = logging.getLogger(__name__)

class HybridRAG:
    # ... (the rest of the class remains exactly as you posted it)
    def __init__(self, vector_db_path: str = settings.VECTOR_DB_PATH):
        logger.info("Initializing HybridRAG system.")
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        self.db = None
        self.ensemble_retriever = None
    def load_retrievers(self, docs: List[Document]):
        if os.path.exists(self.vector_db_path):
            logger.info(f"Loading existing FAISS index from {self.vector_db_path}")
            self.db = FAISS.load_local(
                self.vector_db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.warning(
                f"FAISS index not found at {self.vector_db_path}. "
                "Creating a new one from provided documents."
            )
            self.db = FAISS.from_documents(docs, self.embedding_model)
            self.db.save_local(self.vector_db_path)
            logger.info(f"New FAISS index saved to {self.vector_db_path}")
        faiss_retriever = self.db.as_retriever(search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
        )
        logger.info("Hybrid ensemble retriever initialized successfully.")
    def retrieve(self, query: str) -> str:
        if not self.ensemble_retriever:
            logger.error("Retriever has not been loaded. Please load documents first.")
            return "Retriever has not been loaded. Please load documents first."
        logger.info(f"Retrieving context for query: '{query}'")
        retrieved_docs = self.ensemble_retriever.invoke(query)
        if not retrieved_docs:
            logger.warning(f"No relevant context found for query: '{query}'")
            return "No relevant context found in the knowledge base."
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        return f"Retrieved Context:\n---\n{context}"
    @staticmethod
    def load_and_process_docs(doc_directory: str) -> List[Document]:
        logger.info(f"Loading documents from directory: {doc_directory}")
        loader = DirectoryLoader(doc_directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        if not documents:
            logger.warning(f"No documents found in directory: {doc_directory}")
            return []
        logger.info(f"Splitting {len(documents)} documents into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks.")
        return chunks

def summarize_geojson_for_rag(geojson_data: Dict[str, Any]) -> str:
    # ... (this function remains exactly the same)
    if not isinstance(geojson_data, dict):
        return "Invalid GeoJSON data: not a dictionary."
    analysis_type = geojson_data.get("analysis_type", "General")
    location = geojson_data.get("location", "an unspecified area")
    summary_parts = [f"Geospatial analysis result for {location} ({analysis_type}):"]
    for key, value in geojson_data.items():
        if key in ["analysis_type", "location", "status"]:
            continue
        formatted_key = key.replace("_", " ").capitalize()
        if isinstance(value, dict):
            nested_summary = ", ".join(f"{k}: {v}" for k, v in value.items())
            summary_parts.append(f"- {formatted_key}: {{ {nested_summary} }}.")
        elif isinstance(value, list):
            list_summary = ", ".join(map(str, value))
            summary_parts.append(f"- {formatted_key}: [ {list_summary} ].")
        else:
            summary_parts.append(f"- {formatted_key}: {value}.")
    if len(summary_parts) == 1:
        return "No significant geospatial features were found to summarize."
    return "\n".join(summary_parts)