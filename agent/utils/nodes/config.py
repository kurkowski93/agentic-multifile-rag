"""
Configuration for agent nodes.

This module contains configuration settings, embedding models initialization,
and vector store connections used by agent nodes.
"""
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Configure logging
logger = logging.getLogger(__name__)

# Initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize vector store connection
try:
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "default_collection"),
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    logger.info("Successfully connected to Qdrant vector store")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant vector store: {e}")
    vector_store = None 