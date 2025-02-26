"""Vector store management for the pregnancy knowledge base."""
import time
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

from pregnancy_kb.config import QDRANT_PATH, COLLECTION_NAME, BATCH_SIZE, VECTOR_SIZE, EMBEDDING_MODEL, EMBEDDING_TYPE

def initialize_vector_store():
    """Initialize the Qdrant vector store."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(path=str(QDRANT_PATH))
        
        # Ensure collection exists
        collections = client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            
        # Initialize embeddings based on configuration
        if EMBEDDING_TYPE == 'sentence_transformers':
            # Use fine-tuned Sentence Transformers model
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        else:
            # Use OpenAI embeddings as fallback
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        
        return vector_store, client
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return None, None

def add_documents_in_batches(vector_store, documents: List[Document], batch_size: int = BATCH_SIZE):
    """Add documents to the vector store in batches."""
    if not documents:
        return {}
        
    total_chunks = 0
    categories = {}
    
    # Track categories for reporting
    for doc in documents:
        cat = doc.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    # Process in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} chunks)")
        
        try:
            # Add batch to vector store
            vector_store.add_documents(batch)
            total_chunks += len(batch)
            
            # Sleep briefly to avoid rate limits
            if i + batch_size < len(documents):
                time.sleep(1)
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
    
    print(f"\nSuccessfully added {total_chunks} document chunks to the knowledge base")
    
    return categories 