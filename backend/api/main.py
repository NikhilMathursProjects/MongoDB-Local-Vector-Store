import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
import hashlib

# Import the backend library
# Adjust import path if needed depending on where this is run
try:
    from backend.mongodb_local import MongoDBLocalVectorSearch
except ImportError:
    # If running from root, this import might work slightly differently or need path adjustment
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from backend.mongodb_local import MongoDBLocalVectorSearch

from langchain_core.embeddings import Embeddings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MongoDB Local Vector Dashboard API")

# --- Mock Embeddings for Demo Purposes ---
class MockEmbeddings(Embeddings):
    """
    Deterministic fake embeddings for demo purposes.
    Generates a vector based on the hash of the text to ensure consistency.
    """
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # Create a seed from the text
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        # Generate random vector, then normalize
        vector = np.random.rand(self.dimensions).astype("float32")
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()

# --- Global Vector Store Instance ---
vector_store: Optional[MongoDBLocalVectorSearch] = None
mongo_client: Optional[MongoClient] = None

# --- Models ---
class AddDocumentsRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class StatsResponse(BaseModel):
    document_count: int
    index_type: str
    metric: str
    dimensions: int

# --- Lifecycle ---
@app.on_event("startup")
async def startup_event():
    global vector_store, mongo_client
    
    # Configuration
    MONGO_URI = os.getenv("MONGODB_LOCAL_URI", "mongodb://localhost:27017/")
    DB_NAME = "vector_store_demo"
    COLLECTION_NAME = "vectors"
    
    try:
        logger.info(f"Connecting to MongoDB at {MONGO_URI}...")
        mongo_client = MongoClient(MONGO_URI)
        collection = mongo_client[DB_NAME][COLLECTION_NAME]
        
        # Initialize Embeddings
        embedder = MockEmbeddings(dimensions=384)
        
        # FAISS Config
        faiss_config = {
            'metric': 'cosine', 
            'index_type': 'hnsw',
            'index_params': {
                'dimensions': 384,
                'neighbours': 16,
                'efSearch': 64,
                'efConstruction': 200,
            }
        }
        
        vector_store = MongoDBLocalVectorSearch(
            collection=collection,
            embedder_model=embedder,
            faiss_config=faiss_config,
            index_name="demo_index"
        )
        logger.info("Vector Store Initialized Successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        # We don't raise here to allow the app to start and show status
        vector_store = None

# --- Endpoints ---

@app.get("/health")
def health_check():
    status = "healthy" if vector_store else "unhealthy"
    return {"status": status, "backend": "active"}

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    # Get count from mongo
    count = vector_store._collection.count_documents({})
    
    return {
        "document_count": count,
        "index_type": vector_store._faiss_config.get('index_type'),
        "metric": vector_store._faiss_config.get('metric'),
        "dimensions": 384 # Hardcoded for mock
    }

@app.post("/documents")
def add_documents(request: AddDocumentsRequest):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    try:
        ids = vector_store.add_texts(
            texts=request.texts,
            metadatas=request.metadatas
        )
        return {"status": "success", "inserted_count": len(ids), "ids": [str(id) for id in ids]}
    except Exception as e:
        logger.exception("Error adding documents")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    try:
        results = vector_store.similarity_search_with_score(
            query=request.query,
            k=request.k,
            pre_filter_query=request.filter
        )
        
        response = []
        for doc, score in results:
            response.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ))
        return response
    except Exception as e:
        logger.exception("Error performing search")
        raise HTTPException(status_code=500, detail=str(e))
