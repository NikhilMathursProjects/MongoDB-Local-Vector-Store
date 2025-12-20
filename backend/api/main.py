import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
import hashlib

# Import the backend library
try:
    from backend.mongodb_local import MongoDBLocalVectorSearch
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from backend.mongodb_local import MongoDBLocalVectorSearch

from langchain_core.embeddings import Embeddings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MongoDB Vector Workstation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev convenience with dynamic ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock Embeddings ---
class MockEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        vector = np.random.rand(self.dimensions).astype("float32")
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist() if norm > 0 else vector.tolist()

# --- Models ---
class ConnectionRequest(BaseModel):
    uri: str

class DBListRequest(BaseModel):
    uri: str

class CollectionListRequest(BaseModel):
    uri: str
    database: str

class VectorActionRequest(BaseModel):
    uri: str
    database: str
    collection: str
    
class SearchRequest(VectorActionRequest):
    query: str
    k: int = 5
    filter: Optional[Dict[str, Any]] = None

class AddDocsRequest(VectorActionRequest):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

# --- Helpers ---
def get_client(uri: str) -> MongoClient:
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=2000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Connection String: {e}")

def get_vector_store(uri: str, db_name: str, coll_name: str) -> MongoDBLocalVectorSearch:
    client = get_client(uri)
    collection = client[db_name][coll_name]
    embedder = MockEmbeddings(dimensions=384)
    # Default FAISS config
    faiss_config = {
        'metric': 'cosine',
        'index_type': 'hnsw',
        'index_params': {'dimensions': 384, 'neighbours': 16, 'efSearch': 64, 'efConstruction': 200}
    }
    return MongoDBLocalVectorSearch(
        collection=collection,
        embedder_model=embedder,
        faiss_config=faiss_config
    )

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "active", "mode": "workstation"}

@app.post("/connect")
def check_connection(request: ConnectionRequest):
    """Ping the server to verify connection."""
    client = get_client(request.uri)
    try:
        # Ping command
        client.admin.command('ping')
        return {"status": "connected", "message": "Connection successful"}
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")

@app.post("/databases")
def list_databases(request: DBListRequest):
    """List all databases."""
    client = get_client(request.uri)
    try:
        # Exclude system DBs if desired, or keep them. keeping 'local' is useful sometimes.
        dbs = client.list_database_names()
        return {"databases": dbs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections")
def list_collections(request: CollectionListRequest):
    """List collections in a specific database."""
    client = get_client(request.uri)
    try:
        db = client[request.database]
        colls = db.list_collection_names()
        # Optional: Get stats for each connection? For now just names.
        return {"collections": colls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/search", response_model=List[SearchResult])
def vector_search(request: SearchRequest):
    """Perform vector search on a specific namespace."""
    try:
        vs = get_vector_store(request.uri, request.database, request.collection)
        results = vs.similarity_search_with_score(request.query, k=request.k, pre_filter_query=request.filter)
        
        response = []
        for doc, score in results:
            response.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ))
        return response
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/add")
def add_documents(request: AddDocsRequest):
    """Add documents to a specific namespace."""
    try:
        vs = get_vector_store(request.uri, request.database, request.collection)
        ids = vs.add_texts(texts=request.texts, metadatas=request.metadatas)
        return {"status": "success", "inserted_count": len(ids), "ids": [str(id) for id in ids]}
    except Exception as e:
        logger.exception("Add docs failed")
        raise HTTPException(status_code=500, detail=str(e))
