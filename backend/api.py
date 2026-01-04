from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from mongodb_local import MongoDBLocalVectorSearch

from langchain_core.embeddings import Embeddings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


app=FastAPI()

app.state.mongo_client = None
app.state.collection = None
app.state.faiss_vectorstore = None

class FetchDataRequest(BaseModel):
    n:int

class MongoConnectRequest(BaseModel):
    mongo_uri:str
    database_name:str
    collection_name:str

class MongoQueryRequest(BaseModel):
    mongo_query:Dict[str,Any]

#---------------INDEX REQUEST MODELS--------------------

class IndexAddDataRequest(BaseModel):
    texts:List[str] #could have an issue with the incoming data not being of this type (different language and all)
    metadatas:Optional[List[Dict[str,Any]]]=None

# class InitVectorStore(BaseModel):
#     connection_str:str
#     namespace:str
#     embedder_model:Embeddings

    

# @app.post('init_faiss_vectorstore')
# def init_faiss_vectorstore(req:InitVectorStore):
#     if app.state.collection==None :
#         raise HTTPException(detail='Collection undefined (database not connected)')

#     if app.state.faiss_vectorstore==None:
#         vectorstore=MongoDBLocalVectorSearch(
#             collection=app.state.collection,
#             embedder_model=embed,
#         )

# @app.post('/add_data')
# def faiss_add_data(req:IndexAddDataRequest):
    #check if class object is created with proper init
    
def fetch_n_data(collection,n:int):
    docs = list(collection.find({}, {"_id": 0}).limit(n))
    return docs

@app.post('/connect')
def connect_to_db(req:MongoConnectRequest):
    """connection to mongodb server""" 
    try:
        client:MongoClient = MongoClient(req.mongo_uri)
        collection=client[req.database_name][req.collection_name]
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
    
    app.state.mongo_client=client
    app.state.collection=collection

    data=fetch_n_data(collection,25)
    return {
        'status':'connected',
        'database_name':req.database_name,
        'collection':req.collection_name,
        'data':data
    }


@app.post('/fetch_data')
def fetch_data(req: FetchDataRequest):
    """Returns n data items"""
    collection=app.state.collection
    if collection is None:
        raise HTTPException(status_code=400,detail='Database not connected')
    
    docs=fetch_n_data(collection,25)
    return {
        "count": len(docs),
        "data_item_list": docs
    }

@app.post('/querydb')
def querydb(req:MongoQueryRequest):
    """
    Base MongoDB query (Exact Search)
    """
    collection= app.state.collection
    if collection is None:
        return HTTPException(status_code=400,detail='Database not connected')
    docs=list(collection.find({req.mongo_query}))
    return{
        'count':len(docs),
        'data':docs
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0",reload=False, port=8000)