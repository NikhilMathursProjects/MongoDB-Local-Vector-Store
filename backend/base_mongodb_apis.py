from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient

app=FastAPI()

client=None
collection=None

class FetchDataRequest(BaseModel):
    n:int

class MongoConnectRequest(BaseModel):
    mongo_uri:str
    database_name:str
    collection_name:str

class MongoQueryRequest(BaseModel):
    mongo_query:str

@app.post('/connect')
def connect_to_db(req:MongoConnectRequest):
    """connection to mongodb server""" 
    global client,collection
    client=MongoClient(req.mongo_uri)
    collection=client[req.database_name][req.collection_name]
    return {
        'status':'connected',
        'database_name':req.database_name,
        'collection':req.collection_name
    }


@app.post('/fetch_n_data')
def fetch_n_data(req: FetchDataRequest):
    """To place the n data items in the data window"""
    global collection
    if collection is None:
        return {'error':'Database not connected'}
    
    docs = list(collection.find({}, {"_id": 0}).limit(req.n))
    return {
        "count": len(docs),
        "data": docs
    }

@app.post('/querydb')
def querydb(req:MongoQueryRequest):
    """To refresh the data window with user queried data"""
    global collection
    if collection is None:
        return {'error':'Database not connected'}
    docs=list(collection.find({req.mongo_query}))
    return{
        'count':len(docs),
        'data':docs
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)