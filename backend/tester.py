# from pymongo import MongoClient
# from bson import ObjectId
# from typing import Optional,List

# URI='mongodb://localhost:27017/'
# client=MongoClient(URI)
# collection=client['tester']['tester']
# print(type(collection))

# def convert_to_ObjectId(ids:Optional[List[str]]=None)->List[ObjectId]:
#     if not ids:
#         return None
    
#     object_ids=list(map(ObjectId,ids))
#     return object_ids

# ids=['6834952b903ab45df6efec1b','684fb0336f8d7bdc975905dc']
# obj_ids=convert_to_ObjectId(ids)
# id_query={
#     '_id':{
#         '$in':obj_ids
#     }
# }

# docs=list(collection.find(id_query,{'_id':1,'name':1}))
# print(docs)



# doc={
#     'id':'meow2',
#     'deleted':True
# }
# insert_result=collection.insert_one(doc)


# from pymongo import MongoClient
# from bson import ObjectId
# from typing import Optional,List
# string_ids_list = ["60a7e6b0b9b6f4a8a4b6f8a0", "60a7e6b0b9b6f4a8a4b6f8a1"]


# object_list=convert_to_ObjectId(string_ids_list)
# print(object_list)

# cursor = collection.find({}).limit(25)
# Note: cursor.count() is deprecated in newer pymongo
# for document in cursor:
    # Process each document here (e.g., print it, add to a list, etc.)
    # Using dumps() for pretty printing JSON documents
    # print(document)