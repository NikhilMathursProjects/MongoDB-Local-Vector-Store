import requests
# response = requests.post(
#     "http://127.0.0.1:8000/connect",
#     json={"mongo_uri":'mongodb://localhost:27017/',
#           'database_name':'chatbot_db',
#           'collection_name':'chat_messages',
#          }
# )
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())
reponse=requests.post(
    "http://127.0.0.1:8000/fetch_data",
    json={'n':100}
)
print("Status Code:", reponse.status_code)
print("Response JSON:", reponse.json())
