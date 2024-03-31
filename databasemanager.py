import os
from pymongo import MongoClient
from torch import Tensor
import torch
from typing import  List 

class DatabaseManager:
    def __init__(self, database: str, collection: str):

        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.db = None
        self.collection = None
        
    def establish_connection(self):
        # Use environment variables for sensitive information
        user = os.getenv('MONGO_USERNAME')
        pw = os.getenv('MONGO_PASSWORD')
        link = os.getenv("MONGO_LINK")
        
        CONNECTION_STRING = f"mongodb+srv://{user}:{pw}@{link}"
        
        try: 
            self.client = MongoClient(CONNECTION_STRING)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
        except Exception as e:
            # Log the error
            print(f"Error connecting to Database: {e}")
            raise
        
    def insert_embeddings_batch(self, embeddings: List[float]):
        if not self.collection:
            raise Exception("Database connection is not established.")
        try:
            self.collection.insert_many(embeddings, ordered=False)
        except Exception as e:
            # Log the error
            print(f"Error inserting embeddings batch: {e}")
            raise
            
    
    def load_embeddings(self) -> Tensor:
        embeddings = []
        for doc in self.collection.find({}, {'_id': 0, 'embedding': 1}):
            embeddings.append(doc['embedding'])
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def find_element(self, idx: int) -> List:
        ans = self.collection.find_one({'_id': f"answer_{idx}"}, {'_id': 0, 'ans_list': 1})
        # Check if a document was found
        if ans:
            return ans['ans_list']
        else:
            # Handle the case where no document is found
            return None  
    
    def close_connection(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None

        