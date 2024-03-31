from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from databasemanager import DatabaseManager

class DatabasePipeline:
    def __init__(self, q_model: AutoModel, a_model: AutoModel, tokenizer: AutoTokenizer, answer_loader: DataLoader, device: torch.device = "cpu"):
        self.q_model = q_model
        self.a_model = a_model
        self.device = device 
        self.tokenizer = tokenizer 
        self.answer_loader = answer_loader
    
    # Embed the entire answer corpus, and post them to a MongoDB database.
    def embed_passage(self, max_length: int = 512):
        self.a_model.eval()
        # Process answers in batches
        global_idx = 0
        for answers in self.answer_loader:
            with torch.no_grad():
                encoded_batch = self.tokenizer(
                    text = answers,
                    max_length = max_length,
                    truncation = True,
                    padding="max_length",
                    return_tensors = 'pt'
                )
                encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
                outputs = self.a_model(**encoded_batch)
                batch_embedding = outputs.last_hidden_state[:,0,:]

                #Insert all the output in a batch into the DataBase
                embeddings_list = []
                for embedding in batch_embedding:
                    embedding_id = global_idx
                    embedding_list = embedding.cpu().numpy().tolist()
                    document = {
                        '_id': embedding_id,
                        'embedding': embedding_list
                    }
                    embeddings_list.append(document)
                    global_idx += 1
                self.database_manager_embedding.insert_embeddings_batch(embeddings_list)
            

    def insertanswersdb(self):
        index = 0
        for ans in self.answer_loader:
            ans_list = []
            for a in ans:
                ans_id = index
                ans_lst = a
                document = {
                    '_id': f"answer_{ans_id}",
                    'ans_list': ans_lst
                }
                ans_list.append(document)
                index += 1
            self.database_manager_answer.insert_embeddings_batch(ans_list)    
    
    def connection_db(self, database_manager_embedding: DatabaseManager, database_manager_answer: DatabaseManager):
        self.database_manager_embedding = database_manager_embedding
        self.database_manager_answer = database_manager_answer

        self.database_manager_embedding.establish_connection()
        self.database_manager_answer.establish_connection()


    def disconnect_db(self):
        try:
            self.database_manager_embedding.close_connection()
            self.database_manager_answer.close_connection()
        except NameError as e:
            print(f"An error occurred: {e}") 