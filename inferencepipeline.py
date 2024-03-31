from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
import torch
from torch import Tensor

from typing import Tuple, List
from databasemanager import DatabaseManager


class InferencePipeline:
    def __init__(self, q_model: AutoModel, a_model: AutoModel, tokenizer: AutoTokenizer, device: torch.device = "cpu"):
        self.q_model = q_model
        self.a_model = a_model
        self.device = device 
        self.tokenizer = tokenizer 

    
    def embed_question(self, title: List[str], body: List[str], max_length: int = 512) -> BaseModelOutput:
        self.q_model.eval()
        with torch.no_grad():
            encoded_batch = self.tokenizer(
                text=title, text_pair=body,
                max_length=max_length, truncation=True,
                padding='max_length', return_tensors='pt' 
            )
        encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
        outputs = self.q_model(**encoded_batch)
        batch_embedding = outputs.last_hidden_state[:,0,:]
        return batch_embedding
    
    def inbatch_negative_sampling(self, Q: Tensor, P: Tensor) -> Tensor:
        S = (Q @ P.transpose(0,1)).to(self.device)
        return S


    def get_topk_indices(self, Q: Tensor, P: Tensor, k: int=None) -> Tuple[Tensor, Tensor]:
        S = self.inbatch_negative_sampling(Q, P)
        if k == None:
            k = len(S)
        scores, indices = torch.topk(S, k)

        return indices, scores

    
    def inference(self, title: List[str], body: List[str]) -> List[List[str]]:
        Q = self.embed_question(title, body)
        P = self.database_manager_embedding.load_embeddings().to(self.device)
        idx, scores = self.get_topk_indices(Q, P, k = 5)
        idx.squeeze_(0)
        return [self.database_manager_answer.find_element(ix) for ix in idx]
    
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