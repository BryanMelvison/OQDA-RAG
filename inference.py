import random
import torch
import os
from dotenv import load_dotenv
import model_loading
from inferencepipeline import InferencePipeline
from databasemanager import DatabaseManager

load_dotenv()

# For Reproducibility
random.seed(2024)
torch.manual_seed(2024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Environment Variables
multi_gpu = os.getenv("multi_gpu")
a_name = os.getenv("a_name")
q_name = os.getenv("q_name")
t_name = os.getenv("t_name")
a_path = os.getenv("a_path")
q_path = os.getenv("q_path")
database_embedding = os.getenv("DATABASE_NAME")
collection_embedding = os.getenv("COLLECTION_NAME")

database_answer = os.getenv("DATABASE_NAME_1")
collection_answer = os.getenv("COLLECTION_NAME_1")

# Load Models and Tokenizers
# load_a = model_loading.load_model(a_path, a_name , device )
# load_q = model_loading.load_model(q_path, q_name, device )

load_a = model_loading.load_model_from_gpu(a_path, a_name, device)
load_q = model_loading.load_model_from_gpu(q_path, q_name, device)

a_enc = model_loading.enableMultiGPU(load_a, multi_gpu)
q_enc = model_loading.enableMultiGPU(load_q, multi_gpu)

tokenizer = model_loading.load_tokenizer(t_name)

inference = InferencePipeline(q_enc, a_enc, tokenizer, device)
database_manager_embedding = DatabaseManager(database_embedding, collection_embedding)
database_manager_answer = DatabaseManager(database_answer, collection_answer)

inference.connection_db(database_manager_embedding, database_manager_answer)
title3 = "Roasting a Juicy Turkey"
body3 = "Thanksgiving is coming up and I'm in charge of the turkey this year. I always worry it will be too dry. What's the best way to roast a turkey so it stays moist and flavorful?"
result = inference.inference([title3], [body3])
print(result)

inference.disconnect_db()