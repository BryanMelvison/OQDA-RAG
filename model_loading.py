from transformers import AutoModel, AutoTokenizer
import transformers
import torch.nn as nn
import os
import torch
from collections import OrderedDict


# Helper Function
def enableMultiGPU(model: AutoModel, multi_gpu: bool) -> AutoModel:
    if multi_gpu:
        model = nn.DataParallel(model)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    return model

# To Load our weights to the model (since our model was trained on multiple gpus, we have to do string processing)
def load_model_from_gpu(model_path: str, model_name: str, device : torch.device = 'cpu') -> AutoModel:

    our_state_dict = torch.load(model_path, map_location = device)

    new_state_dict = OrderedDict()
    for k, v in our_state_dict.items():

        name = k[7:] if k.startswith('module.') else k  # Remove the 'module.' prefix
        new_state_dict[name] = v

    # Load the state dict into your model
    model = AutoModel.from_pretrained(model_name, state_dict=new_state_dict).to(device)
    return model

def load_model(model_path: str, model_name: str, device: torch.device = 'cpu') -> AutoModel:
    model =  transformers.AutoModel.from_pretrained(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    return model

def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name)