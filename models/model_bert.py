from transformers import (
    BertConfig,
    BertModel,
    logging,
)
import torch.nn as nn
import torch

class Bert_Model(nn.Module):
    def __init__(self,config_path : str, bert_ckp_path : str):
        super(Bert_Model,self).__init__()
        config = BertConfig.from_json_file(config_path)
        self.bert = BertModel(config)
    
        self.bert.load_state_dict(torch.load(bert_ckp_path))
        
    def forward(self,token_ids,token_type_ids):

        outputs = self.bert(token_ids,token_type_ids)
        last_hidden_states = outputs.last_hidden_state  # [B, 502, 250]
        pooler_output = outputs.pooler_output
        return last_hidden_states, pooler_output
