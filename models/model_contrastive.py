import torch
import torch.nn as nn
from models.model_fuse import build_bert_model,build_vit_model

class contrastive_dna_model(nn.Module):
    """
    Vision Language Transformer
    """
    def __init__(self, args):
        super(contrastive_dna_model,self).__init__()
        self.pooling_method = args.pooling_method
        self.vit2bert_proj = args.vit2bert_proj

        if self.vit2bert_proj:
            self.visual2text_proj = torch.nn.Linear(args.vit_embed_dim, args.bert_embed_dim)

        self.textmodel = build_bert_model(args)
        self.visualmodel = build_vit_model(args)
        
        if not args.train_bert:
            for parameter in self.textmodel.parameters():
                parameter.requires_grad_(False)
        if not args.train_vit:
            for parameter in self.visualmodel.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        visual_data = x[0]
        text_data = x[1]
        B = visual_data.shape[0]
        # visual backbone
        visual_src = self.visualmodel(visual_data) # [B, 65, 768]
        text_outputs = self.textmodel(text_data[0].squeeze(1), token_type_ids=text_data[1].squeeze(1)) # text_data.shape [B, 1, 502])
        text_src = text_outputs.last_hidden_state 

        visual_feature = visual_src  # global pool without cls token [B, vl_dim]
        
        if self.pooling_method == "cls_output":
            text_feature = text_src[:, 0]
        elif self.pooling_method == "pooler_output":
            text_feature = text_outputs.pooler_output # [B, vl_dim]
        elif self.pooling_method == "average_pooling":
            text_feature = text_src[:,1:].mean(dim=1) # TODO handle padding
        else:
            raise NotImplementedError(f"Unsupported pooling method: {self.pooling_method}")
        if self.vit2bert_proj:
            visual_feature = self.visual2text_proj(visual_feature)
            
        return visual_feature, text_feature
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # bert_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'LayerNorm' in n and 'textmodel' in n]
        # vit_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'norm' in n and 'viusalmodel' in n]
        # cls_token = ['visualmodel.cls_token']
        # pos_embed = ['textmodel.embeddings.position_embeddings.weight', 'visualmodel.pos_embed']
        # no_decay = bert_layer_norm_bias + vit_layer_norm_bias + pos_embed + cls_token
        no_decay = ['visualmodel.cls_token', 'visualmodel.pos_embed']
        return set(no_decay)
    