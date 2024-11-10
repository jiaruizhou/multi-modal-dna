
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
import timm
import copy
import models.models_vit as models_vit
from transformers import BertModel

import util.misc as misc

class vl_encoder(nn.Module):
    """ Vision Language Transformer
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(vl_encoder,self).__init__()
        # if True:
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        return self.encoder(src)



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder,self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output

class fuse_dna_model(nn.Module):
    """ Vision Language Transformer
    """

    def __init__(self, args):
        super(fuse_dna_model,self).__init__()
        
        self.global_pooling_vit_bert = args.global_pooling_vit_bert
        self.hidden_dim = args.vl_hidden_dim

        self.textmodel = build_bert_model(args)
        self.visualmodel = build_vit_model(args)
        


        if not args.train_bert:
            for parameter in self.textmodel.parameters():
                parameter.requires_grad_(False)
        # else:
            # del self.textmodel.bert.pooler.dense
        
        
        if not args.train_vit:
            for parameter in self.visualmodel.parameters():
                parameter.requires_grad_(False)
        else:
            del self.visualmodel.head
            del self.visualmodel.norm

        if not self.global_pooling_vit_bert:
            num_total = args.text_token_nums + args.visual_token_nums + 1 # 502 + 65 +1
            self.vl_pos_embed = nn.Embedding(num_total, self.hidden_dim)
            
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

            self.visual_proj = nn.Linear(args.vit_embed_dim, args.vl_hidden_dim) # 768 -> 256
            self.text_proj = nn.Linear(args.bert_embed_dim, args.vl_hidden_dim) # 250 -> 256

            self.vl_model = build_vl_transformer(args)

            self.mlp = Classification_Head(args)
        else:
            self.mlp = Classification_Head(args)

    def forward(self, x):
        visual_data = x[0]
        text_data = x[1]
        B = visual_data.shape[0]
        # visual backbone
        visual_src = self.visualmodel(visual_data) # [B, 65, 768]
        text_outputs = self.textmodel(text_data[0].squeeze(1), token_type_ids=text_data[1].squeeze(1)) # text_data.shape [B, 1, 502])
        text_src, pooler_text_src = text_outputs.last_hidden_state, text_outputs.pooler_output

        if not self.global_pooling_vit_bert:
                        
            visual_feature = self.visual_proj(visual_src) # [B, 65, vl_dim]
            text_feature = self.text_proj(text_src) # [B, 502, vl_dim]

            cls_tokens = self.cls_token.expand(B, -1, -1)  # # [B, 1, vl_dim] stole cls_tokens impl from Phil Wang, thanks
            vl_pos = self.vl_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, total_num, vl_dim]
            vl_src = torch.cat([cls_tokens, visual_feature, text_feature], dim=1) # [B, total_num, vl_dim] [B, 568, 256]
            vl_src = vl_src + vl_pos
            
            vl_feature = self.vl_model(vl_src)
            
            outputs = self.mlp(vl_feature)
        else:
            visual_feature = visual_src  # global pool without cls token [B, vl_dim]
            text_feature = pooler_text_src # [B, vl_dim]
            vl_feature = torch.cat([visual_feature,text_feature],dim=-1)
            outputs = self.mlp(vl_feature)
        return outputs
    
    @torch.jit.ignore
    def no_weight_decay(self):
        bert_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'LayerNorm' in n and 'textmodel' in n]
        vit_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'norm' in n and 'viusalmodel' in n]
        cls_token = ['cls_token', 'visualmodel.cls_token']
        pos_embed = ['textmodel.bert.embeddings.position_embeddings.weight', 'visualmodel.pos_embed', 'vl_pos_embed.weight']
        no_decay = bert_layer_norm_bias + vit_layer_norm_bias + pos_embed + cls_token
        return set(no_decay)

class Classification_Head(nn.Module):
    def __init__(self,args):
        super(Classification_Head,self).__init__()
        
        self.global_pooling_vit_bert =  args.global_pooling_vit_bert
        if not self.global_pooling_vit_bert:
            
            self.head = torch.nn.Linear(args.vl_hidden_dim, 5)
            self.head2 = torch.nn.Linear(self.head.in_features + self.head.out_features, 44)
            self.head3 = torch.nn.Linear(self.head2.in_features + self.head2.out_features, 156)
        else:
            self.head = torch.nn.Linear(args.bert_embed_dim + args.vit_embed_dim, 5)
            self.head2 = torch.nn.Linear(self.head.in_features + self.head.out_features, 44)
            self.head3 = torch.nn.Linear(self.head2.in_features + self.head2.out_features, 156)

        
        self.fc_norm = nn.LayerNorm(self.head.in_features)
        self.fc_norm2 = nn.LayerNorm(self.head2.in_features)
        self.fc_norm3 = nn.LayerNorm(self.head3.in_features)

    
    def forward(self,src):

        x1 = src
        out1 = self.fc_norm(x1)  # 
        outcome1 = self.head(out1)  # ([Batch, 5])
        x2 = torch.cat([x1, outcome1], dim=-1)  # 
        out2 = self.fc_norm2(x2)
        outcome2 = self.head2(out2)  # ([Batch, 44])
        x3 = torch.cat([x2, outcome2], dim=-1)  # 
        out3 = self.fc_norm3(x3)
        outcome3 = self.head3(out3)  # ([Batch, 156])

        return (outcome1, outcome2, outcome3)


def build_vl_transformer(args):
    return vl_encoder(
        d_model=args.vl_hidden_dim,
        dropout=args.vl_dropout,
        nhead=args.vl_nheads,
        dim_feedforward=args.vl_dim_feedforward,
        num_encoder_layers=args.vl_enc_layers,
    )

def build_bert_model(args):
    return BertModel.from_pretrained(args.bert_resume)


def build_vit_model(args):

    vit_model = models_vit.__dict__[args.vit_model](
        num_classes=args.cls_num,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool_vit
    )

    misc.load_vit_model(args=args, model_without_ddp=vit_model)

    return vit_model
