
import torch
import torch.nn as nn
import copy

import models.models_vit as models_vit
import models.models_mae as models_mae
from transformers import BertModel, BertConfig
from timm.layers import trunc_normal_
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
        self.pooling_method = args.pooling_method
        self.global_pooling_vit_bert = args.global_pooling_vit_bert
        self.hidden_dim = args.vl_hidden_dim
        self.vit2bert_proj = args.vit2bert_proj
        self.gated_fuse = args.gated_fuse

        self.textmodel = build_bert_model(args)
        self.visualmodel = build_vit_model(args)
        
        if not args.train_bert:
            for parameter in self.textmodel.parameters():
                parameter.requires_grad_(False)
        if not args.train_vit:
            for parameter in self.visualmodel.parameters():
                parameter.requires_grad_(False)

        if not self.global_pooling_vit_bert:
            num_total = args.text_token_nums + args.visual_token_nums + 1 # 502 + 65 +1
            self.vl_pos_embed = nn.Embedding(num_total, self.hidden_dim)
            
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

            self.visual_proj = nn.Linear(args.vit_embed_dim, args.vl_hidden_dim) # 768 -> 256
            self.text_proj = nn.Linear(args.bert_embed_dim, args.vl_hidden_dim) # 250 -> 256

            self.vl_model = build_vl_transformer(args)
            fused_feat_dim = args.vl_hidden_dim
        else:
            if self.gated_fuse:
                if args.gate_num == 0:
                    self.gate_fuse = Gate(args.vit_embed_dim, args.bert_embed_dim)
                elif args.gate_num == 1:
                    self.gate_fuse = Gate1(args.vit_embed_dim, args.bert_embed_dim)
                elif args.gate_num == 2:
                    self.gate_fuse = Gate2(args.vit_embed_dim, args.bert_embed_dim)
                elif args.gate_num == 3:
                    self.gate_fuse = Gate3(args.vit_embed_dim, args.bert_embed_dim)
                elif args.gate_num == 4:
                    self.gate_fuse = Gate4(args.vit_embed_dim, args.bert_embed_dim)
                fused_feat_dim = args.vit_embed_dim
            elif self.vit2bert_proj:
                self.visual2text_proj = torch.nn.Linear(args.vit_embed_dim, args.bert_embed_dim)
                fused_feat_dim = 2 * args.bert_embed_dim
            else:
                fused_feat_dim = args.bert_embed_dim + args.vit_embed_dim

        self.mlp = Classification_Head(fused_feat_dim, 
                                       all_head_trunc=args.all_head_trunc,
                                       no_norm_before_head=args.no_norm_before_head)


    def forward(self, x):
        visual_data = x[0]
        text_data = x[1]
        B = visual_data.shape[0]
        # visual backbone
        visual_src = self.visualmodel(visual_data) # [B, 65, 768]
        text_outputs = self.textmodel(text_data[0].squeeze(1), token_type_ids=text_data[1].squeeze(1)) # text_data.shape [B, 1, 502])
        text_src = text_outputs.last_hidden_state 

        if not self.global_pooling_vit_bert:
                        
            visual_feature = self.visual_proj(visual_src) # [B, 65, vl_dim]
            text_feature = self.text_proj(text_src) # [B, 502, vl_dim]

            cls_tokens = self.cls_token.expand(B, -1, -1)  # # [B, 1, vl_dim] stole cls_tokens impl from Phil Wang, thanks
            vl_pos = self.vl_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, total_num, vl_dim]
            vl_src = torch.cat([cls_tokens, visual_feature, text_feature], dim=1) # [B, total_num, vl_dim] [B, 568, 256]
            vl_src = vl_src + vl_pos
            
            vl_feature = self.vl_model(vl_src)

        else:
            visual_feature = visual_src  # global pool without cls token [B, vl_dim]
            
            if self.pooling_method == "cls_output":
                text_feature = text_src[:, 0]
            elif self.pooling_method == "pooler_output":
                text_feature = text_outputs.pooler_output # [B, vl_dim]
            elif self.pooling_method == "average_pooling":
                text_feature = text_src[:,1:].mean(dim=1) # TODO handle padding
            else:
                raise NotImplementedError(f"Unsupported pooling method: {self.pooling_method}")
            if self.gated_fuse:
                vl_feature = self.gate_fuse(visual_feature, text_feature)
            else:
                if self.vit2bert_proj:
                    visual_feature = self.visual2text_proj(visual_feature)
                vl_feature = torch.cat([visual_feature,text_feature],dim=-1)
        
        outputs = self.mlp(vl_feature)
        return outputs
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # bert_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'LayerNorm' in n and 'textmodel' in n]
        # vit_layer_norm_bias = [n for n in self.named_parameters() if "bias" in n and 'norm' in n and 'viusalmodel' in n]
        # cls_token = ['visualmodel.cls_token']
        # pos_embed = ['textmodel.embeddings.position_embeddings.weight', 'visualmodel.pos_embed']
        # no_decay = bert_layer_norm_bias + vit_layer_norm_bias + pos_embed + cls_token
        no_decay = ['visualmodel.cls_token', 'visualmodel.pos_embed']
        return set(no_decay)


class Gate(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.proj = nn.Linear(x2_dim, x1_dim)
        self.wz = nn.Linear(x1_dim * 2, x1_dim)
    
    def forward(self, x1, x2):
        x2 = self.proj(x2)
        gate_lambda = self.wz(torch.cat([x1, x2], dim=1)).sigmoid()
        out = x1 + gate_lambda * x2
        return out


class Gate1(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.proj = nn.Linear(x2_dim, x1_dim)
        self.wz = nn.Linear(x1_dim * 2, x1_dim)
    
    def forward(self, x1, x2):
        x2 = self.proj(x2)
        gate_lambda = self.wz(torch.cat([x1, x2], dim=1)).sigmoid()
        out = (1 - gate_lambda) * x1 + gate_lambda * x2
        return out


class Gate2(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.proj = nn.Linear(x2_dim, x1_dim)
        self.wz = nn.Linear(x1_dim * 2, 1)
    
    def forward(self, x1, x2):
        x2 = self.proj(x2)
        gate_lambda = self.wz(torch.cat([x1, x2], dim=1)).sigmoid()
        out = x1 + gate_lambda * x2
        return out


class Gate3(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.proj = nn.Linear(x2_dim, x1_dim)
        self.wz = nn.Linear(x1_dim, x1_dim)
    
    def forward(self, x1, x2):
        x2 = self.proj(x2)
        gate_lambda = self.wz(x1).sigmoid()
        out = x1 + gate_lambda * x2
        return out


class Gate4(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.proj = nn.Linear(x2_dim, x1_dim)
        self.wz = nn.Linear(x1_dim, 1)
    
    def forward(self, x1, x2):
        x2 = self.proj(x2)
        gate_lambda = self.wz(x1).sigmoid()
        out = x1 + gate_lambda * x2
        return out


class Classification_Head(nn.Module):
    def __init__(self, vl_dim, all_head_trunc, no_norm_before_head):
        super(Classification_Head,self).__init__()
        self.no_norm_before_head = no_norm_before_head
        self.head = torch.nn.Linear(vl_dim, 5)
        self.head2 = torch.nn.Linear(self.head.in_features + self.head.out_features, 44)
        self.head3 = torch.nn.Linear(self.head2.in_features + self.head2.out_features, 156)

        if not self.no_norm_before_head:
            self.fc_norm = nn.LayerNorm(self.head.in_features)
            self.fc_norm2 = nn.LayerNorm(self.head2.in_features)
            self.fc_norm3 = nn.LayerNorm(self.head3.in_features)

        trunc_normal_(self.head.weight, std=2e-5)
        if all_head_trunc:
            trunc_normal_(self.head2.weight, std=2e-5)
            trunc_normal_(self.head3.weight, std=2e-5)
    
    def forward(self,src):

        out1 = x1 = src
        if not self.no_norm_before_head:
            out1 = self.fc_norm(x1)  # 
        outcome1 = self.head(out1)  # ([Batch, 5])

        out2 = x2 = torch.cat([x1, outcome1], dim=-1)  # 
        if not self.no_norm_before_head:
            out2 = self.fc_norm2(x2)
        outcome2 = self.head2(out2)  # ([Batch, 44])

        out3 = x3 = torch.cat([x2, outcome2], dim=-1)  # 
        if not self.no_norm_before_head:
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
    if args.bert_resume:
        bert = BertModel.from_pretrained(args.bert_resume,add_pooling_layer=(args.pooling_method=="pooler_output"))
    else:
        config = BertConfig.from_json_file("./bertax_pytorch/config.json")
        bert = BertModel(config, add_pooling_layer=(args.pooling_method == "pooler_output"))
    return bert


def build_vit_model(args):

    vit_model = models_vit.__dict__[args.vit_model](
        num_classes=args.cls_num,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool_vit
    )

    misc.load_vit_model(args=args, model_without_ddp=vit_model)

    del vit_model.head
    del vit_model.norm

    return vit_model


def build_mae_model(args):

    mae_model = models_mae.__dict__[args.mae_model_name]()

    misc.load_mae_model(args=args, model_without_ddp=mae_model)

    return mae_model
