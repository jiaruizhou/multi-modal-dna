# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
import timm


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

    def forward_features(self, x):
        B = x.shape[0] # x.shape [B, 1, 32, 32]
        
        x = self.patch_embed(x) # x.shape [B, 64, 768]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768] stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1) # [B, 65, 768]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
        return x    
    
def vit_base_patch4_5mer(**kwargs):
    model = VisionTransformer(img_size=32, in_chans=1, patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
