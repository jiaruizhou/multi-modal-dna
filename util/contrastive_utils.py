import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCE(torch.nn.Module):
    def __init__(self) -> None:
        super(InfoNCE,self).__init__()
    def forward(self,queries,labels):
        query = queries / queries.norm(p=2, dim=-1, keepdim=True)
        passage = queries /queries.norm(p=2, dim=-1, keepdim=True)
        sim_matrix = torch.mm(query, passage.t())
        label = (labels.unsqueeze(0) == labels.unsqueeze(0).t()).to(dtype=torch.float32) # torch.arange(bs, device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix, label)
        return loss

def contrastive_loss(logits: torch.Tensor, labels=None) -> torch.Tensor:
    if labels == None:
        return nn.functional.binary_cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    else:

        label = (labels.unsqueeze(0) == labels.unsqueeze(0).t()).to(dtype=torch.float32)
        return nn.functional.binary_cross_entropy_with_logits(logits, label)

        return nn.functional.cross_entropy(logits, label)

def clip_loss(logits_per_text, logits_per_image, labels):
    caption_loss = contrastive_loss(logits_per_text,labels)
    image_loss = contrastive_loss(logits_per_image,labels)
    return (caption_loss + image_loss) / 2.0

class CLIPLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPLoss,self).__init__()
        self.use_logit_scale = args.use_logit_scale
        if self.use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    # def cross_entropy(preds, targets, reduction='none'):
    #     log_softmax = nn.LogSoftmax(dim=-1)
    #     loss = (-targets * log_softmax(preds)).sum(1)
    #     if reduction == "none":
    #         return loss
    #     elif reduction == "mean":
    #         return loss.mean()
    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html

    

    
    def forward(self, image_embeds, text_embeds, labels ):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits
        if self.use_logit_scale:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 1/0.07
            # logit_scale = torch.tensor(1 / 0.07,requires_grad=True)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        
        loss = clip_loss(logits_per_text, logits_per_image, labels)

        return loss
        # # Getting Image and Text Features
        # # Calculating the Loss
        # logits = (text_embeddings @ image_embeddings.T) / self.temperature
        # images_similarity = image_embeddings @ image_embeddings.T
        # texts_similarity = text_embeddings @ text_embeddings.T
        # targets = F.softmax(
        #     (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        # )
        # texts_loss = self.cross_entropy(logits, targets, reduction='none')
        # images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        # loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # return loss.mean()


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.95):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        query = F.normalize(features, dim=-1)
        passage = F.normalize(features,dim=-1)
        sim = torch.mm(query,passage.t())

        label = (labels.unsqueeze(0) == labels.unsqueeze(0).t())
        distances = torch.mean(torch.cdist(query.unsqueeze(0).repeat(query.shape[0], 1, 1).float(), \
            passage.unsqueeze(1).repeat(1, passage.shape[0], 1).float(), p=2), dim=-1)
        whole_loss = ((label) * torch.pow(sim, 2) + ~label * torch.pow(torch.clamp(self.margin - sim, min=0.0), 2))
        
        loss_contrastive = torch.mean(whole_loss)
        return loss_contrastive

# class ContrastiveLoss(torch.nn.Module):

#     def __init__(self, margin=0.95):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, query, passage):
#         query = F.normalize(query, dim=-1)
#         passage = F.normalize(passage,dim=-1)
#         sim=torch.mm(query,passage.t())

#         label = torch.eye(query.shape[0]).float().to(query.device)
#         # distances = torch.mean(torch.cdist(query.unsqueeze(0).repeat(query.shape[0], 1, 1).float(), \
#         #     passage.unsqueeze(1).repeat(1, passage.shape[0], 1).float(), p=2), dim=-1)
#         whole_loss = ((label) * torch.pow(sim, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - sim, min=0.0), 2))
        
#         loss_contrastive = torch.mean(whole_loss)
#         return loss_contrastive