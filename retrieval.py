# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
from pathlib import Path

import torch
import torch.backends
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc

from models.model_contrastive import contrastive_dna_model
from models.model_fuse import build_bert_model, build_vit_model, build_mae_model
from contrastive_train import train_one_epoch, evaluate, to_device

from util.custom_datasets import Visual_Text_Dataset
from util.metrics import macro_average_precision, get_result_rank
import os

import csv
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
# torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--pred', action='store_true', help='Perform evaluation only')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # DNA data set
    parser.add_argument('--data', default="", type=str, help="the dataset name for tax")
    parser.add_argument('--kmer', default=5, type=int, help="the k-mer size of fcgr")  # input size 2**k x 2**k

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./contrastive_output/', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./contrastive_output/', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    
    parser.add_argument('--train_vit', action='store_true', help='train the vit')
    parser.add_argument('--train_bert', action='store_true', help='train the vit')
    parser.add_argument('--pooling_method', default='cls_output', choices=["cls_output", "pooler_output", "average_pooling"], help='the bert feature')

    # vit Model set
    parser.add_argument('--vit_model', default='vit_base_patch4_5mer', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--cls_num', default=5, help="the class number used for vit head")  #['superkingdom', 'kingdom', 'phylum', 'family']
    parser.add_argument('--vit_resume', default='./output/output_b_p4_5mer/checkpoint-540.pth', help='resume from vit checkpoint')
    parser.add_argument('--vit_embed_dim', default=768, type=int, help='Dimension of the vision transformer')

    # BERT Model paramerters
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--bert_resume', default='./bertax_pytorch', help='resume from bert checkpoint')
    parser.add_argument('--config_path', default='./bertax_pytorch/config.json', help='resume from bert checkpoint')
    parser.add_argument('--bert_embed_dim', default=250, type=int, help='Dimension of the vision transformer')
    parser.add_argument('--global_pooling_vit_bert', action='store_true', help='Use global pooled features of ViT and BERT')
    parser.add_argument('--global_pool_vit', action='store_true', help='Use global pooled features of ViT')
    parser.add_argument('--vit2bert_proj',action='store_true', help='do not use layernorm all mlp head')

    parser.add_argument('--model', default='fuse', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--retrieval_resume', default='', help='resume from fuse model checkpoint')

    parser.add_argument('--single_gpu', action='store_true', help='Use single GPU to train')
    parser.add_argument('--all_head_trunc', action='store_true', help='Use trunc norm for all mlp head')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cross_gpu_sample', action='store_true')
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--name', default='fuse',choices=['vit','bert','fuse','mae'], type=str, help='Name of model to train')
    parser.add_argument('--get_encode', action='store_true')
    parser.add_argument('--save_tmp_files', action='store_true')
    parser.add_argument('--mae_model_name', default="mae_vit_base_patch4_5mer")
    parser.add_argument('--test_ckp', default=10, type=int)
    return parser

def get_feature_vec(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_db = Visual_Text_Dataset(args, files=args.data_path, kmer=args.kmer, phase="train")
    dataset_q = Visual_Text_Dataset(args, files=args.data_path, kmer=args.kmer, phase="test")

    assert len(dataset_db) % misc.get_world_size() == 0
    assert len(dataset_q) % misc.get_world_size() == 0

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_db = torch.utils.data.DistributedSampler(dataset_db, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_q = torch.utils.data.DistributedSampler(dataset_q, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_db = torch.utils.data.DataLoader(
        dataset_db,
        sampler=sampler_db,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_q = torch.utils.data.DataLoader(dataset_q, sampler=sampler_q, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
    if args.name == 'fuse':
        model = contrastive_dna_model(args)
    elif args.name == 'bert':
        model = build_bert_model(args)
    elif args.name == 'vit':
        model = build_vit_model(args)
    elif args.name == 'mae':
        model = build_mae_model(args)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.name == 'fuse' or args.name == 'bert' or args.name == 'vit':
        misc.load_retrieval_model(args=args, model_without_ddp=model_without_ddp)
    elif args.name == 'mae':
        pass

    with torch.no_grad():
        if args.name == 'fuse':
            db_features, db_labels = get_vl_encoded(args, data_loader_db, model, device)
            q_features, q_labels = get_vl_encoded(args, data_loader_q, model, device)

        elif args.name == 'bert' or args.name == 'vit' or args.name == 'mae':
            db_features, db_labels = get_encoded(args, data_loader_db, model, device)
            q_features, q_labels = get_encoded(args, data_loader_q, model, device)
        if misc.is_main_process() and args.name == "fuse":
            print(f"[RANK] {misc.get_rank()}: The shape of {args.name} encoded feature DB len {len(db_features)} len sample[0] {len(db_features[0])} {db_features[0][0].shape} Q:{len(q_features[0])}")
        elif misc.is_main_process():
            print(f"[RANK] {misc.get_rank()}: The shape of {args.name} encoded feature DB {db_features.shape} Q:{q_features.shape}")

        misc.save_on_master(db_features,os.path.join(args.output_dir,f"dbvecs_{args.name}_ckp{args.test_ckp}.pt"))
        misc.save_on_master(db_labels,os.path.join(args.output_dir,f"dby_{args.name}_ckp{args.test_ckp}.pt"))
        misc.save_on_master(q_features,os.path.join(args.output_dir,f"qvecs_{args.name}_ckp{args.test_ckp}.pt"))
        misc.save_on_master(q_labels,os.path.join(args.output_dir,f"qy_{args.name}_ckp{args.test_ckp}.pt"))
    return db_features, db_labels, q_features, q_labels


class AllGather(torch.autograd.Function):
    """ 
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank] 
    
@torch.no_grad()
def get_gather_data(features):
    features = features.contiguous()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise ValueError("cross_process_negatives requires torch.distributed to be initialized")
    gathered_feature = [torch.zeros_like(features) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_feature, features)# AllGather.apply
    return torch.cat(gathered_feature, dim=0)

@torch.no_grad()
def get_encoded(args, data_loader, model, device):
    print(f"Get embedding from contrastive model on {args.data} with model {args.model}")
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    features = []
    labels_supk, labels_phyl, labels_genus = [], [], []
    i = 0
    for batch in metric_logger.log_every(data_loader, 40, header):     

        if args.name == "vit" or args.name == 'mae':
            samples = batch[0][0]
        elif args.name == "bert":
            samples = batch[0][1]

        to_device(samples, device)
        to_device(batch[1], device)
        label_supk, label_phyl, label_genus = batch[1][0], batch[1][1], batch[1][2]
        with torch.amp.autocast('cuda', enabled=args.amp):
            if args.name == "vit":
                feature = model(samples)
            elif args.name == "bert":
                feature = model(samples[0].squeeze(1), token_type_ids=samples[1].squeeze(1)).last_hidden_state[:, 0]
            elif args.name == "mae":
                _, _, _, feature = model(samples)
            if args.cross_gpu_sample:
                feature = get_gather_data(feature).detach().cpu()
                label_supk, label_phyl, label_genus = get_gather_data(label_supk).detach().cpu(), get_gather_data(label_phyl).detach().cpu(), get_gather_data(label_genus).detach().cpu()
            if misc.is_main_process():
                features.extend(feature)
                labels_supk.extend(label_supk)
                labels_phyl.extend(label_phyl)
                labels_genus.extend(label_genus)
    if misc.is_main_process():
        features = torch.stack(features)
        labels_supk, labels_phyl, labels_genus = torch.stack(labels_supk), torch.stack(labels_phyl), torch.stack(labels_genus)
    return  (features), (labels_supk, labels_phyl, labels_genus)
        
@torch.no_grad()
def get_vl_encoded(args, data_loader, model, device):
    print(f"Get embedding from contrastive model on {args.data} with model {args.model}")
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    visual_features, text_features = [], []
    labels_supk, labels_phyl, labels_genus = [], [], []
    i = 0
    for batch in metric_logger.log_every(data_loader, 40, header):
        samples = batch[0]
        to_device(samples, device)
        to_device(batch[1], device)

        label_supk, label_phyl, label_genus = batch[1][0], batch[1][1], batch[1][2]
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            visual_feature, text_feature = model(samples)
            if args.cross_gpu_sample:
                visual_feature, text_feature = get_gather_data(visual_feature).detach().cpu(), get_gather_data(text_feature).detach().cpu()
                label_supk, label_phyl, label_genus = get_gather_data(label_supk).detach().cpu(), get_gather_data(label_phyl).detach().cpu(), get_gather_data(label_genus).detach().cpu()
            if misc.is_main_process():
                visual_features.extend(visual_feature)
                text_features.extend(text_feature)
                labels_supk.extend(label_supk)
                labels_phyl.extend(label_phyl)
                labels_genus.extend(label_genus)  
    if misc.is_main_process():
        visual_features, text_features = torch.stack(visual_features), torch.stack(text_features)
        labels_supk, labels_phyl, labels_genus = torch.stack(labels_supk), torch.stack(labels_phyl), torch.stack(labels_genus)
    return  (visual_features, text_features), (labels_supk, labels_phyl, labels_genus)
    
def retreival(args, qvecs, dbvecs, q_label, db_label):
    topk_list=(1, 5, 10, 25)
    topk = max(topk_list)
    if not os.path.exists(os.path.join(args.output_dir,f'results_{args.name}_ckp{args.test_ckp}.pt')):

        batch_size = 600
        qvecs = F.normalize(qvecs, dim=-1)
        dbvecs = F.normalize(dbvecs, dim=-1)
        start_time = time.time()
        batch_size = min(batch_size, qvecs.size(0))
        if qvecs.size(0) % batch_size == 0:
            num = qvecs.size(0) // batch_size
        else:
            num = qvecs.size(0) // batch_size + 1
        print('>> Total iteration:{}'.format(num))

        results=[[],[],[]]
        with torch.no_grad():
            for i in tqdm(range(num),leave=False):
                start = i * batch_size
                end = min(start + batch_size, qvecs.size(0))
                query = qvecs[start:end]
                score = torch.einsum("bd,kd->bk", query, dbvecs)
                _, topk_indice = torch.topk(score, topk, dim=-1)

                results[0].extend(db_label[rank_dict['supk']][topk_indice])
                results[1].extend(db_label[rank_dict['phyl']][topk_indice])
                results[2].extend(db_label[rank_dict['genus']][topk_indice])

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('retrieval time {}'.format(total_time_str))
        results = torch.tensor(np.array(results))
        torch.save(results, os.path.join(args.output_dir,f'results_{args.name}_ckp{args.test_ckp}.pt'))
    else:
        results = torch.load(os.path.join(args.output_dir,f'results_{args.name}_ckp{args.test_ckp}.pt'))

    # result for rank k
    
    for tax_rank in list(rank_dict.keys()):
        acc = retreival_acc(results[rank_dict[tax_rank]], q_label[rank_dict[tax_rank]], tk=topk_list)
        print(f"ACC @ {topk_list} on {tax_rank}: {acc}")
        avep, aver, avef = [],[],[]
        results_rk = get_result_rank(results[rank_dict[tax_rank]], topk_list=topk_list)
        for i, topk_i in enumerate(topk_list):
            avepk, averk, avefk = macro_average_precision(results_rk[i], q_label[rank_dict[tax_rank]])
            avep.append(avepk.item())
            aver.append(averk.item())
            avef.append(avefk.item())
        print(f"avep aver avef @ {topk_i} on {tax_rank}: {avep} {aver} {avef}")
        log = [args.test_ckp, args.name, args.data, tax_rank, topk_list, acc, avep, aver, avef]
        write_log(args, log)

def retreival_acc(output, target, tk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(tk), output.size()[1])
    batch_size = target.size(0)
    pred = output.t()  # [topk,batch]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # torch.any(torch.eq(True, correct, dim=1))  #pred.eq(target.reshape(1, -1).expand_as(pred))
    return [(torch.any(correct[:min(k, maxk)], dim=0).float().sum(0) * 100. / batch_size).item() for k in tk]

def write_log(args, log):
    if os.path.exists(os.path.join(args.output_dir,"retrieval.csv")) is False:
        with open(os.path.join(args.output_dir,"retrieval.csv"), 'a+') as f:
            csv_write = csv.writer(f)
            csv_head = ["checkpoint", "model_name", "dataset", "tax_rank", "topk_list", "ACC", "AveP", "AveR", "AveF"]  # 
            csv_write.writerow(csv_head)

    with open(os.path.join(args.output_dir,"retrieval.csv"), 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(log)

rank_dict = {"supk": 0, "phyl": 1, "genus": 2}
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model,args.data)
    args.log_dir = args.output_dir
    args.data_path = os.path.join(args.data_path, args.data + "/")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.get_encode:
        db_features, db_labels, q_features, q_labels = get_feature_vec(args)
        exit(0)
    else:
        db_features = torch.load(os.path.join(args.output_dir,f"dbvecs_{args.name}_ckp{args.test_ckp}.pt"))
        db_labels = torch.load(os.path.join(args.output_dir,f"dby_{args.name}_ckp{args.test_ckp}.pt"))
        q_features = torch.load(os.path.join(args.output_dir,f"qvecs_{args.name}_ckp{args.test_ckp}.pt"))
        q_labels = torch.load(os.path.join(args.output_dir,f"qy_{args.name}_ckp{args.test_ckp}.pt"))
    if args.name == 'fuse':
        qvecs, dbvecs = torch.cat([q_features[0],q_features[1]],dim=-1), torch.cat([db_features[0],db_features[1]],dim=-1)
    elif args.name == 'bert' or args.name == 'vit' or args.name == 'mae':
        qvecs, dbvecs = q_features, db_features

    print(f"Start retrieval on {args.data}")
    retreival(args, qvecs, dbvecs, q_labels, db_labels)
