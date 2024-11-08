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
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.custom_datasets import Finetune_Dataset_All, Pred_Dataset
import ai4.multi_modal_dna.models.models_mae as models_mae

from engine_finetune import train_one_epoch, get_encoded


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir2', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir2', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--data', default="", type=str)
    parser.add_argument('--kmer', default=5, type=int)
    parser.add_argument('--tax_rank', default="phylum")  #['superkingdom', 'kingdom', 'phylum', 'family']

    return parser


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str("7")
    # os.environ["SLURM_PROCID"] = "1"
    if args.data == "all":
        args.data_path = ["./data/" + data_name + "/" for data_name in ["similar", "non_similar", "final"]]

    args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_db = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="train")
    dataset_q = Finetune_Dataset_All(args, files=args.data_path, kmer=args.kmer, phase="val")

    # dataset_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_db = torch.utils.data.SequentialSampler(dataset_db)
        sampler_q = torch.utils.data.SequentialSampler(dataset_q)

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
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    with torch.no_grad():
        db_features, db_labels = get_encoded(args, data_loader_db, model, device)
        q_features, q_labels = get_encoded(args, data_loader_q, model, device)
        torch.save(db_features, args.output_dir + "db_glb.npy")
        torch.save(db_labels, args.output_dir + "db_y_glb.npy")
        torch.save(q_features, args.output_dir + "q_glb.npy")
        torch.save(q_labels, args.output_dir + "q_y_glb.npy")


import torch.nn.functional as F


def retreival(args, qvecs, dbvecs, q_label, db_label, topk):
    batch_size = 200
    qvecs = F.normalize(qvecs, dim=-1).squeeze(1)
    dbvecs = F.normalize(dbvecs, dim=-1).squeeze(1)
    start_time = time.time()
    batch_size = min(batch_size, qvecs.size(0))
    if qvecs.size(0) % batch_size == 0:
        num = qvecs.size(0) // batch_size
    else:
        num = qvecs.size(0) // batch_size + 1
    print('>> Total iteration:{}'.format(num))

    total_acc1_1 = total_acc5_1 = total_acc1_2 = total_acc5_2 = total_acc1_3 = total_acc5_3 = total_acc10 = 0.0

    with torch.no_grad():
        for i in range(num):
            start = i * batch_size
            end = min(start + batch_size, qvecs.size(0))
            query = qvecs[start:end]
            score = torch.einsum("bd,kd->bk", query, dbvecs)
            _, topk_indice = torch.topk(score, topk, dim=-1)
            acc1_1, acc5_1 = retreival_acc(db_label[rank_dict['supk']][topk_indice], q_label[rank_dict['supk']][start:end], tk=(1, 5))
            total_acc1_1 += acc1_1
            total_acc5_1 += acc5_1
            acc1_2, acc5_2 = retreival_acc(db_label[rank_dict['phyl']][topk_indice], q_label[rank_dict['phyl']][start:end], tk=(1, 5))
            total_acc1_2 += acc1_2
            total_acc5_2 += acc5_2
            acc1_3, acc5_3 = retreival_acc(db_label[rank_dict['genus']][topk_indice], q_label[rank_dict['genus']][start:end], tk=(1, 5))
            total_acc1_3 += acc1_3
            total_acc5_3 += acc5_3
            if (i + 1) % 4000 == 0 or (i + 1) == qvecs.size(0):
                print('\r>>>> {}/{} done...'.format(i + 1, num), end='')
        # pdb.set_trace()
    print("Acc@1 {} Acc@5 {} on {}".format(total_acc1_1 / num, total_acc5_1 / num, "supk"))
    write_log(args, [args.kmer, args.data, "supk", total_acc1_1.item() / num, total_acc5_1.item() / num])

    print("Acc@1 {} Acc@5 {} on {}".format(total_acc1_2 / num, total_acc5_2 / num, "phyl"))
    write_log(args, [args.kmer, args.data, "phyl", total_acc1_2.item() / num, total_acc5_2.item() / num])
    print("Acc@1 {} Acc@5 {} on {}".format(total_acc1_3 / num, total_acc5_3 / num, "genus"))
    write_log(args, [args.kmer, args.data, "genus", total_acc1_3.item() / num, total_acc5_3.item() / num])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('retrieval time {}'.format(total_time_str))
    return total_acc1_1 / num, total_acc5_1 / num, total_acc1_2 / num, total_acc5_2 / num,\
        total_acc1_3 / num, total_acc5_3 / num


def retreival_acc(output, target, tk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(tk), output.size()[1])
    batch_size = target.size(0)
    pred = output.t()  # [topk,batch]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # torch.any(torch.eq(True, correct, dim=1))  #pred.eq(target.reshape(1, -1).expand_as(pred))
    return [torch.any(correct[:min(k, maxk)], dim=0).float().sum(0) * 100. / batch_size for k in tk]


def retreival_precision(output, target, tk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(tk), output.size()[1])
    batch_size = target.size(0)
    pred = output.t()  # [topk,batch]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # torch.any(torch.eq(True, correct, dim=1))  #pred.eq(target.reshape(1, -1).expand_as(pred))
    return [torch.any(correct[:min(k, maxk)], dim=0).float().sum(0) * 100. / batch_size for k in tk]


import os

import csv


def create_csv(args):
    path = "./output_dir2/retrieval.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ["kmer", "dataset", "tax_rank", "Acc@1", "Acc@5"]  # "AveP", "AveR", "AveF", "loss" "eval_loss", "eval_Acc", "test_loss",
        csv_write.writerow(data_row)


def write_log(args, log):
    if os.path.exists("./output_dir2/retrieval.csv") is False:
        create_csv(args)
    with open("./output_dir2/retrieval.csv", 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(log)


from util.tax_entry import supk_dict, phyl_dict, genus_dict, new_genus_dict, new_phyl_dict, new_supk_dict

rank_dict = {"supk": 0, "phyl": 1, "genus": 2}
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir += "/{}/".format(args.data)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.output_dir + "db_glb.npy"):
        main(args)
    dbvecs = torch.load(args.output_dir + "db_glb.npy")
    # db_label = torch.load(args.output_dir + "db_y_glb.npy")
    qvecs = torch.load(args.output_dir + "q_glb.npy")
    # q_label = torch.load(args.output_dir + "q_y_glb.npy")

    ##########

    db_label0 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/train_supk.npy".format(args.data))
    db_label1 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/train_phyl.npy".format(args.data))
    db_label2 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/train_genus.npy".format(args.data))

    supk_set = set(supk_dict.keys())
    genus_set = set(genus_dict.keys())
    phyl_set = set(phyl_dict.keys())
    db_label0 = [supk_dict['unknown'] if element not in supk_set else supk_dict[element] for element in db_label0]
    db_label1 = [phyl_dict['unknown'] if element not in phyl_set else phyl_dict[element] for element in db_label1]
    db_label2 = [genus_dict['unknown'] if element not in genus_set else genus_dict[element] for element in db_label2]

    db_label = torch.tensor([db_label0, db_label1, db_label2])
    # qvecs = torch.load(args.output_dir + "q.npy")
    qvecs = torch.from_numpy(np.array(qvecs))
    q_label0 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/test_supk.npy".format(args.data))
    q_label1 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/test_phyl.npy".format(args.data))
    q_label2 = torch.load("/data/zhoujr/AI-Micro/mae/data/{}/test_genus.npy".format(args.data))
    q_label0 = [supk_dict['unknown'] if element not in supk_set else supk_dict[element] for element in q_label0]
    q_label1 = [phyl_dict['unknown'] if element not in phyl_set else phyl_dict[element] for element in q_label1]
    q_label2 = [genus_dict['unknown'] if element not in genus_set else genus_dict[element] for element in q_label2]

    q_label = torch.tensor([q_label0, q_label1, q_label2])

    #########
    print("retrieval on {}".format(args.data))
    acc1, acc5, acc1_2, acc5_2, acc1_3, acc5_3 = retreival(args, qvecs, dbvecs, q_label, db_label, args.topk)
