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
import timm.optim.optim_factory as optim_factory
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import random
from pathlib import Path

import torch
import torch.backends
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# assert timm.__version__ == "0.3.2"  # version check
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc


from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.model_contrastive import class_finetuning

from fuse_finetune import train_one_epoch, evaluate

from util.custom_datasets import Visual_Text_Dataset
from util.tax_entry import annotate_predictions
from torch.utils.data import random_split

import os

# torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N', help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    # parser.add_argument('--global_pool', action='store_true')
    # parser.set_defaults(global_pool=False)
    
    
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
    
    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')
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

    # vit Model set
    parser.add_argument('--vit_model', default='vit_base_patch4_5mer', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--cls_num', default=5, help="the class number used for vit head")  #['superkingdom', 'kingdom', 'phylum', 'family']
    parser.add_argument('--vit_resume', default='./output/output_b_p4_5mer/checkpoint-540.pth', help='resume from vit checkpoint')
    parser.add_argument('--vit_embed_dim', default=768, type=int, help='Dimension of the vision transformer')

    # BERT Model paramerters
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--bert_resume', default='./bertax_pytorch', help='resume from bert checkpoint')
    parser.add_argument('--config_path', default='./bertax_pytorch/config.json', help='resume from bert checkpoint')
    parser.add_argument('--bert_embed_dim', default=250, type=int, help='Dimension of the vision transformer')


    # pooling
    # parser.add_argument('--classfication_global_pool', action='store_true')
    # parser.set_defaults(classfication_global_pool=True)
    parser.add_argument('--global_pooling_vit_bert', action='store_true', help='Use global pooled features of ViT and BERT')
    parser.add_argument('--global_pool_vit', action='store_true', help='Use global pooled features of ViT')
    # for vl model
    parser.add_argument('--model', default='fuse', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--resume', default='', help='resume from fuse model checkpoint')

    parser.add_argument('--train_vit', action='store_true', help='train the vit')
    parser.add_argument('--train_bert', action='store_true', help='train the vit')
    parser.add_argument('--pooling_method', default='cls_output', choices=["cls_output", "pooler_output", "average_pooling"], help='the bert feature')
    parser.add_argument('--visual_token_nums', default=65, type=int)
    parser.add_argument('--text_token_nums', default=502, type=int)
    parser.add_argument('--vl_hidden_dim', default=256, type=int, help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dropout', default=0.1, type=int, help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int, help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=1, type=int, help='Number of encoders in the vision-language transformer')
    parser.add_argument('--single_gpu', action='store_true', help='Use single GPU to train')
    parser.add_argument('--all_head_trunc', action='store_true', help='Use trunc norm for all mlp head')

    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--loss_scale', action='store_true', help='Enable loss scaling')
    parser.add_argument('--vit2bert_proj',action='store_true', help='do not use layernorm all mlp head')
    parser.add_argument('--no_norm_before_head',action='store_true', help='do not use layernorm all mlp head')

    parser.add_argument('--contrastive_resume', default='', help='resume from fuse model checkpoint')
    parser.add_argument('--train_contrastive', action='store_true', help='train the contrastive model')

    return parser

def main(args):
    if not args.pred:
        args.data_path = os.path.join(args.data_path, args.data + "/")

    args.output_dir = os.path.join(args.output_dir, args.model,args.data)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # We do not add RANK to seed for correct data spliting in distributed training.
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not args.eval:
        dataset = Visual_Text_Dataset(args, files=args.data_path, kmer=args.kmer, phase="train")

        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size

        # 使用 random_split 进行分割
        dataset_train, _ = random_split(dataset, [train_size, val_size])
        # todo change the datasets
        dataset_val = Visual_Text_Dataset(args, files=args.data_path, kmer=args.kmer, phase="test")

    if args.eval:
        dataset_val = Visual_Text_Dataset(args, files=args.data_path, kmer=args.kmer, phase="test")
    
    fuse_model = class_finetuning(args)
    fuse_model.to(device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if not args.eval:
            print("dataset {} for {} is ready".format(args.data_path, args.kmer))
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.nb_classes)
    model_without_ddp = fuse_model
    n_parameters = sum(p.numel() for p in fuse_model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        fuse_model = torch.nn.parallel.DistributedDataParallel(fuse_model, device_ids=[args.gpu])
        model_without_ddp = fuse_model.module


    if args.contrastive_resume:
        param_groups = lrd.contrastive_param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
    else:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
 
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler(loss_scale=args.loss_scale)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    
    
    if args.eval:
        test_stats = evaluate(args, data_loader_val, fuse_model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images for superkingdom: {test_stats['acc1_0']:.1f}%")
        print(f"Accuracy of the network on the {len(dataset_val)} test images for phylum: {test_stats['acc1_1']:.1f}%")
        print(f"Accuracy of the network on the {len(dataset_val)} test images for genus: {test_stats['acc1_2']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(fuse_model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)

        if args.output_dir:

            misc.save_model(args=args, model=fuse_model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(args, data_loader_val, fuse_model, device)

        if log_writer is not None:
            for k in test_stats:
                log_writer.add_scalar(f'perf/test_{k}', test_stats[k], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    assert (args.amp and args.loss_scale) or (not args.amp and not args.loss_scale), "Mixed precision training requires loss scaling"
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
