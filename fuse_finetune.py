# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import csv
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from util.metrics import accuracy, macro_average_precision, compute_roc,plot_roc, plot_pr_curve, micro_average_precision, weighted_macro_average_precision, weighted_micro_average_precision
import util.misc as misc
import util.lr_sched as lr_sched
# from torchmetrics import Accuracy
import os
import numpy as np
import torch.nn.functional as F

def create_csv(csv_path):
    
    with open(csv_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ["kmer", "dataset", "tax_rank", "Acc@1", "Acc@5", "MacroAveP", "MicroAveP", "Weighted_macro_avep", "Weighted_micro_avep", "AveR", "AveF", "loss", "fpr", "tpr", "roc_auc"]  # "eval_loss", "eval_Acc", "test_loss",
        csv_write.writerow(data_row)

def write_log(output_dir, log):

    csv_path = os.path.join(output_dir , "test_result.csv")

    if os.path.exists(csv_path) is False:
        create_csv(csv_path)
    with open(csv_path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(log)

def to_device(x, device, non_blocking=True):
    if isinstance(x, torch.Tensor):
        x = x.to(device, non_blocking=non_blocking)
    elif isinstance(x, list) or isinstance(x, tuple):
        for i in range(len(x)):
            x[i] = to_device(x[i], device, non_blocking=non_blocking)
    elif isinstance(x, dict):
        for k in x:
            x[k] = to_device(x[k], device, non_blocking=non_blocking)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")
    return x

def has_nan(x):
    if isinstance(x, torch.Tensor):
        return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))
    elif isinstance(x, list) or isinstance(x, tuple):
        return any([has_nan(xx) for xx in x])
    elif isinstance(x, dict):
        return any([has_nan(v) for v in x.values()])
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def unwrap_ddp(model):
    if hasattr(model, "module"):
        return unwrap_ddp(model.module)
    else:
        return model

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    args=None):
    model.train(True)
    # if not args.train_contrastive:
    #     unwrap_ddp(model).encode.eval()
    # else:
    if not args.train_bert:
        unwrap_ddp(model).textmodel.eval()
    if not args.train_vit:
        unwrap_ddp(model).visualmodel.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
            #  (fcgr,(token_id,token_ids))
        samples = to_device(samples, device, non_blocking=True)
        targets1, targets2, targets3 = targets
        targets1, targets2, targets3 = targets1.to(device, non_blocking=True), targets2.to(device, non_blocking=True), targets3.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda', enabled=args.amp):
            outputs1, outputs2, outputs3 = model(samples)
            loss1 = criterion(outputs1, targets1)
            loss2 = criterion(outputs2, targets2)
            loss3 = criterion(outputs3, targets3)
            loss = (loss1 + loss2 + loss3) / 3

        loss1_value = loss1.item()
        loss1_value_reduce = misc.all_reduce_mean(loss1_value)
        loss2_value = loss2.item()
        loss2_value_reduce = misc.all_reduce_mean(loss2_value)
        loss3_value = loss3.item()
        loss3_value_reduce = misc.all_reduce_mean(loss3_value)

        loss_value = loss.item()
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if not math.isfinite(loss_value_reduce):
            print("[Warning] Loss is {}, skipped.".format(loss_value))
            continue
        if not args.single_gpu:
            torch.distributed.barrier()

        loss /= accum_iter
        # with torch.autograd.detect_anomaly():
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            global_step = epoch * len(data_loader) // accum_iter + data_iter_step // accum_iter
            log_writer.add_scalar('train/loss', loss_value_reduce, global_step)
            log_writer.add_scalar('train/loss1', loss1_value_reduce, global_step)
            log_writer.add_scalar('train/loss2', loss2_value_reduce, global_step)
            log_writer.add_scalar('train/loss3', loss3_value_reduce, global_step)
            log_writer.add_scalar('train/lr', max_lr, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(args, data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    outputs = [[], [], []]
    targets = [[], [], []]
    for batch in metric_logger.log_every(data_loader, 40, header):
        samples = batch[0]
        target = batch[1]
        to_device(samples, device)
            
        target1, target2, target3 = target
        targets[0].extend(target1)
        targets[1].extend(target2)
        targets[2].extend(target3)
        target1, target2, target3 = target1.to(device, non_blocking=True), \
            target2.to(device, non_blocking=True), target3.to(device, non_blocking=True)
        # compute output
        with torch.amp.autocast('cuda', enabled=args.amp):
            output = model(samples)
            outputs[0].extend(output[0])
            outputs[1].extend(output[1])
            outputs[2].extend(output[2])
            loss1 = criterion(output[0], target1)
            loss2 = criterion(output[1], target2)
            loss3 = criterion(output[2], target3)
            loss = (criterion(output[0], target1)+ \
                criterion(output[1], target2)+criterion(output[2], target3))/3
            
        metric_logger.update(loss=loss.item())
        metric_logger.meters['loss0'].update(loss1.item())
        metric_logger.meters['loss1'].update(loss2.item())
        metric_logger.meters['loss2'].update(loss3.item())

    metric_logger.synchronize_between_processes()

    rank_dicts = {"supk": 0, "phyl": 1, "genus": 2}
    # classes = [5, 44, 156]
    for i, rank in enumerate(rank_dicts.keys()):
        print("*********** For {} ***********".format(rank))
        y_pred = torch.stack(outputs[i]).cpu()
        y_true = torch.stack(targets[i]).cpu()
        acc1, acc5 = accuracy(y_pred, y_true, topk=(1, 5))

        y_pred = F.softmax(y_pred.clone().detach().float(), dim=-1).numpy()
        y_pred_max = np.argmax(y_pred, axis=-1)
        # if args.eval:
        #     if "lr" in args.resume:
        #         torch.save(targets,os.path.join(args.output_dir,'lr_targets.pt'))
        #         torch.save(outputs,os.path.join(args.output_dir,'lr_outputs.pt'))
        #     else:
        #         torch.save(targets,os.path.join(args.output_dir,'targets.pt'))
        #         torch.save(outputs,os.path.join(args.output_dir,'outputs.pt'))
        # bert no prtrain np.float64(77.09313560207956), np.float64(74.10604592475511), np.float64(75.10317073198426)
        # 去掉unknown vit no prertain(np.float64(83.9320768509343), np.float64(80.38444245297221), np.float64(81.62940612553942))
        # no pretrain (np.float64(81.75344249722707), np.float64(76.07827878568477), np.float64(77.99760910057618))
        macro_avep, aver, avef = macro_average_precision(y_pred_max, y_true)
        
        micro_avep = micro_average_precision(y_pred_max, y_true)
        weighted_macro = weighted_macro_average_precision(y_pred_max, y_true)
        weighted_micro = weighted_micro_average_precision(y_pred_max, y_true)
        
        # ACC = Accuracy(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        print(" RESULTS acc1:{} acc5:{} \nmacro_avep:{} micro_avep:{} weighted macro_avep:{} weighted micro_avep:{} \naver:{} avef:{}".format(acc1, acc5, macro_avep, micro_avep, weighted_macro, weighted_micro, aver, avef))
        # if args.eval:
        #     new_macro = new_macro_average_precision(torch.stack(outputs[i]).cpu(), torch.stack(targets[i]).cpu())
        #     print("average_precision_score  ", new_macro)
        if args.eval:
            # plot_roc(args, rank, torch.stack(targets[i]).cpu(), torch.stack(outputs[i]).cpu(), all_curves=True)
            # plot_pr_curve(args, rank, torch.stack(targets[i]).cpu(), torch.stack(outputs[i]).cpu())
            fpr, tpr, roc_auc = compute_roc(args,y_pred,y_true)
        # plot_roc(args, rank, torch.stack(targets[i]).cpu(), torch.stack(outputs[i]).cpu(), all_curves=True)
        # plot_pr_curve(args, rank, torch.stack(targets[i]).cpu(), torch.stack(outputs[i]).cpu())
            log = [args.kmer, args.data, rank, acc1.item(), acc5.item(), macro_avep, micro_avep, weighted_macro, weighted_micro, aver, avef, metric_logger.__getattr__('loss{}'.format(i)).global_avg,roc_auc['micro']]

            # log = [args.kmer, args.data, rank, acc1.item(), acc5.item(), macro_avep, micro_avep, weighted_macro, weighted_micro, aver, avef, metric_logger.__getattr__('loss{}'.format(i)).global_avg]
            write_log(args.output_dir, log)

        metric_logger.meters['avep{}'.format(i)].update(macro_avep)
        metric_logger.meters['aver{}'.format(i)].update(aver)
        metric_logger.meters['avef{}'.format(i)].update(avef)
        metric_logger.meters['acc1_{}'.format(i)].update(acc1.item())
        metric_logger.meters['acc5_{}'.format(i)].update(acc5.item())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def hook(args, data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Hook:'
    model.eval()

    visual_features = []
    text_features = []
    fused_features = []
    lambda_features = []
    target1, target2, target3 = [], [], []

    # input_block = []
    fmap_block = []
    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        # input_block.append(data_input.detach().cpu())

    model.textmodel.register_forward_hook(forward_hook)
    model.visualmodel.register_forward_hook(forward_hook)
    model.gate_fuse.register_forward_hook(forward_hook)
    model.gate_fuse.wz.register_forward_hook(forward_hook)
    

    for batch in metric_logger.log_every(data_loader, 40, header):
        samples = batch[0]
        target = batch[1]
        to_device(samples, device)
            
        target1.extend(target[0])
        target2.extend(target[1])
        target3.extend(target[2])
        

        # target1, target2, target3 = target1.to(device, non_blocking=True), \
            # target2.to(device, non_blocking=True), target3.to(device, non_blocking=True)
        # compute output
        with torch.amp.autocast('cuda', enabled=args.amp):
            _ = model(samples)

            if args.pooling_method == "cls_output":
                text_feature = fmap_block[1].last_hidden_state[:, 0].detach().cpu()
            elif args.pooling_method == "pooler_output":
                text_feature = fmap_block[1].last_hidden_state.pooler_output.detach().cpu() # [B, vl_dim]
            elif args.pooling_method == "average_pooling":
                text_feature = fmap_block[1].last_hidden_state[:,1:].mean(dim=1).detach().cpu() # TODO handle padding


            visual_features.extend(fmap_block[0].detach().cpu())
            text_features.extend(text_feature)
            fused_features.extend(fmap_block[2].detach().cpu())
            lambda_features.extend(fmap_block[3].sigmoid().cpu())
            fmap_block.pop()
            fmap_block.pop()
            fmap_block.pop()
            fmap_block.pop()

    visual_features = torch.stack(visual_features)
    text_features = torch.stack(text_features)
    fused_features = torch.stack(fused_features)
    lambda_features = torch.stack(lambda_features)

    torch.save(visual_features,os.path.join(args.output_dir,'visual_features.pt'))
    torch.save(text_features,os.path.join(args.output_dir,'text_features.pt'))
    torch.save(fused_features,os.path.join(args.output_dir,'fused_features.pt'))
    torch.save(lambda_features,os.path.join(args.output_dir,'lambda_features.pt'))
    torch.save([target1,target2,target3],os.path.join(args.output_dir,'targets.pt'))



torch.no_grad()
def evaluate_time(args, data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    outputs = [[], [], []]
    targets = [[], [], []]

    with torch.cuda.amp.autocast():
        import time
        start_time = time.perf_counter()
        for batch in metric_logger.log_every(data_loader, 40, header):
            samples = batch[0]
            target = batch[1]
            to_device(samples, device)
                
            target1, target2, target3 = target
            target1, target2, target3 = target1.to(device, non_blocking=True), \
                target2.to(device, non_blocking=True), target3.to(device, non_blocking=True)
            # compute output
            with torch.amp.autocast('cuda', enabled=args.amp):
                output = model(samples)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")