import torch
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
import math
import torch.distributed as dist


def unwrap_ddp(model):
    if hasattr(model, "module"):
        return unwrap_ddp(model.module)
    else:
        return model


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



def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
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

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
            #  (fcgr,(token_id,token_ids))
        samples = to_device(samples, device, non_blocking=True)
        

        with torch.amp.autocast('cuda', enabled=args.amp):
            visual_feature, text_feature = model(samples)

        if args.cross_process_negatives:
            visual_feature = visual_feature.contiguous()
            text_feature = text_feature.contiguous()
            bs = args.batch_size
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                raise ValueError("cross_process_negatives requires torch.distributed to be initialized")
            visual_features = [torch.zeros_like(visual_feature) for _ in range(torch.distributed.get_world_size())]
            AllGather.apply(visual_features, visual_feature)
            visual_feature = torch.cat(visual_features, dim=0)
            text_features = [torch.zeros_like(text_feature) for _ in range(torch.distributed.get_world_size())]
            AllGather.apply(text_features, text_feature)
            text_feature = torch.cat(text_features, dim=0)

        loss = criterion(visual_feature,text_feature)
        visual_feature, text_feature = visual_feature.detach().cpu(), text_feature.detach().cpu()
        visual_feature = visual_feature / visual_feature.norm(p=2, dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        sim_matrix = torch.mm(visual_feature, text_feature.t())
        
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        ranks = torch.zeros_like(sim_matrix, dtype=torch.long)
        rank = ranks.scatter_(1, sorted_indices, torch.arange(sim_matrix.size(1)).expand_as(sim_matrix)).diag() + 1
        assert rank.dim() == 1
        avg_rank = rank.float().mean().item()

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
            log_writer.add_scalar('train/lr', max_lr, global_step)
            log_writer.add_scalar('train/rank', avg_rank, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(args, data_loader, model, criterion, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 40, header):
        samples = batch[0]
        
        to_device(samples, device)
        
        # compute output
        with torch.amp.autocast('cuda', enabled=args.amp):
            visual_feature, text_feature = model(samples)
            loss = criterion(visual_feature,text_feature)
        
        visual_feature, text_feature = visual_feature.detach().cpu(), text_feature.detach().cpu()

        visual_feature = visual_feature / visual_feature.norm(p=2, dim=-1, keepdim=True)
        visual_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        
        sim_matrix = visual_feature @ text_feature.t()

        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        ranks = torch.zeros_like(sim_matrix, dtype=torch.long)
        rank = ranks.scatter_(1, sorted_indices, torch.arange(sim_matrix.size(1)).expand_as(sim_matrix)).diag() + 1
        assert rank.dim() == 1
        avg_rank = rank.float().mean().item()
        
        metric_logger.meters['loss'].update(loss.item())
        metric_logger.meters['avg_rank'].update(avg_rank)
        

    metric_logger.synchronize_between_processes()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

