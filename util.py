# util.py
# This utility module is largely adapted from the official SupContrast implementation:
# https://github.com/HobbitLong/SupContrast

from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist

import json
import pathlib
from torch.utils.data import Subset


class TwoCropTransform:
    """Create two crops of the same image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top-k predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    """Standard step or cosine learning rate schedule."""
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """Linear warm-up of learning rate over the first warm_epochs."""
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    """Create an SGD optimizer for the model parameters."""
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, criterion, opt, epoch, save_file):
    """Save training state to a checkpoint file."""
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def gather_tensors(tensor):
    """
    All-gather helper (fully detached).

    If torch.distributed is not initialized, returns the input tensor unchanged.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    output = torch.cat(tensor_list, dim=0)
    return output


def gather_with_grad(local_feats):
    """
    All-gather that preserves gradients for the local slice.

    Output shape: [world_size * local_bsz, feat_dim].
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_feats

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
    dist.all_gather(gathered, local_feats.detach())
    gathered[rank] = local_feats

    return torch.cat(gathered, dim=0)


def load_class_subset(dataset, subset_json):
    """Return a torchvision dataset restricted to indices listed in subset_json."""
    if subset_json is None:
        return dataset
    idx = json.loads(pathlib.Path(subset_json).read_text())
    return Subset(dataset, idx)
