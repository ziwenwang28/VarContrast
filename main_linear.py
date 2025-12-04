"""
DDP version of main_linear.py with global Top-1 and Top-5 accuracy.
Launch with:
  torchrun --nproc_per_node=4 main_linear_ddp.py --ckpt <path> ...
"""

from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms, datasets

from util import AverageMeter, adjust_learning_rate, warmup_learning_rate
from networks.resnet_big import VarConResNet, LinearClassifier


def make_linear_optimizer(opt, model):
    """Return SGD (default) or RMSProp optimizer for the linear classifier."""
    if opt.linear_opt == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=0.9,
            weight_decay=0,
            alpha=0.99,
        )
    else:  # 'sgd'
        return torch.optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=0.9,
            weight_decay=0,
        )


def parse_option():
    parser = argparse.ArgumentParser('Linear evaluation with DDP')

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_folder', type=str, default='./datasets/')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)

    # model
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to pretrained encoder checkpoint')

    # schedule
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--warm', action='store_true')

    # optimizer choice
    parser.add_argument('--linear_opt', default='sgd',
                        choices=['sgd', 'rmsprop'])

    opt = parser.parse_args()

    # parse lr decay epochs
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    # number of classes
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100

    # warm-up configuration
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def set_loader(opt):
    """Create train/val loaders with DistributedSampler."""
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=True,
            transform=train_transform,
            download=True,
        )
        val_dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=False,
            transform=val_transform,
            download=True,
        )
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=True,
            transform=train_transform,
            download=True,
        )
        val_dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=False,
            transform=val_transform,
            download=True,
        )

    # DDP samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader, train_sampler, val_loader, val_sampler


def set_model(opt, local_rank):
    """Load pretrained encoder and create linear classifier."""
    model = VarConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # load pretrained weights
    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    # handle "module." prefix from DDP checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        new_state_dict[new_k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    if local_rank == 0:
        print(f"Loaded pretrained encoder from: {opt.ckpt}")
        print(f"Load message: {msg}")

    # move to GPU
    model = model.cuda(local_rank)
    classifier = classifier.cuda(local_rank)
    criterion = criterion.cuda(local_rank)

    # wrap encoder and classifier with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)

    return model, classifier, criterion


def compute_global_topk_correct(output, labels, topk=(1, 5)):
    """
    Compute global top-k accuracy across all GPUs.
    Returns accuracy percentages.
    """
    maxk = max(topk)
    bsz = labels.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    correct_1 = correct[:1].reshape(-1).float().sum(0)
    correct_5 = correct[:5].reshape(-1).float().sum(0)

    # all-reduce correct counts
    dist.all_reduce(correct_1, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_5, op=dist.ReduceOp.SUM)

    bsz_tensor = torch.tensor([bsz], dtype=torch.float, device=labels.device)
    dist.all_reduce(bsz_tensor, op=dist.ReduceOp.SUM)

    acc1 = correct_1 / bsz_tensor * 100.0
    acc5 = correct_5 / bsz_tensor * 100.0

    return acc1.item(), acc5.item()


def train(train_loader, train_sampler, model, classifier, criterion,
          optimizer, epoch, opt, scaler, local_rank):
    """Train linear classifier for one epoch."""
    model.eval()       # freeze encoder
    classifier.train()

    # shuffle per epoch for DistributedSampler
    train_sampler.set_epoch(epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)
        bsz = labels.size(0)

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # forward: encoder frozen, only classifier trains
        with autocast("cuda"):
            with torch.no_grad():
                feats = model.module.encoder(images)
            output = classifier(feats.detach())
            loss = criterion(output, labels)

        # global accuracy
        acc1, acc5 = compute_global_topk_correct(output, labels, topk=(1, 5))

        losses.update(loss.item(), bsz)
        top1_meter.update(acc1, bsz)
        top5_meter.update(acc5, bsz)

        # backward and optimizer step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0 and local_rank == 0:
            print(
                f'Train: [{epoch}][{idx+1}/{len(train_loader)}]\t'
                f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                f'Acc@1 {top1_meter.val:.2f} ({top1_meter.avg:.2f})\t'
                f'Acc@5 {top5_meter.val:.2f} ({top5_meter.avg:.2f})'
            )
            sys.stdout.flush()

    return losses.avg, top1_meter.avg


def validate(val_loader, model, classifier, criterion, opt, local_rank):
    """Validation with global top-1/top-5 accuracy."""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)
            bsz = labels.size(0)

            with autocast("cuda"):
                feats = model.module.encoder(images)
                output = classifier(feats)
                loss = criterion(output, labels)

            acc1, acc5 = compute_global_topk_correct(output, labels, topk=(1, 5))

            losses.update(loss.item(), bsz)
            top1_meter.update(acc1, bsz)
            top5_meter.update(acc5, bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and local_rank == 0:
                print(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc@1 {top1_meter.val:.2f} ({top1_meter.avg:.2f})\t'
                    f'Acc@5 {top5_meter.val:.2f} ({top5_meter.avg:.2f})'
                )

    if local_rank == 0:
        print(f' * Acc@1 {top1_meter.avg:.2f}  Acc@5 {top5_meter.avg:.2f}')

    return losses.avg, top1_meter.avg


def main():
    opt = parse_option()

    # DDP init
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True

    if local_rank == 0:
        print('=' * 60)
        print('Linear Evaluation (DDP)')
        print(f'Dataset: {opt.dataset}')
        print(f'Model: {opt.model}')
        print(f'Checkpoint: {opt.ckpt}')
        print(f'Batch size per GPU: {opt.batch_size}')
        print(f'Total batch size: {opt.batch_size * dist.get_world_size()}')
        print(f'Learning rate: {opt.learning_rate}')
        print(f'Optimizer: {opt.linear_opt}')
        print(f'Epochs: {opt.epochs}')
        print('=' * 60)

    # data
    train_loader, train_sampler, val_loader, val_sampler = set_loader(opt)

    # model
    model, classifier, criterion = set_model(opt, local_rank)

    # optimizer (only classifier parameters)
    optimizer = make_linear_optimizer(opt, classifier)

    scaler = GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(1, opt.epochs + 1):
        # learning rate schedule
        if opt.linear_opt == 'rmsprop':
            # exponential decay (gamma = 0.98)
            gamma = 0.98
            lr = opt.learning_rate * (gamma ** (epoch - 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            adjust_learning_rate(opt, optimizer, epoch)

        time1 = time.time()
        train_loss, train_acc = train(
            train_loader,
            train_sampler,
            model,
            classifier,
            criterion,
            optimizer,
            epoch,
            opt,
            scaler,
            local_rank,
        )
        time2 = time.time()

        if local_rank == 0:
            print(f'Epoch {epoch}, time {time2-time1:.2f}s, train_acc {train_acc:.2f}')

        val_loss, val_acc = validate(
            val_loader,
            model,
            classifier,
            criterion,
            opt,
            local_rank,
        )

        if val_acc > best_acc:
            best_acc = val_acc

    if local_rank == 0:
        print('=' * 60)
        print(f'Best Validation Accuracy: {best_acc:.2f}%')
        print('=' * 60)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
