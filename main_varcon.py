"""
DDP version of VarCon training.
"""

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms, datasets
from util import (
    TwoCropTransform,
    AverageMeter,
    adjust_learning_rate,
    warmup_learning_rate,
    set_optimizer,
    save_model,
    gather_tensors,
    gather_with_grad,
)
from networks.resnet_big import VarConResNet
from losses import VarConLoss
from torch.amp import autocast, GradScaler


def parse_option():
    parser = argparse.ArgumentParser('argument for VarCon training with DDP')

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'])
    parser.add_argument('--mean', type=str, help='mean for custom dataset')
    parser.add_argument('--std', type=str, help='std for custom dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/')
    parser.add_argument('--size', type=int, default=32)

    # Method (VarCon only in this repo)
    parser.add_argument('--method', type=str, default='VarCon',
                        choices=['VarCon'])

    # Temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature (tau1 for VarCon)')

    # VarCon specific
    parser.add_argument('--epsilon', type=float, default=0.02,
                        help='epsilon for VarCon adaptive temperature')

    # Training settings
    parser.add_argument('--cosine', action='store_true',
                        help='use cosine learning rate schedule')
    parser.add_argument('--syncBN', action='store_true',
                        help='use synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='use warmup learning rate')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # tau2 analysis
    parser.add_argument('--tau2_record_epochs', type=str, default='10,50,100,200',
                        help='epochs to record tau2 distribution')

    opt = parser.parse_args()

    # Parse lr_decay_epochs
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    # Parse tau2_record_epochs
    opt.tau2_record_epochs = set(int(e) for e in opt.tau2_record_epochs.split(','))

    # Set paths
    opt.model_path = './save/{}/{}_models'.format(opt.method, opt.dataset)
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.format(
        opt.method, opt.dataset, opt.model, opt.learning_rate,
        opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    opt.model_name += '_eps_{}'.format(opt.epsilon)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.batch_size > 256:
        opt.warm = True

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warmup_to = opt.learning_rate
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    return opt


def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=True,
            transform=TwoCropTransform(train_transform),
            download=True
        )
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=True,
            transform=TwoCropTransform(train_transform),
            download=True
        )
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(
            root=opt.data_folder,
            transform=TwoCropTransform(train_transform)
        )
    else:
        raise ValueError(opt.dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, train_sampler


def set_model(opt):
    """Build VarCon backbone and VarCon loss."""
    model = VarConResNet(name=opt.model)

    num_classes_map = {'cifar10': 10, 'cifar100': 100, 'path': 1000}
    num_classes = num_classes_map.get(opt.dataset, 1000)

    criterion = VarConLoss(
        num_classes=num_classes,
        feat_dim=128,
        temperature=opt.temp,
        epsilon=opt.epsilon,
        normalize=True,
    )

    return model, criterion


def train(train_loader, train_sampler, model, criterion, optimizer, epoch, opt, scaler):
    """One epoch of VarCon training with AMP + DDP."""
    model.train()
    train_sampler.set_epoch(epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # VarCon-specific meters
    kl_meter = AverageMeter()
    nll_meter = AverageMeter()
    tau2_meter = AverageMeter()
    conf_meter = AverageMeter()

    # Collect tau2 distribution
    all_tau2_this_epoch = []

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0).cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with autocast("cuda"):
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)

            # VarCon branch
            feat_all = torch.cat([f1, f2], dim=0)
            labels_all = torch.cat([labels, labels], dim=0)

            feat_all_gathered = gather_with_grad(feat_all)
            labels_all_gathered = gather_tensors(labels_all)

            loss_dict = criterion(feat_all_gathered, labels_all_gathered)
            loss = loss_dict['total_loss']

            # Update meters
            kl_meter.update(loss_dict['kl_div'].item(), bsz)
            nll_meter.update(loss_dict['nll'].item(), bsz)
            tau2_meter.update(loss_dict['avg_tau2'].item(), bsz)
            conf_meter.update(loss_dict['avg_confidence'].item(), bsz)

            # Collect tau2 distribution (only on rank 0)
            if epoch in opt.tau2_record_epochs and dist.get_rank() == 0:
                all_tau2_this_epoch.append(loss_dict['all_tau2s'].cpu().numpy())

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress (rank 0 only)
        if (idx + 1) % opt.print_freq == 0 and dist.get_rank() == 0:
            msg = (f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                   f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   f'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                   f'KL {kl_meter.val:.3f} NLL {nll_meter.val:.3f}'
                   f' tau2 {tau2_meter.val:.4f} conf {conf_meter.val:.4f}')
            print(msg)
            sys.stdout.flush()

    # Return results
    result = {'loss': losses.avg}
    result['kl_div'] = kl_meter.avg
    result['nll'] = nll_meter.avg
    result['avg_tau2'] = tau2_meter.avg
    result['avg_confidence'] = conf_meter.avg
    if all_tau2_this_epoch:
        result['tau2_distribution'] = np.concatenate(all_tau2_this_epoch)

    return result


def main():
    # Initialize distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        device_id=torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
    )
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    opt = parse_option()
    opt.local_rank = local_rank

    # Create save folder (rank 0 only)
    if dist.get_rank() == 0:
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

    # Wait for rank 0 to create folder
    dist.barrier()

    # Print config (rank 0 only)
    if dist.get_rank() == 0:
        print('=' * 60)
        print(f'Method: {opt.method}')
        print(f'Dataset: {opt.dataset}')
        print(f'Model: {opt.model}')
        print(f'Batch size per GPU: {opt.batch_size}')
        print(f'Total batch size: {opt.batch_size * dist.get_world_size()}')
        print(f'Epochs: {opt.epochs}')
        print(f'Learning rate: {opt.learning_rate}')
        print(f'Temperature (tau1): {opt.temp}')
        print(f'Epsilon: {opt.epsilon}')
        print(f'Save folder: {opt.save_folder}')
        print('=' * 60)

    # Build data loader
    train_loader, train_sampler = set_loader(opt)

    # Build model and criterion
    model, criterion = set_model(opt)

    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Build optimizer
    optimizer = set_optimizer(opt, model)

    # Mixed precision scaler
    scaler = GradScaler("cuda")

    # tau2 distribution storage
    tau2_distributions = {}

    # Training loop
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        time1 = time.time()
        result = train(
            train_loader, train_sampler, model, criterion, optimizer, epoch, opt, scaler
        )
        time2 = time.time()

        # Logging (rank 0 only)
        if dist.get_rank() == 0:
            msg = f'Epoch {epoch}, time {time2 - time1:.2f}s, loss {result["loss"]:.4f}'
            msg += (f', KL {result["kl_div"]:.4f}, NLL {result["nll"]:.4f}'
                    f', tau2 {result["avg_tau2"]:.4f}, conf {result["avg_confidence"]:.4f}'
                    f', eps {opt.epsilon:.4f}')
            print(msg)

            # Save tau2 distribution
            if 'tau2_distribution' in result:
                tau2_distributions[f'epoch_{epoch}'] = result['tau2_distribution']
                print(f"  Recorded tau2 distribution: {len(result['tau2_distribution'])} samples")

            # Save checkpoint
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
                save_model(model, optimizer, criterion, opt, epoch, save_file)

    # Save final model and tau2 distributions (rank 0 only)
    if dist.get_rank() == 0:
        save_file = os.path.join(opt.save_folder, 'last.pth')
        save_model(model, optimizer, criterion, opt, opt.epochs, save_file)

        if tau2_distributions:
            dist_path = os.path.join(opt.save_folder, 'tau2_distributions.npz')
            np.savez(dist_path, **tau2_distributions)
            print(f"Saved tau2 distributions to {dist_path}")

        print('Training complete!')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
