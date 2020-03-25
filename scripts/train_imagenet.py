r"""
Training script for torchvision based ResNet-50 on ImageNet dataset. This is
adopted from official PyTorch examples with minimal modifications:

https://github.com/pytorch/examples/tree/master/imagenet/main.py

Key modifications (NO CHANGE IN HYPERPARAMETERS)::

1. Add option to train on partial ImageNet dataset (semi-supervised setup) for
   our data efficiency experiments.
   - We use our own torchvision-wrapped ImageNet dataset class for this.
   - We use albumentations for transformation because we use our dataset class.

2. Only support single node multi-GPU training (typically 8 GPUs), one GPU per
   process. Restrict GPU usage by setting ``CUDA_VISIBLE_DEVICES``.

3. Use custom logging (consistent with rest of our codebase) for less clutter.

4. Blackify and thin out code (remove code not specific to ResNet-50 for brevity).
"""

import argparse
import os
import random
import shutil

import albumentations as alb
from loguru import logger
import torch
from torch import nn, optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from viswsl.data.datasets.downstream_datasets import ImageNetDataset
import viswsl.utils.distributed as vdist
from viswsl.utils.metrics import TopkAccuracy
from viswsl.utils.timer import Timer

# fmt: off
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR",
                    help="path to dataset")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=90, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=256, type=int,
                    metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float,
                    metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                    metavar="W", help="weight decay (default: 1e-4)",
                    dest="weight_decay")
parser.add_argument("--log-every", default=10, type=int,
                    metavar="N", help="Log after every these many batches.")
parser.add_argument("--resume", default="", type=str, metavar="PATH",
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="use pre-trained model")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:2345", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--seed", default=0, type=int,
                    help="seed for initializing training. ")
parser.add_argument("--serialization-dir", default="/tmp/imagenet_train",
                    help="Path to serialize checkpoints during training.")
parser.add_argument("--data-percentage", default=100, type=float,
                    help="Percentage of data to train on.")
# fmt: on

best_acc1 = 0

# Counter for logging events to tensorboard with appropriate timestep.
GLOBAL_ITER = 0


def main():
    _A = parser.parse_args()
    random.seed(_A.seed)
    torch.manual_seed(_A.seed)
    cudnn.deterministic = True

    _A.world_size = torch.cuda.device_count()
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=_A.world_size, args=(_A.world_size, _A))


def main_worker(gpu, ngpus_per_node, _A):
    global best_acc1
    _A.gpu = gpu
    logger.info(f"Use GPU: {_A.gpu} for training")

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    _A.rank = _A.gpu
    dist.init_process_group(
        backend=_A.dist_backend,
        init_method=_A.dist_url,
        world_size=_A.world_size,
        rank=_A.rank,
    )
    # Create model (pretrained or random init).
    model = models.resnet50(pretrained=True) if _A.pretrained else models.resnet50()

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(_A.gpu)
    model.cuda(_A.gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    _A.batch_size = int(_A.batch_size / ngpus_per_node)
    _A.workers = int((_A.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[_A.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(_A.gpu)

    optimizer = optim.SGD(
        model.parameters(), _A.lr, momentum=_A.momentum, weight_decay=_A.weight_decay
    )

    # optionally resume from a checkpoint
    if _A.resume:
        if os.path.isfile(_A.resume):
            logger.info(f"=> loading checkpoint '{_A.resume}'")

            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(_A.resume, map_location=f"cuda:{_A.gpu}")
            _A.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]

            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(_A.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                f"=> loaded checkpoint '{_A.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logger.info(f"=> no checkpoint found at '{_A.resume}'")

    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # We modify the data loading code to use our ImageNet dataset class and
    # transforms from albumentations (however, transformation steps are same).
    # -------------------------------------------------------------------------
    train_dataset = ImageNetDataset(
        root=_A.data, split="train", percentage=_A.data_percentage
    )
    logger.info(f"Size of dataset: {len(train_dataset)}")
    val_dataset = ImageNetDataset(root=_A.data, split="val")
    # Val dataset is used sparsely, don't keep it around in memory by caching.

    normalize = alb.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=1.0,
        always_apply=True,
    )
    # Override image transform (class definition has transform according to
    # downstream linear classification protocol).
    # fmt: off
    train_dataset.image_transform = alb.Compose([
        alb.RandomResizedCrop(224, 224, always_apply=True),
        alb.HorizontalFlip(p=0.5),
        alb.ToFloat(max_value=255.0, always_apply=True),
        normalize,
    ])
    val_dataset.image_transform = alb.Compose([
        alb.Resize(256, 256, always_apply=True),
        alb.CenterCrop(224, 224, always_apply=True),
        alb.ToFloat(max_value=255.0, always_apply=True),
        normalize,
    ])
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=_A.batch_size,
        num_workers=_A.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_A.batch_size,
        num_workers=_A.workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    # fmt: on
    # -------------------------------------------------------------------------

    # Keep track of time per iteration and ETA.
    timer = Timer(start_from=0, total_iterations=_A.epochs * len(train_loader))

    writer = SummaryWriter(log_dir=_A.serialization_dir)
    for epoch in range(_A.start_epoch, _A.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, _A)

        train(train_loader, model, criterion, optimizer, epoch, timer, writer, _A)
        acc1 = validate(val_loader, model, criterion, writer, _A)

        # Remember best top-1 accuracy and save checkpoint.
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if vdist.is_master_process():
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                _A.serialization_dir,
            )


def train(train_loader, model, criterion, optimizer, epoch, timer, writer, _A):
    global GLOBAL_ITER
    model.train()

    # A tensor to accumulate loss for logging (and have smooth training curve).
    # Reset to zero every `_A.log_every` iteration.
    train_loss: torch.Tensor = torch.tensor(0.0).cuda(_A.gpu)

    for i, batch in enumerate(train_loader):
        timer.tic()
        images, target = batch["image"], batch["label"]

        images = images.cuda(_A.gpu, non_blocking=True)
        target = target.cuda(_A.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        train_loss += loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        timer.toc()
        GLOBAL_ITER += 1

        if i % _A.log_every == 0:
            train_loss /= _A.log_every
            vdist.average_across_processes(train_loss)

            if _A.rank == 0:
                logger.info(
                    f"Epoch: [{epoch}] | {timer.stats} | Loss: {train_loss:.3f}"
                )
                writer.add_scalar("loss/train", train_loss, GLOBAL_ITER)
            train_loss = torch.zeros_like(train_loss)


def validate(val_loader, model, criterion, writer, _A):
    global GLOBAL_ITER
    top1 = TopkAccuracy(top_k=1)
    top5 = TopkAccuracy(top_k=5)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, target = batch["image"], batch["label"]
            images = images.cuda(_A.gpu, non_blocking=True)
            target = target.cuda(_A.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # Accumulate accuracies for current batch.
            top1(output, target)
            top5(output, target)

        # Average Top-1 and Top-5 accuracies across all processes.
        top1_avg = torch.tensor(top1.get_metric(reset=True)).cuda(_A.gpu)
        top5_avg = torch.tensor(top5.get_metric(reset=True)).cuda(_A.gpu)

        vdist.average_across_processes(top1_avg)
        vdist.average_across_processes(top5_avg)

        writer.add_scalar("metrics/top1", top1_avg, GLOBAL_ITER)
        writer.add_scalar("metrics/top5", top5_avg, GLOBAL_ITER)

        logger.info(f"Acc@1 {top1_avg:.3f} Acc@5 {top5_avg:.3f}")
    return top1_avg


def save_checkpoint(state, is_best: bool, serialization_dir: str):
    global GLOBAL_ITER
    filename = os.path.join(serialization_dir, f"checkpoint_{GLOBAL_ITER}.pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(serialization_dir, "best.pth"))


def adjust_learning_rate(optimizer, epoch, _A):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = _A.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
