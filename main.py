import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from scripts.parser import parser
from scripts.meter import AverageMeter, ProgressMeter
import scripts.augmentation as aug
import scripts.momentum as momentum
import scripts.clustering as clustering
import scripts.loss as loss_script
from scripts.data_process import data_process
import pcl.builder
import pcl.loader


def worker_loader():
    args = parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_cluster = args.num_cluster.split(',')

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        worker(args.gpu, ngpus_per_node, args)


def worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.multiprocessing_distributed = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass():
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> create model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = True

    # Data loading code
    pre_train_dir = os.path.join(args.data, 'pre_train')
    train_dir = os.path.join(args.data, 'train')
    eval_dir = os.path.join(args.data, 'train')

    # center-crop augmentation
    eval_augmentation = aug.moco_eval()

    pre_train_dataset = pcl.loader.PreImager(pre_train_dir, eval_augmentation)

    train_dataset = pcl.loader.ImageFolderInstance(
        train_dir,
        pcl.loader.TwoCropsTransform(eval_augmentation))
    eval_dataset = pcl.loader.ImageFolderInstance(
        eval_dir,
        eval_augmentation)

    if args.distributed:
        pre_train_sampler = torch.utils.data.distributed.DistributedSampler(pre_train_dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        pre_train_sampler = None
        train_sampler = None
        eval_sampler = None

    if args.batch_size//pre_train_dataset.class_number < 2:
        raise NotImplementedError("Batch size must above double number of classes.")

    pre_train_loader = torch.utils.data.DataLoader(
        pre_train_dataset,
        batch_size=args.batch_size//pre_train_dataset.class_number,
        shuffle=(pre_train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=pre_train_sampler,
        drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size//2, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)

    print("=> Pre-train")
    # main loop
    for epoch in range(args.start_epoch, args.epochs):

        cluster_result = None
        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = momentum.compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

            if args.gpu == 0:
                features[
                    torch.norm(features, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                features = features.numpy()
                cluster_result = clustering.run_kmeans(features, args)  # run kmeans clustering on master node
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))

            dist.barrier()
            # broadcast clustering result
            for _k, data_list in cluster_result.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if epoch >= args.warmup_epoch:
            train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)
        else:
            train(pre_train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch + 1) % args.save_freq == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                               and args.rank % ngpus_per_node == 0)):
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, '{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        im_q, im_k = data_process(cluster_result, images, args.gpu)

        # compute output
        output, target, output_proto, target_proto = model(im_q=im_q, im_k=im_k,
                                                           cluster_result=cluster_result, index=index)

        loss = loss_script.proto_with_quality(output, target, output_proto, target_proto, criterion, acc_proto, images,
                                              args.num_cluster)


        losses.update(loss.item(), images[0].size(0))
        acc = loss_script.accuracy(output, target)[0]
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    worker_loader()
