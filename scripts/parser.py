import argparse
import torchvision.models as models


def parser():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    _parser = argparse.ArgumentParser(description='PyTorch ImageNet Training PCL')
    _parser.add_argument('data', metavar='DIR',
                         help='path to dataset')
    _parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                         choices=model_names,
                         help='model architecture: ' +
                              ' | '.join(model_names) +
                              ' (default: resnet50)')
    _parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                         help='number of data loading workers (default: 4)')
    _parser.add_argument('--epochs', default=200, type=int, metavar='N',
                         help='number of total epochs to run')
    _parser.add_argument('--warmup-epoch', default=100, type=int,
                         help='number of warm-up epochs to only train with InfoNCE loss')
    _parser.add_argument('--exp-dir', default='experiment', type=str,
                         help='experiment directory')

    _parser.add_argument('-b', '--batch-size', default=8, type=int,
                         metavar='N',
                         help='mini-batch size (default: 8), this is the total '
                              'batch size of all GPUs on the current node when '
                              'using Data Parallel or Distributed Data Parallel')
    _parser.add_argument('-lr', '--learning-rate', default=0.03, type=float,
                         metavar='LR', help='initial learning rate', dest='lr')
    _parser.add_argument('--cos', action='store_true',
                         help='use cosine lr schedule')
    _parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                         help='learning rate schedule (when to drop lr by 10x)')

    _parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum of SGD solver')
    _parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)',
                         dest='weight_decay')
    _parser.add_argument('--low-dim', default=128, type=int,
                         help='feature dimension (default: 128)')
    _parser.add_argument('--num-cluster', default='20,25,30', type=str,
                         help='number of clusters')
    _parser.add_argument('--pcl-r', default=16, type=int,
                         help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: '
                              '16)')
    _parser.add_argument('--moco-m', default=0.999, type=float,
                         help='moco momentum of updating key encoder (default: 0.999)')
    _parser.add_argument('--temperature', default=0.2, type=float,
                         help='softmax temperature')

    _parser.add_argument('-p', '--print-freq', default=10, type=int,
                         metavar='N', help='print frequency (default: 10)')
    _parser.add_argument('--save-freq', default=10, type=int,
                         metavar='N', help='save frequency (default: 10)')

    _parser.add_argument('--world-size', default=1, type=int,
                         help='number of nodes for distributed training')
    _parser.add_argument('--rank', default=0, type=int,
                         help='node rank for distributed training')
    _parser.add_argument('--dist-url', default='tcp://172.0.0.1:23456', type=str,
                         help='url used to set up distributed training')
    _parser.add_argument('--dist-backend', default='nccl', type=str,
                         help='distributed backend')
    _parser.add_argument('--seed', default=None, type=int,
                         help='seed for initializing training. ')
    _parser.add_argument('--gpu', default=None, type=int,
                         help='GPU id to use.')
    _parser.add_argument('--multiprocessing-distributed', action='store_true',
                         help='Use multi-processing distributed training to launch '
                              'N processes per node, which has N GPUs. This is the '
                              'fastest way to use PyTorch for either single node or '
                              'multi node data parallel training')

    _parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
    _parser.add_argument('--resume', default='', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')

    _parser.add_argument('--mlp', action='store_true',
                         help='use mlp head')

    _parser.add_argument('--cost', type=str, default='0.5')
    _parser.add_argument('--n-run', type=int, default=1)
    _parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')

    return _parser
