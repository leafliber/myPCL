from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import random
import numpy as np

from torchvision import transforms, datasets
import torchvision.models as models

from sklearn.svm import LinearSVC

from scripts.parser import parser
import scripts.augmentation as aug
import pcl.loader

def calculate_ap(rec, prec):
    """
    Computes the AP under the precision recall curve.
    """
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def get_precision_recall(targets, preds):
    """
    [P, R, score, ap] = get_precision_recall(targets, preds)
    Input    :
        targets  : number of occurrences of this class in the ith image
        preds    : score for this image
    Output   :
        P, R   : precision and recall
        score  : score which corresponds to the particular precision and recall
        ap     : average precision
    """
    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack((
        targets[:, np.newaxis].astype(np.float64),
        preds[:, np.newaxis].astype(np.float64)
    ))
    ind = np.argsort(preds)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap


def main():
    args = parser().parse_args()

    if not args.seed is None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    eval_augmentation = aug.moco_eval()

    pre_train_dir = os.path.join(args.data, 'pre_train')
    eval_dir = os.path.join(args.data, 'eval')

    train_dataset = pcl.loader.PreImager(pre_train_dir, eval_augmentation)
    val_dataset = pcl.loader.ImageFolderInstance(
        eval_dir,
        eval_augmentation)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=128)

    # load from pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            # rename pre-trained keys
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
            model.fc = torch.nn.Identity()
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda()
    model.eval()

    test_feats = []
    test_labels = []
    print('==> calculate test features')
    for idx, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        feat = model(images)
        feat = feat.detach().cpu()
        test_feats.append(feat)
        test_labels.append(target)

    test_feats = torch.cat(test_feats, 0).numpy()
    test_labels = torch.cat(test_labels, 0).numpy()

    test_feats_norm = np.linalg.norm(test_feats, axis=1)
    test_feats = test_feats / (test_feats_norm + 1e-5)[:, np.newaxis]

    result = {}

    k_list = ['full']

    for k in k_list:
        cost_list = args.cost.split(',')
        result_k = np.zeros(len(cost_list))
        for i, cost in enumerate(cost_list):
            cost = float(cost)
            avg_map = []
            for run in range(args.n_run):
                print(len(train_dataset))

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

                train_feats = []
                train_labels = []
                print('==> calculate train features')
                for idx, (images, target) in enumerate(train_loader):
                    images = images.cuda(non_blocking=True)
                    feat = model(images)
                    feat = feat.detach()

                    train_feats.append(feat)
                    train_labels.append(target)

                train_feats = torch.cat(train_feats, 0).cpu().numpy()
                train_labels = torch.cat(train_labels, 0).cpu().numpy()

                train_feats_norm = np.linalg.norm(train_feats, axis=1)
                train_feats = train_feats / (train_feats_norm + 1e-5)[:, np.newaxis]

                print('==> training SVM Classifier')
                cls_ap = np.zeros((args.num_class, 1))
                test_labels[test_labels == 0] = -1
                train_labels[train_labels == 0] = -1
                for cls in range(args.num_class):
                    clf = LinearSVC(
                        C=cost, class_weight={1: 2, -1: 1}, intercept_scaling=1.0,
                        penalty='l2', loss='squared_hinge', tol=1e-4,
                        dual=True, max_iter=2000, random_state=0)
                    clf.fit(train_feats, train_labels[:, cls])

                    prediction = clf.decision_function(test_feats)
                    P, R, score, ap = get_precision_recall(test_labels[:, cls], prediction)
                    cls_ap[cls][0] = ap * 100
                mean_ap = np.mean(cls_ap, axis=0)

                print('==> Run%d mAP is %.2f: ' % (run, mean_ap))
                avg_map.append(mean_ap)

            avg_map = np.asarray(avg_map)
            print('Cost:%.2f - Average ap is: %.2f' % (cost, avg_map.mean()))
            print('Cost:%.2f - Std is: %.2f' % (cost, avg_map.std()))
            result_k[i] = avg_map.mean()
        result[k] = result_k.max()
    print(result)


if __name__ == '__main__':
    main()

