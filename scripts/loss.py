import torch
import numpy as np
import math
import random
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
def proto_with_quality(output, target, output_proto, target_proto, criterion, acc_proto, images, num_cluster):
    # InfoNCE loss
    loss = criterion(output, target)

    # ProtoNCE loss
    if output_proto is not None:
        loss_proto = 0
        for proto_out, proto_target in zip(output_proto, target_proto):
            loss_proto += criterion(proto_out, proto_target)
            accp = accuracy(proto_out, proto_target)[0]
            acc_proto.update(accp[0], images[0].size(0))

        # average loss across all sets of prototypes
        loss_proto /= len(num_cluster)
        loss += loss_proto

        # Quality loss
        im_q = torch.split(images[0], split_size_or_sections=1, dim=0)
        im_q = [torch.squeeze(im, dim=0) for im in im_q]
        im_k = torch.split(images[1], split_size_or_sections=1, dim=0)
        im_k = [torch.squeeze(im, dim=0) for im in im_k]

        l_psnr = []
        l_ssim = []
        for i in range(min(len(im_q), len(im_k))):
            k = im_k[i]
            q_index = random.randint(0,i-2)
            if q_index >= i:
                q_index += 1
            q = im_q[q_index]
            psnr_temp = PSNR(k,q)
            if psnr_temp >= 50:
                psnr_temp = 0
            elif psnr_temp <= 30:
                psnr_temp = 1
            else:
                psnr_temp = (50-psnr_temp)/20
            l_psnr.append(psnr_temp)
            l_ssim.append(1-SSIM(k,q))

        loss += np.mean(l_psnr)+np.mean(l_ssim)


    return loss


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