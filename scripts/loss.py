import torch
import numpy as np
import math
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
        mse = np.mean((images[1]/255.0-images[2]/255.0)**2)
        psnr = 20 * math.log10(1 / math.sqrt(mse))


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