import torch
def data_process(cluster_result, images, gpu):
    im_q = []
    im_k = []
    if cluster_result is None:
        class_number = len(images)
        class_len = len(images[0])
        for _i in range(0, class_len, 2):
            for c in range(class_number):
                im_q.append(images[c][_i])
                im_k.append(images[c][_i + 1])
        im_q = torch.stack(im_q)
        im_k = torch.stack(im_k)
    else:
        im_q = images[0]
        im_k = images[1]

    if gpu is not None:
        im_q = im_q.cuda(gpu, non_blocking=True)
        im_k = im_k.cuda(gpu, non_blocking=True)

    return im_q, im_k