from PIL import ImageFilter
import random
import torch.utils.data as tud
import torchvision.datasets as datasets


class PreImager(tud.Dataset):
    def __init__(self, samples_dir, aug):
        data_meta = datasets.ImageFolder(samples_dir)
        images = data_meta.imgs
        self.classes = data_meta.classes
        self.class_number = len(self.classes)
        self.class_to_index = data_meta.class_to_idx
        img_class = [[] for i in range(self.class_number)]

        for img in images:
            img_class[img[1]].append(img[0])
        lens = [len(c) for c in img_class]
        self.length = min(lens)
        self.samples_dir = samples_dir

        self.aug = aug
        self.images = img_class
        self.loader = data_meta.loader

    def __getitem__(self, index):
        imgs = []
        for i in range(self.class_number):
            img = self.loader(self.images[i][index])
            out = self.aug(img)
            imgs.append(out)
        return imgs, index

    def __len__(self):
        return self.length

class TenserImager(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, target


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index