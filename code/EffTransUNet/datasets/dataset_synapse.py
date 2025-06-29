import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    # image = ndimage.rotate(image, angle, order=0, reshape=False)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_intensity(image, contrast_range=(0.9,1.1), brightness_range=(-0.1,0.1), p=0.5):
    if random.random() < p:
        alpha = random.uniform(*contrast_range)
        beta  = random.uniform(*brightness_range)
        image = image * alpha + beta
        image = np.clip(image, image.min(), image.max())
    return image

def random_gamma(image, gamma_range=(0.7,1.5), p=0.5):
    if random.random() < p:
        gamma = random.uniform(*gamma_range)
        # 先归一化到 [0,1]
        mn, mx = image.min(), image.max()
        image = (image - mn) / (mx - mn + 1e-8)
        image = np.power(image, gamma)
        image = image * (mx - mn) + mn
    return image

def random_noise(image, noise_std=0.01, p=0.5):
    if random.random() < p:
        noise = np.random.normal(0, noise_std, size=image.shape)
        image = image + noise
        image = np.clip(image, image.min(), image.max())
    return image

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 1. 旋转 / 翻转 / 旋转角度
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 2. 随机仿射（平移 + 缩放）
        if random.random() < 0.3:
            # 简单平移：上下左右各 10%
            tx = random.uniform(-0.1,0.1) * image.shape[0]
            ty = random.uniform(-0.1,0.1) * image.shape[1]
            image = ndimage.shift(image, shift=(tx,ty), order=3, mode='nearest')
            label = ndimage.shift(label, shift=(tx,ty), order=0, mode='nearest')

        # 3. 缩放回目标尺寸
        x, y = image.shape
        if (x, y) != tuple(self.output_size):
            zoom_factors = (self.output_size[0]/x, self.output_size[1]/y)
            image = zoom(image, zoom_factors, order=3)
            label = zoom(label, zoom_factors, order=0)

        # 4. 强度相关增强
        image = random_intensity(image, p=0.5)
        image = random_gamma(image, p=0.5)
        image = random_noise(image, p=0.3)

        # 5. To Tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        return {'image': image, 'label': label}

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
