import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, transforms as T
import torchvision.transforms.functional as F

# —— 从 dataset_synapse 拷贝过来的增强函数 ——  # 【新增】
def random_intensity(img, contrast_range=(0.9,1.1), brightness_range=(-0.1,0.1), p=0.5):
    if random.random() < p:
        # img 是 Tensor in [C,H,W]
        alpha = random.uniform(*contrast_range)
        beta  = random.uniform(*brightness_range)
        img = img * alpha + beta
        img = img.clamp(0.0, 1.0)
    return img

def random_gamma(img, gamma_range=(0.7,1.5), p=0.5):
    if random.random() < p:
        gamma = random.uniform(*gamma_range)
        img = F.adjust_gamma(img, gamma, gain=1)
    return img

def random_noise(img, noise_std=0.01, p=0.5):
    if random.random() < p:
        noise = torch.randn_like(img) * noise_std
        img = img + noise
        img = img.clamp(0.0, 1.0)
    return img
# —— end 拷贝 ——


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

        # 基础 Resize + ToTensor + Normalize
        base = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]

        if transform is None:
            self.transform = self.get_default_transform(mode, base)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return {'image': img, 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

    def get_default_transform(self, mode, base):
        if mode == 'train':
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),    # 保留原有
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.3),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(0.2,0.2,0.2,0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),

                    # —— 新增：强度、伽马与噪声 ——  # 【新增】
                    transforms.Lambda(lambda x: random_intensity(x, p=0.5)),
                    transforms.Lambda(lambda x: random_gamma(x, p=0.5)),
                    transforms.Lambda(lambda x: random_noise(x, p=0.3)),
                    # —— 强度增强 end ——

                    # 再统一到网络输入
                    *base
                ]
            )
        else:
            # 验证 / 测试无需这些随机变换
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    *base
                ]
            )
