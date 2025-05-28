#  @title # datasets
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_io import load_image


class TrainDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        degraded_path = os.path.join(root, 'degraded')
        self.image_names = sorted(os.listdir(degraded_path))
        self.image_names = [(name, f'{name[:4]}_clean{name[4:]}')
                                for name in self.image_names]
        self.transforms = transforms
        print(f'{root}: 共 {len(self.image_names)} 張圖像')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return 圖像: tensor(C, H, W), 乾淨的圖: tensor(C, H, W)
        """

        degraded_dir = os.path.join(self.root, 'degraded')
        clean_dir = os.path.join(self.root, 'clean')

        image_name = self.image_names[idx]
        degraded_path = os.path.join(degraded_dir, image_name[0])
        clean_path = os.path.join(clean_dir, image_name[1])

        image = load_image(degraded_path)
        target = load_image(clean_path)

        if self.transforms is not None:
            # print(image)
            image, target = self.transforms(image, target)
            # target = self.transforms(target)

        return image, target


class TestDataset(Dataset):
    def __init__(self, root, transforms=None):
        root = os.path.join(root, 'degraded')
        self.root = root
        self.image_names = sorted(os.listdir(root))
        self.transforms = transforms
        print(f'{root}: 共 {len(self.image_names)} 張圖像')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return 圖像: tensor(C, H, W), 圖像名稱: str
        """

        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, image_name)
        image = load_image(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, image_name