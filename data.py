from torch.utils.data import Dataset
import os
from PIL import Image
import random
import torchvision.transforms.functional as F
import numpy as np

class HuaLiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['bg', 'woodelf', 'waterpolo', 'lantern', 'ice']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class HorizontalRandomPerspective:
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BILINEAR):
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            startpoints = [(0, 0), (width, 0), (0, height), (width, height)]
            displacement = int(self.distortion_scale * width)
            
            # Randomly choose to distort either the left or right side
            if random.choice([True, False]):
                # Distort the left side
                endpoints = [
                    (random.randint(-displacement, displacement), 0),
                    (width, 0),
                    (random.randint(-displacement, displacement), height),
                    (width, height)
                ]
            else:
                # Distort the right side
                endpoints = [
                    (0, 0),
                    (width + random.randint(-displacement, displacement), 0),
                    (0, height),
                    (width + random.randint(-displacement, displacement), height)
                ]
            
            img = F.perspective(img, startpoints, endpoints, self.interpolation)
        return img

def __main__():
    pass