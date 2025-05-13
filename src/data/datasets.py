import os

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from ..training import config

class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = config.LABEL_MAP

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        img_path = row['filepath']
        label_str = row['label']
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at {img_path}.")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}.")

        if label_str not in self.label_map:
            raise ValueError(f"Label {label_str} not found in label map.")
        label_index = self.label_map[label_str]

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(label_index, dtype=torch.long)

# print("PlantVillageDataset class defined.")