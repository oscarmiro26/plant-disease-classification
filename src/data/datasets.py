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

        img_path = self.dataframe.iloc[idx]['filepath']
        label_str = self.dataframe.iloc[idx]['label']
        
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at {img_path}.")
        except Exception as e:
            raise IOError(f"Error opening image file: {e}.")

        label_index = config.LABEL_MAP.get(label_str)
        if label_index is None:
            raise ValueError(f"Label {label_str} not found in label map.")
        
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(label_index, dtype=torch.long)

print("PlantVillageDataset class defined.")