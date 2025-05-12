import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.training import config
from src.data import datasets, preprocessing, sampler, splitting
from src.models import cnn_shallow


print(f"Using device: {config.DEVICE}")