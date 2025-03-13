import h5py
import torch
import random
import numpy as np
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        # Open the file once and keep it open for reading
        self.hdf = h5py.File(self.dataset_path, 'r')
        self.image_ids = list(self.hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self.hdf[img_id]['img'][()]  # Read the image directly as a NumPy array
        label = self.hdf[img_id]['label'][()] if self.mode == 'train' else -1  # Use -1 for test mode

        return self.preprocessing(torch.tensor(img)).float(), label

    def __del__(self):
        # Ensure the file is closed when the dataset is deleted
        if hasattr(self, 'hdf') and self.hdf is not None:
            self.hdf.close()
import sys
def precompute(dataloader, model, device):
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)
# def precompute(dataloader, model, device):
    # xs, ys = [], []

    # # progress_bar = tqdm(dataloader, total=len(dataloader), file=sys.stderr, dynamic_ncols=True, ascii=True, disable=False)
    
    # for data in tqdm(dataloader):
        
    #     x, y = data
    #     with torch.no_grad():
    #         xs.append(model(x.to(device)).detach().cpu().numpy())
    #     ys.append(y.numpy())
        
    #     # progress_bar.set_description(f"Precomputing features [{len(xs)}/{len(dataloader)}]")

    # xs = np.vstack(xs)
    # ys = np.hstack(ys)
    
    return torch.tensor(xs), torch.tensor(ys)
class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        super(PrecomputedDataset, self).__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()