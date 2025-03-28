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
import os 
def save_precomputeddataset(features,labels,stage,dir="data/"):

    if not os.path.exists(dir):
        os.makedirs(dir)
    # save the precomputed dataset in a h5 format
    output_file=os.path.join(dir, f"precomputed_dataset_{stage}.h5")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("features", data=features.numpy())
        f.create_dataset("labels", data=labels.numpy())
        f.close()
def load_precomputeddataset(dataset_path):
    # load the precomputed dataset in a h5 format
    with h5py.File(dataset_path, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        f.close()
    # convert to torch
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    return features, labels

def models_have_same_weights(model1, model2, rtol=1e-5, atol=1e-8):
    return model1.name==model2.name
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
    data_path="data/precomputed/"
    def __init__(self, dataloader, feature_extractor,device,stage="train",cache=True):
        #! REMOVE CACHE IF YOU DONT HAVE INFINITE STORAGE
        
        super(PrecomputedDataset, self).__init__()
        self.stage=stage        
        self.cache=cache
        self.data_path=os.path.join(self.data_path,f"{feature_extractor.name}")

        self.dataset_path=os.path.join(self.data_path, f"precomputed_dataset_{self.stage}.h5")
        
        # run the preprocessing
        features,labels=self.load_features_labels(dataloader,feature_extractor,device)
        self.features = features
        self.labels = labels.unsqueeze(-1)
    def load_features_labels(self,dataloader,feature_extractor,device):
        features,labels=None,None
        # precompute the features if it doesn't exist
        if  not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        model_path=os.path.join(self.data_path, f"FM_features.pt")
        if os.path.exists(model_path):
            model=torch.load(model_path)
            print("old cache name",model.name)
            print("new cache name",feature_extractor.name)
            if models_have_same_weights(model,feature_extractor) and os.path.exists(self.dataset_path):
                print("the same model is here is in the precomputed path, we load the features from disk")
                features,labels=load_precomputeddataset(self.dataset_path)

        else: 
            #do nothing and the computation
            pass

        if isinstance(features,type(None)) or isinstance(labels,type(None)): 
            print("the cache is different, we recompute the features and labels")

            features,labels=precompute(dataloader=dataloader,model=feature_extractor,device=device)
            # save the features and labels
            print("the cache",self.cache)
            if self.cache:
                print("we now save the features and labels in path",self.data_path)

                save_precomputeddataset(features,labels,dir=self.data_path,stage=self.stage)
                #save the model
                torch.save(feature_extractor, model_path)
            
        return features,labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()