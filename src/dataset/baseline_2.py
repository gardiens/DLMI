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
from .baseline import save_precomputeddataset,load_precomputeddataset,models_have_same_weights,precompute
from src.transforms.naive_transforms import get_transform



class BaselineDataset_totrain(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset_totrain, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        # Open the file once and keep it open for reading
        self.hdf = h5py.File(self.dataset_path, 'r')
        self.image_ids = list(self.hdf.keys())
        # images,labels,center=self.load_all_images([self.dataset_path])
        # self.images = images
        # self.labels=labels
    def __len__(self):
        return len(self.image_ids)
    def load_all_images(self,list_paths):

        all_images = []
        all_labels = []
        all_centers = []

        for path in list_paths:
            print("Loading dataset:", path)
            with h5py.File(path, 'r') as hdf:
                keys = list(hdf.keys())
                for key in tqdm(keys):
                    img_array = np.array(hdf[key]['img'])  # shape (3, 96, 96)
                    label = int(np.array(hdf[key]['label']))
                    center = int(np.array(hdf[key]['metadata'])[0])

                    all_images.append(img_array)
                    all_labels.append(label)
                    all_centers.append(center)

        # Conversion en tenseurs PyTorch
        images_np = np.stack(all_images, axis=0)   # shape (N, 3, 96, 96)
        labels_np = np.array(all_labels, dtype=np.int64)
        centers_np = np.array(all_centers, dtype=np.int64)

        images_t = torch.from_numpy(images_np)
        labels_t = torch.from_numpy(labels_np)
        centers_t = torch.from_numpy(centers_np)

        return images_t, labels_t, centers_t
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self.hdf[img_id]['img'][()]  # Read the image directly as a NumPy array
        label = self.hdf[img_id]['label'][()] if self.mode == 'train' else -1  # Use -1 for test mode
        img=torch.from_numpy(img)
        label=torch.tensor(label)
        # label=torch.tensor(label)
        # return self.preprocessing(torch.tensor(img)).float(), label.float().flatten()
        # img=self.images[idx]
        # label=self.labels[idx]
        # Apply preprocessing
        # Passage en np.array (H, W, C) pour Albumentations
        image_np = img.permute(1, 2, 0).numpy().astype(np.float32)  # (96, 96, 3)

        if self.preprocessing:
            augmented = self.preprocessing(image=image_np)
            img = augmented["image"]
        else:
            image_torch = torch.from_numpy(image_np).permute(2, 0, 1)

        # img = self.preprocessing(img).float()
        # permute it back 
        img=torch.tensor(img).permute(2, 0, 1)  
        return img, label.float().flatten()
    def __del__(self):
        # Ensure the file is closed when the dataset is deleted
        if hasattr(self, 'hdf') and self.hdf is not None:
            self.hdf.close()

    
class PrecomputedDataset_totrain(Dataset):
    data_path="data/precomputed/"
    def __init__(self, dataloader, feature_extractor,device,stage="train",cache=True):
        #! REMOVE CACHE IF YOU DONT HAVE INFINITE STORAGE
        
        super(PrecomputedDataset_totrain, self).__init__()
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