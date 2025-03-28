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
from src.dataset.baseline import BaselineDataset,PrecomputedDataset,precompute
from src.models.basemodel import baseLine
import hydra 
import logging
from src.transforms.naive_transforms import get_transform
logging.getLogger('xformers').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
try:
    import clearml

    clearml_found = True
    # clearml_found = False
except ImportError:
    clearml_found = False
def setup_clearml(task_name,cfg):
    if clearml_found:
        from src.logger.clearml import safe_init_clearml

        task = safe_init_clearml(project_name="DLMI", task_name=task_name)
        task.connect(cfg)

    return task
def train(linear_probing,NUM_EPOCHS,train_dataloader,val_dataloader,optimizer,criterion,metric,PATIENCE,device):
    for epoch in tqdm(range(NUM_EPOCHS)):
        linear_probing.train()
        train_metrics, train_losses = [], []
        for train_x, train_y in train_dataloader:
            optimizer.zero_grad()
            train_pred = linear_probing(train_x.to(device))
            loss = criterion(train_pred, train_y.to(device))
            loss.backward()
            optimizer.step()
            train_losses.extend([loss.item()]*len(train_y))
            train_metric = metric(train_pred.cpu(), train_y.int().cpu())
            train_metrics.extend([train_metric.item()]*len(train_y))
        print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')

        linear_probing.eval()
        val_metrics, val_losses = [], []
        min_loss =np.inf
        for val_x, val_y in val_dataloader:
            with torch.no_grad():
                val_pred = linear_probing(val_x.to(device))
            loss = criterion(val_pred, val_y.to(device))
            val_losses.extend([loss.item()]*len(val_y))
            val_metric = metric(val_pred.cpu(), val_y.int().cpu())
            val_metrics.extend([val_metric.item()]*len(val_y))
        print(f'Epoch valid [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(val_losses):.4f} | Metric {np.mean(val_metrics):.4f}')

        if np.mean(val_losses) < min_loss:
            mean_val_loss = np.mean(val_losses)
            print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
            min_loss = mean_val_loss
            best_epoch = epoch
            torch.save(linear_probing.state_dict(), 'best_model.pth')

        if epoch - best_epoch == PATIENCE:
            print("We exceeded the patience, we stop at epoch",epoch)
            break
    return linear_probing
def set_seed():
    SEED = 0
    torch.random.manual_seed(SEED)
    random.seed(SEED)

def test_model(model,test_dataset,device,BATCH_SIZE,test_ids,name_out="baseline.csv"):
    # test_dataset=BaselineDataset(TEST_IMAGES_PATH,preprocessing=preprocessing,mode="test")
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    print("start testing")
    x,y=precompute(test_dataloader, model.feature_extractor, device)
    test_dataset = PrecomputedDataset(features=x, labels=y)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE,num_workers=2)
    predictions = []

    for test_x, _ in tqdm(test_dataloader,leave=True):
        with torch.no_grad():
            test_pred = model.linear_probing(test_x.to(device))
        predictions.append(test_pred.cpu().numpy())
    predictions = np.vstack(predictions)

    #* Put the threshold in 0 1 
    solutions_data = {'ID': [], 'Pred': []}
    solutions_data["Pred"]=(predictions>0.5).squeeze().astype(int)
    solutions_data["ID"]=test_ids
    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv(name_out)
    print("The predictions are saved in the file",name_out)
    return solutions_data

@hydra.main(version_base="1.2", config_path="configs", config_name="main.yaml")
def main(cfg):
    set_seed()
    if clearml_found:
        task = setup_clearml(
            cfg=cfg,task_name=cfg.task_name
        )
    print("start training",flush=True)
    BATCH_SIZE = 128
    num_workers=2
    TRAIN_IMAGES_PATH = 'train.h5'
    VAL_IMAGES_PATH = 'val.h5'
    TEST_IMAGES_PATH = 'test.h5'
    SEED = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_out_submit = (
        "submit/" + cfg.name_out_submit + str(random.randint(0, 255)) + ".csv"
    )
    #! Preprocessing 
    print("Preprocessing the dataset",flush=True)

    preprocessing = get_transform(transform_name=cfg.transform_name)
    train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, preprocessing, 'train')
    val_dataset = BaselineDataset(VAL_IMAGES_PATH, preprocessing, 'train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE,num_workers=num_workers)   
    #* Apply the preprocesing to the dataset

    print("loading the model")
    Main_model= baseLine(device)

    feature_extractor=Main_model.feature_extractor


    # --- Setup functions
    OPTIMIZER = cfg.optimizer.optimizer
    OPTIMIZER_PARAMS = cfg.optimizer.optimizer_params #{'lr': 0.001}
    print("optimizer params",OPTIMIZER_PARAMS,type(OPTIMIZER_PARAMS))
    LOSS = cfg.loss
    METRIC = cfg.metric
    NUM_EPOCHS = cfg.num_epochs
    PATIENCE = cfg.patience
    linear_probing=Main_model.linear_probing
    # Load function 
    metric = getattr(torchmetrics, METRIC)('binary')

    optimizer = getattr(torch.optim, OPTIMIZER)(linear_probing.parameters(), **OPTIMIZER_PARAMS)
    criterion = getattr(torch.nn, LOSS)()
    print("the optimizer",optimizer)

    #* ---precompute the features---
    print("Precompute the features",flush=True)
    x,y=precompute(train_dataloader, feature_extractor, device)
    print("We did it once",flush=True)
    train_dataset = PrecomputedDataset(features=x, labels=y)
    x,y=precompute(val_dataloader, feature_extractor, device)
    val_dataset = PrecomputedDataset(features=x, labels=y)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE,num_workers=num_workers)
    
    #* ---train the model---
    print("Start training the last layer",flush=True)

    linear_probing = train(linear_probing,NUM_EPOCHS,train_dataloader,val_dataloader,optimizer,criterion,metric,PATIENCE,device)
    
    #* Test the model
    test_dataset = BaselineDataset(TEST_IMAGES_PATH, preprocessing, 'test')
    print("Test the model",flush=True)
    test_model(model=Main_model,test_dataset=test_dataset,device=device,BATCH_SIZE=BATCH_SIZE,test_ids=test_dataset.image_ids,name_out=name_out_submit)
    return linear_probing
    
if __name__=="__main__":
    
    main()