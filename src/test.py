import numpy as np 
import pandas as pd
import torch
import tqdm 
from .transforms.tta_transforms import get_transform_tta
import clearml 
def apply_tta(sample,num_tta:int,tta_transform):
    #? We suppose TTA to be a tensor here 
    # create a output tensor of size (num_tta,sample.shape)

    output=torch.zeros((num_tta,sample.shape[0],sample.shape[1],sample.shape[2]))
    
    # the zero one is the original one
    output[0]=sample
    for i in range(1,num_tta):
        output[i]=tta_transform(sample)
    return output
def test_model_TTA(model,test_dataset,device,BATCH_SIZE,test_ids,test_dataloader,name_out="baseline.csv",combine_strategy="mean",tta_strategy="simple",threshold=0.5,clearml_found=False,to_save:bool=True):
    model=model.eval() 
  
    predictions = []
    tta_transform=get_transform_tta(tta_strategy)
    for sample,_ in tqdm.tqdm(test_dataset):
        with torch.no_grad(): 
            sample=sample.to(device)
            sample_tta=apply_tta(sample,num_tta=4,tta_transform=tta_transform).to(device)
            pred=model(sample_tta)
            if combine_strategy=="mean":
                pred=pred.mean(dim=0)
            elif combine_strategy=="max":
                pred=pred.max(dim=0)[0]
            else:
                raise ValueError("combine_strategy not recognized, got "+combine_strategy)
            predictions.append(pred.cpu().numpy())

    predictions=np.vstack(predictions)
    #* Put the threshold in 0 1 
    solutions_data = {'ID': [], 'Pred': []}
    solutions_data["Pred"]=(predictions>threshold).squeeze().astype(int)
    solutions_data["ID"]=test_ids

    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv(name_out) if to_save else None
    print("The predictions are saved in the file",name_out)
    if clearml_found:
        task=clearml.Task.current_task()
        task.set_user_properties(
            {
                "name": "submission file",
                "description": "name of submitted file",
                "value": name_out,
            }
        )


    return solutions_data



def test_model(model,test_dataset,device,BATCH_SIZE,test_ids,test_dataloader,name_out="baseline.csv",clearml_found=False):
    # test_dataset=BaselineDataset(TEST_IMAGES_PATH,preprocessing=preprocessing,mode="test")
    model=model.eval() 
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
    if clearml_found:
        task=clearml.Task.current_task()
        task.set_user_properties(
            {
                "name": "submission file",
                "description": "name of submitted file",
                "value": name_out,
            }
        )


    return solutions_data
