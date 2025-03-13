
import h5py 

import pandas as pd 
def transform_to_csv(name_out_submit:str,preds,id):
    solutions_data = {'ID': [], 'Pred': []}

    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv('name_out_submit.csv')