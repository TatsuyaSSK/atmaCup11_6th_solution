# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 02:20:17 2021

@author: r00526841
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
import inspect
from matplotlib_venn import venn2

from utils import *

def copyImg():
    
    ppath_to_dir = INPUT_DIR/"photos"
    
    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
    #print(f"load df_train : {df_train.shape}")
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test.pkl')
    
    for index, row in df_train.iterrows():
        target_num = row["target"]
        ppath_to_image = ppath_to_dir / row["image_name"]
        
        
        pp_dir = ppath_to_dir / f"train/{target_num}"
        os.makedirs(pp_dir, exist_ok=True)
        
        
        new_pp = pp_dir / row["image_name"]
        shutil.copy(ppath_to_image, new_pp)
        
    for index, row in df_test.iterrows():
        
        ppath_to_image = ppath_to_dir / row["image_name"]
        pp_dir = ppath_to_dir / f"test"
        os.makedirs(pp_dir, exist_ok=True)
        
        
        new_pp = pp_dir / row["image_name"]
        shutil.copy(ppath_to_image, new_pp)
        
def drop_art_series(df_train):

    gp = df_train.groupby("art_series_id")["target"].nunique()
    drop_series = gp[gp>1].index
    df_train = df_train.loc[~df_train["art_series_id"].isin(drop_series)]   

    return df_train
        
def sub_round():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
    df_oof = pd.read_csv(OUTPUT_DIR/"20210713_030317_ResNet_Wrapper--0.800492--_oof.csv", index_col="object_id") 
    df_sub = pd.read_csv(OUTPUT_DIR/"20210713_030317_ResNet_Wrapper--0.800492--_submission.csv")
    print(df_sub.describe())

    #pdb.set_trace()
    df_train["oof"] =  df_oof["target"]
    print(df_train["oof"].describe())
    from sklearn.metrics import mean_squared_error
    def my_eval(y_pred, y_true):

        return np.sqrt(mean_squared_error(y_true, y_pred))

    initial_rmse =  my_eval(y_pred=df_train["oof"].values, y_true=df_train["target"].values)
    print(f"initial_rmse:{initial_rmse}")
    
    def calc_loss_f(_th_list, X, y_true):
        
        print(_th_list)
        #new_target = df_input_X["target"].map(lambda x: 0 if x < _th_list[0] else (1 if x < _th_list[1] else (2 if x < _th_list[2] else 3))).values

        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < _th_list[0]:
                X_p[i] = 0
            elif pred >= _th_list[0] and pred < _th_list[1]:
                X_p[i] = 1
            elif pred >= _th_list[1] and pred < _th_list[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3



        score  = my_eval(y_pred=X_p, y_true=y_true)
        print(score)
        return score
    
    initial_th_list = [0.5, 1.5, 2.5] #
    loss_partial = partial(calc_loss_f, X=df_train["oof"].values, y_true=df_train["target"].values)
    opt_result = sp.optimize.minimize(loss_partial, initial_th_list, method='nelder-mead')
    
    th_list = opt_result["x"].tolist()
    print(th_list)

    df_sub["target"] = df_sub["target"].map(lambda x: 0 if x < th_list[0] else (1 if x < th_list[1] else (2 if x < th_list[2] else 3)))
    df_sub.to_csv(OUTPUT_DIR/"20210713_030317_ResNet_Wrapper--0.800492--_submission_round.csv", index=False)
    
        
        
if __name__ == '__main__':
    
    sub_round()