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
        
        
        
        
        
        
if __name__ == '__main__':
    
    copyImg()