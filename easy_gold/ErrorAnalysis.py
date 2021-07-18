
from utils import *
from image_utils import *

import numpy as np
import pandas as pd


class ErrorAnalysis():
    def __init__(self, _df_train, _df_test, _index_col, _target_col_list, _prefix):

        self.df_train = _df_train
        self.df_test = _df_test
        self.index_col = _index_col
        self.target_col_list = _target_col_list

        self.df_y_true = self.df_train[self.target_col_list]
        self.prefix = _prefix


    def procAnalysis(self, df_oof, df_y_pred, index_list):

        for target_col in self.target_col_list:

            prefix=f'{self.prefix}_{target_col}'
            se_oof = df_oof.loc[index_list, target_col].astype(float)
            se_y_true = self.df_y_true.loc[index_list, target_col].astype(float)
            se_y_pred = df_y_pred[target_col].astype(float)

           #pdb.set_trace()
            compPredTarget(se_oof.values, se_y_true.values, index_list=se_y_true.index, title_str=f"{prefix}", lm_flag=True)
            compHist(np_oof=se_oof.values, np_y_pred=se_y_pred.values, np_y_true=se_y_true.values, title_str=prefix)



    def procClass(self, df_oof, df_y_pred, index_list):

        for target_col in self.target_col_list:

            prefix=f'{self.prefix}_{target_col}'
            se_oof = df_oof.loc[index_list, target_col].astype(float)
            se_y_true = self.df_y_true.loc[index_list, target_col].astype(float)
            se_y_pred = df_y_pred[target_col].astype(float)

            if target_col == "target":
                se_oof = se_oof.map(lambda x: 0 if x < 0.5 else (1 if x < 1.5 else (2 if x < 2.5 else 3)))
                se_y_pred = se_y_pred.map(lambda x: 0 if x < 0.5 else (1 if x < 1.5 else (2 if x < 2.5 else 3)))
            else:
                se_oof = se_oof.map(lambda x: 0 if x < 0.5 else 1)
                se_y_pred = se_y_pred.map(lambda x: 0 if x < 0.5 else 1)

            compPredTarget(se_oof.values, se_y_true.values, index_list=se_y_true.index, title_str=f"{prefix}_class", lm_flag=True)
            compHist(np_oof=se_oof.values, np_y_pred=se_y_pred.values, np_y_true=se_y_true.values, title_str=f"{prefix}_class")

            print(classification_report(se_y_true.values.flatten(), se_oof.values.flatten()))




