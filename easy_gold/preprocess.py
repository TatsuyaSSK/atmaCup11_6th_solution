# coding:utf-8

from pathlib import Path

import numpy as np
import pandas as pd

import argparse
import inspect
from eda import procEDA, showDetails

from utils import *

from log_settings import MyLogger

my_logger = MyLogger()
logger = my_logger.generateLogger("preprocess", LOG_DIR+"/preprocess.log")



    
    
def countNull(df, target_):
    
    list_col=list(df.columns)
    for del_col in target_:
        list_col.remove(del_col)
    
    df = addNanPos(df, list_col)
    
    df["null_all_count"] = df[list_col].isnull().sum(axis=1)
    
    #ord
    #df["null_ord_count"] = df[getColumnsFromParts(["ord_"], list(df.columns))].isnull().sum(axis=1)


    
    
    
    return df



def checkNan(df, target_, fill_val=-999):
    
    #df = df.replace([np.inf, -np.inf], np.nan)
    nan_cols = showNAN(df)
    
    for col in nan_cols:
        if not col in target_:
            if not ON_KAGGLE:
                print("fill na : ", col)
            #df[col].fillna(df[col].mode()[0], inplace=True)
            df[col].fillna(fill_val, inplace=True)
    
    return df
    
    
def removeColumns(df, drop_cols=[], not_proc_list=[]):
    
    
    
    #df_train, df_test = self.getTrainTest()
    #drop_cols += get_useless_columnsTrainTest(df_train, df_test, null_rate=0.95, repeat_rate=0.95)
    
    exclude_columns=not_proc_list + drop_cols
    dCol = checkCorreatedFeatures(df, exclude_columns=exclude_columns, th=0.99999)
    drop_cols.extend(dCol)
    
    #df_train, _ = self.getTrainTest()
    #null_features = nullImporcance(df_train.drop(self.target_, axis=1), df_train[self.target_], th=80, n_runs=100)
    #logger.debug("null_features")
    #logger.debug(null_features)
    
    #null_features = ['nom_9', 'nom_6', 'nom_5', 'nom_7', 'nom_8', 'nom_2', 'nom_3', 'ord_1', 'nom_1', 'ord_0', 'nom_4', 'month_sin', 'day', 'month', 'month_angle_rad', 'day_sin', 'month_cos', 'bin_2', 'bin_1', 'bin_0']
    #drop_cols.extend(null_features)
    final_drop_cols = []
    for col in drop_cols:
        if not col in not_proc_list and col in df.columns:
            if not ON_KAGGLE:
                print("remove : {}".format(col))
            final_drop_cols.append(col)
    
    df.drop(columns=final_drop_cols, inplace=True)
    return df
    


    
class Preprocessor:
    
    def __init__(self, _df_train, _df_test, index_col, _str_target_value, _regression_flag=1):
        
        self.target_ = _str_target_value
        self.regression_flag_ = _regression_flag
        self.index_col = index_col
        
        self.all_feature_func_dict = {t[0]:t[1] for t in inspect.getmembers(self, inspect.ismethod) if "fe__" in t[0] }
        self.path_to_f_dir = PATH_TO_FEATURES_DIR
        
        if _df_test is not None:
            self.df_all_ = pd.concat([_df_train, _df_test], sort=False) #, ignore_index=True)
        else:
            self.df_all_ = _df_train
            
        if not ON_KAGGLE:
            logger.debug("self.df_all_.index : {}".format(self.df_all_.index))
            logger.debug("self.df_all_.columns : {}".format(self.df_all_.columns))

        self.original_columns_ = list(self.df_all_.columns)
        

        
        for del_col in self.target_:
            if del_col in self.original_columns_:
                self.original_columns_.remove(del_col)
        
        
        
        self.train_idx_ = _df_train.index.values
        self.test_idx_ = _df_test.index.values if _df_test is not None else []
        
        
        if not ON_KAGGLE:
            print(f"original_columns :{self.original_columns_}")
            print("df_train : ", _df_train.shape)
            print("self.train_idx_ : ", self.train_idx_)
            
            if _df_test is not None:
                print("df_test_ : ", _df_test.shape)
                print("self.test_idx_ : ", self.test_idx_)
        
        self.add_cols_ = []
        self.model = None
        
    def getTrainTest(self):
        
        
        if len(self.test_idx_) == 0:
            df_train = self.df_all_
        else:
            df_train = self.df_all_.loc[self.df_all_.index.isin(self.train_idx_)]
    
        
        df_test = self.df_all_.loc[self.df_all_.index.isin(self.test_idx_)]
        
        for col in self.target_:
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)
        
        
        
        if not ON_KAGGLE:
            pass
            # print("**** separate train and test ****")
            # print("df_train : ", df_train.shape)
            # print("self.train_idx_ : ", df_train.index)
            # print("df_test_ : ", df_test.shape)
            # print("self.test_idx_ : ", df_test.index)
        

        return df_train, df_test
    
    

        
        
    def fe__fillna_Age(self, df):
        
        
        df['fillna_Age'] = df['Age'].fillna(df['Age'].median())
       
        
    def fe__fillna_Fare(self, df):
        df["fillna_Fare"] = df['Fare'].fillna(np.mean(df['Fare']))
        
    def fe__IsAlone(self, df):
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        
    def fe__FamilySize(self, df):
        df["FamilySize"] = df['Parch'] + df['SibSp'] + 1
        
        
    def fe__label_Embarked(self, df):
        df['label_Embarked'] = df['Embarked'].fillna('S')
        
        df['label_Embarked'] = df['label_Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        

        
    def fe__label_Sex(self, df):
        
        df['label_Sex'] = df['Sex'].replace(['male', 'female'], [0, 1])

  
    
    
    def loadFs(self, df, f_name_list):
        
        for f_name in f_name_list:
            force_flag = True if f_name in self.force_list else False
            df = self.loadF(df, f_name, force_flag)
            
        return df
    
    def loadF(self, df, f_name, force=False):
        
        if f_name in df.columns:
            print(f"{f_name} already exists in columns")
            return df
        
        import warnings
        warnings.filterwarnings('ignore')
        
        path_to_f = self.path_to_f_dir /f"{f_name}.pkl"
        
        if (path_to_f.exists()) & (force==False):
            
            df[f_name] = pd.read_pickle(path_to_f)
            print(f"load {f_name}")
            
        else:
            
            with timer2(f"create {f_name}"):
                self.all_feature_func_dict[f"fe__{f_name}"](df)

            df[f_name].to_pickle(path_to_f)
            
            if not ON_KAGGLE:
                df_train, df_test = self.getTrainTest()
                showDetails(df_train, df_test, new_cols=[f_name], target_val=self.target_[0], corr_flag=False)
                
            
            
        return df
        
        
    def proc(self, params):
        
        
    
        
        all_feature_names = [name.replace("fe__", "") for name in  self.all_feature_func_dict.keys()]
        self.force_list = []
        if params["force"] is not None:
            if len(params["force"]) == 0:
                
                self.force_list = all_feature_names
            else:
                self.force_list = params["force"]
        
        load_list = all_feature_names
        
        for f_name in load_list:
            
            force_flag = True if f_name in self.force_list else False
            self.df_all_ = self.loadF(self.df_all_, f_name, force=force_flag)
            
            
            
        self.df_all_ =checkNan(self.df_all_, target_=self.target_, fill_val=-999)
        
        
        exclude_label= ['Name',  'Ticket', 'Cabin', "Sex", "Embarked"]
        self.df_all_, _ = proclabelEncodings(self.df_all_, not_proc_list=exclude_label+self.target_)
        
        
        self.df_all_ =  removeColumns(self.df_all_, drop_cols=exclude_label, not_proc_list=self.target_)


    

def procMain(df_train, df_test, index_col, target_col, setting_params):
    

            
    myPre = Preprocessor(df_train, df_test, index_col, target_col)
    myPre.proc(params=setting_params)
    df_proc_train, df_proc_test = myPre.getTrainTest()

    return df_proc_train, df_proc_test




def main(setting_params):

    
    target_col = ['Survived']
    index_col = "PassengerId"


    full_load_flag=setting_params["full_load_flag"]
    save_pkl=1
    if ON_KAGGLE==True or full_load_flag==1:
        
        df_train = pd.read_csv(INPUT_DIR / "train.csv")
        df_train = df_train.set_index(index_col)
        
        df_test = pd.read_csv(INPUT_DIR / "test.csv")
        df_test = df_test.set_index(index_col)
        
        
        if save_pkl and ON_KAGGLE==False:
            df_train.to_pickle(INPUT_DIR / 'train.pkl')
            df_test.to_pickle(INPUT_DIR / 'test.pkl')
        
    else:
        
        
        df_train = pd.read_pickle(INPUT_DIR / 'train.pkl')
        df_test = pd.read_pickle(INPUT_DIR / 'test.pkl')

        
    logger.debug("df_train:{}".format(df_train.shape))
    logger.debug("df_test:{}".format(df_test.shape))

    
 

    df_proc_train, df_proc_test =  procMain(df_train, df_test, index_col, target_col, setting_params) 
    #df_proc_train = reduce_mem_usage(df_proc_train)

    df_proc_train.to_pickle(PROC_DIR / f'df_proc_train_{setting_params["mode"]}.pkl')
    df_proc_test.to_pickle(PROC_DIR / f'df_proc_test_{setting_params["mode"]}.pkl')

    
    logger.debug(f"df_proc_train:{df_proc_train.columns}")
    logger.debug("df_proc_train:{}".format(df_proc_train.shape))
    
    logger.debug(f"df_proc_test:{df_proc_test.columns}")
    logger.debug("df_proc_test:{}".format(df_proc_test.shape))
    
    


        
def argParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="lgb", choices=['lgb','exp','nn','graph', 'ave', 'stack'] )
    
    parser.add_argument('-stack_dir', '--stacking_dir_name', type=int, )
    parser.add_argument('-full', '--full_load_flag', action="store_true")
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-f', '--force', nargs='*')
    


    args=parser.parse_args()

    setting_params= vars(args)

    return setting_params

if __name__ == '__main__':
    setting_params=argParams()
    
    print(setting_params["force"])

    
    main(setting_params = setting_params)









