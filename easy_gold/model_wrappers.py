# coding:utf-8

import os
import pathlib
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb 
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from abc import ABCMeta, abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger




# from deeptables.models.deeptable import DeepTable, ModelConfig
# from deeptables.models.deepnets import WideDeep
# from deeptables.models.layers import MultiColumnEmbedding, dt_custom_objects
# from deeptables.utils import consts
#import tensorflow as tf
#from tensorflow.keras.models import load_model

from DNNmodel import *
import cloudpickle

from utils import *
import inspect





from log_settings import MyLogger

my_logger = MyLogger()
logger = my_logger.generateLogger("model_wrapper", LOG_DIR+"/model_wrapper.log").getChild(__file__)




def extractModelParameters(original_param, model):
    
    model_params_keys = model.get_params().keys()
    model_params = {}
    for k, v in original_param.items():
        if k in model_params_keys:
            model_params[k] = v
    logger.debug(model_params)
    return model_params

def extractModelParametersWithStr(original_param, exclude_str="__"):

    
    model_params = {}
    for k, v in original_param.items():
        if not exclude_str in k:
            model_params[k] = v
    logger.debug(model_params)
    return model_params

class Averaging_Wrapper(object):
    
    def __init__(self, df_train, df_test, target_col, path_to_meta_feature_dir, rate_list=None):
        self.model = None
        self.initial_params = {
            "random_seed_name__":"random_state",
            
        }
        self.target_col = target_col
        self.rate_list_ = rate_list
        
        self.best_iteration_ = 1
        
        

        self.df_meta_train, self.df_meta_test = self.setMetaFromFiles(path_to_meta_feature_dir)
        idx1 = set(self.df_meta_train.index)
        idx2 = set(df_train.index)
        print(f"df_meta_train : {self.df_meta_train.shape}")
        print(f"df_train : {df_train.shape}")

        
        #
        
        # #for mRNA
        # #self.df_meta_train = self.df_meta_train.loc[self.df_meta_train.index.isin(df_train.index),:]
        self.df_meta_train[target_col] = df_train.loc[self.df_meta_train.index, target_col]

        # meta_train_mol_ids = df_train.loc[self.df_meta_train.index,"id"].unique()
        # scored_length = df_train.loc[self.df_meta_train.index,"seq_scored"].values[0]
        # print(f"meta_train_mol_ids : {len(meta_train_mol_ids)}")
        # print(f"scored_length:{scored_length}")
        # scored_mol_pos_ids = [f"{mol_id}_{i}" for mol_id in meta_train_mol_ids for i in range(scored_length)]
        # print(f"scored_mol_pos_ids : {len(scored_mol_pos_ids)}")

        # self.df_meta_train = self.df_meta_train.loc[scored_mol_pos_ids, :]
        # print(self.df_meta_train)

        
        #y_pred_list, oof_list = self.setMetaFromFiles(path_to_meta_feature_dir)
        
        #self.setMeta(df_train, df_test, target_col, y_pred_list, oof_list)
        
    def setMeta(self, df_train, df_test, target_col:str, y_pred_list:list, oof_list:list):
        
        np_y_pred = np.concatenate(y_pred_list, 1)
        np_oof = np.concatenate(oof_list, 1)
        print(np_y_pred.shape)
        print(np_oof.shape)
        self.df_meta_test = pd.DataFrame(np_y_pred, index=df_test.index)
        self.df_meta_train = pd.DataFrame(np_oof, index=df_train.index)

        self.df_meta_train[target_col] = df_train[target_col]
        
    
    def setMetaFromFiles(self, path_to_meta_feature_dir):
        pp_dir = pathlib.Path(path_to_meta_feature_dir)
        
        y_pred_list=[]
        oof_list = []
        for f in pp_dir.glob('*--_oof.csv'):
            oof_f_name = f.name
            print(oof_f_name)
            
            df_oof = pd.read_csv(str(f.parent/oof_f_name), index_col=0)[self.target_col]
            
            print(f"df_oof : {df_oof}")
            oof_list.append(df_oof)

            
            
            pred_f_name = oof_f_name.replace("oof", "submission")
            print(pred_f_name)
            
            df_pred = pd.read_csv(str(f.parent/pred_f_name), index_col=0)[self.target_col]
            
            print(f"df_pred : {df_pred}")
            y_pred_list.append(df_pred)
        
        df_oof = pd.concat(oof_list, axis=1)
        df_oof.columns=[i for i in range(0, len(df_oof.columns))]
        
        df_pred = pd.concat(y_pred_list, axis=1)
        df_pred.columns=[i for i in range(0, len(df_pred.columns))]

            
        
        return df_oof, df_pred
    

    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
        
        pass

        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        best_score_dict={}
        eval_metric_func_dict = params["eval_metric_func_dict__"]
        print(X_train)
        
        if self.rate_list_ == None:
            
            f = list(eval_metric_func_dict.values())[0]

            
            def calc_loss_f(_rate_list, df_input_X, y_true):
                
                this_pred = np.zeros(len(df_input_X))
                for c, r in zip(df_input_X.columns, _rate_list):
                    this_pred += (df_input_X[c] * r).values

                score  = f(y_pred=this_pred, y_true=y_true)
                
                return score
            
            initial_rate_list = [0.5] * X_train.shape[1]
            loss_partial = partial(calc_loss_f, df_input_X=X_train, y_true=y_train)
            opt_result = sp.optimize.minimize(loss_partial, initial_rate_list, method='nelder-mead')
            
            self.rate_list_ = opt_result["x"].tolist()
            print(f"*****  opt result : {self.rate_list_} *****")
                
            #pdb.set_trace()
        
            
        
        
        y_train_pred = self.predict(X_train) #X_train.mean(axis=1).values
        print(y_train_pred)

        
        best_score_dict["train"]=calcEvalScoreDict(y_true=y_train, y_pred=y_train_pred, eval_metric_func_dict=eval_metric_func_dict)

        if X_valid is not None:
            #y_valid_pred = self.model.predict(X_valid)
            y_valid_pred = self.predict(X_valid) #X_valid.mean(axis=1).values
            best_score_dict["valid"] = calcEvalScoreDict(y_true=y_valid, y_pred=y_valid_pred, eval_metric_func_dict=eval_metric_func_dict)
            
        if X_holdout is not None:
            
            #y_holdout_pred = self.model.predict(X_holdout)
            y_holdout_pred = self.predict(X_holdout)#X_holdout.mean(axis=1).values
            best_score_dict["holdout"] = calcEvalScoreDict(y_true=y_holdout, y_pred=y_holdout_pred, eval_metric_func_dict=eval_metric_func_dict)


        self.best_score_ = best_score_dict
        logger.debug(self.best_score_)
        
        self.setFeatureImportance(X_train.columns)

    def predict(self, X_test):
        
        if self.rate_list_ == None:
            print(f"X_test : {X_test}")
            print(f"X_test.mean(axis=1) : {X_test.mean(axis=1)}")
            return X_test.mean(axis=1).values
        else:
            pred = np.zeros(len(X_test))
            for c, r in zip(X_test.columns, self.rate_list_):
                pred += (X_test[c] * r).values
            return pred

    def setFeatureImportance(self, columns_list):
        self.feature_importances_ = np.zeros(len(columns_list))
    

class Model_Wrapper_Base(object):
    
    def __init__(self):
        self.model = None
        self.initial_params = {
            "random_seed_name__":"random_state",
            
        }
        
        self.best_iteration_ = 1
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        self.edit_params = params

        best_score_dict={}
        eval_metric_func_dict = params["eval_metric_func_dict__"]
        
        self.model = self.model.set_params(**extractModelParameters(params, self.model))
        
        self.model.fit(X=X_train, y=y_train)
        #y_train_pred = self.model.predict(X_train)
        y_train_pred = self.predict(X_train)
        

        if X_valid is not None:
            #y_valid_pred = self.model.predict(X_valid)
            y_valid_pred = self.predict(X_valid)
            best_score_dict["valid"] = calcEvalScoreDict(y_true=y_valid, y_pred=y_valid_pred, eval_metric_func_dict=eval_metric_func_dict)
            

        if X_holdout is not None:
            
            #y_holdout_pred = self.model.predict(X_holdout)
            y_holdout_pred = self.predict(X_holdout)
            best_score_dict["holdout"] = calcEvalScoreDict(y_true=y_holdout, y_pred=y_holdout_pred, eval_metric_func_dict=eval_metric_func_dict)


        self.best_score_ = best_score_dict
        logger.debug(self.best_score_)
        self.setFeatureImportance(X_train.columns)

    def predict(self, X_test):
        return self.model.predict(X_test)


    def setFeatureImportance(self, columns_list):
        self.feature_importances_ = np.zeros(len(columns_list))


class ElasticNetRegression_Wrapper(Model_Wrapper_Base):


    def __init__(self):
        super().__init__()
        self.model = ElasticNet()
        
        tmp_param = {
                 'alpha': 0.0001,#0.0005,
                 'copy_X': True,
                 'fit_intercept': True,
                 'l1_ratio': 0.001,#0.05,
                 'max_iter': 1000,
                 'normalize': False,
                 'positive': False,
                 'precompute': False,
                 'selection': 'cyclic',
                 'tol': 0.0001,
                 'warm_start': False
                 }
        self.initial_params.update(tmp_param)
        
class Lasso_Wrapper(Model_Wrapper_Base):
    
    def __init__(self):
        super().__init__()
        self.model = Lasso()
        
        tmp_param = {
                 'alpha': 0.0005,
                 
                 }
        self.initial_params.update(tmp_param)
        
class Ridge_Wrapper(Model_Wrapper_Base):
    
    def __init__(self):
        super().__init__()
        self.model = Ridge()
        
        tmp_param = {
                 'alpha': 0.9,#0.0005,
                 
                 }
        self.initial_params.update(tmp_param)

class SVR_Wrapper(Model_Wrapper_Base):
    
    def __init__(self):
        super().__init__()
        self.model = SVR()
        
        tmp_param = {
                 'kernel':'linear', 
                 'C':100, 
                 'gamma':'auto',
                 
                 }
        self.initial_params.update(tmp_param)
    
        
class LogisticRegression_Wrapper(Model_Wrapper_Base):


    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()
        
        tmp_param = {
                 'penalty':'l2',
                 'C': 0.049,#0.05,#0.03,
                 'class_weigh':{0: 1, 1: 1.42},
                 #'verbose':0,
                 'max_iter':5000,
                 'solver': 'liblinear',#'lbfgs', 
                 "fit_intercept":True,
                 #'num_class':2,

                 'binary_flag':1,
                 }
        self.initial_params.update(tmp_param)


    def predict(self, X_test):

        if self.edit_params['binary_flag']:
            print("probability")
            return self.model.predict_proba(X_test)[:, 1]
        else:
            #return self.model.predict(X_test)
            return self.model.predict_proba(X_test)

    def setFeatureImportance(self, columns_list):
        self.feature_importances_ = pd.Series(self.model.coef_[0])

class RandomForestClassifier_Wrapper(Model_Wrapper_Base):


    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()
        
        tmp_param = {
                 'n_jobs':-1,
                 'verbose':True,
                 'n_estimators':1000, 
                 'max_features': 3,
                 'min_samples_leaf': 5,
                 #'min_samples_split': 5,#2,
                 #"oob_score":True,

                 #'num_class':2,
                 'binary_flag':1,
                 }
        self.initial_params.update(tmp_param)


    def predict(self, X_test):

        if self.edit_params['binary_flag']:
            print("probability")
            return self.model.predict_proba(X_test)[:, 1]
        else:
            #return self.model.predict(X_test)
            return self.model.predict_proba(X_test)

def show_cuda(text=None):

    print(f"******[show cuda : {text}]********")
    #print(f"******{torch.cuda.memory_stats()}")
    #print(f"****** memory_cached : {torch.cuda.memory_cached()}/{torch.cuda.max_memory_cached()}******")
    #print(f"****** memory_allocated : {torch.cuda.memory_allocated()}/{torch.cuda.max_memory_allocated()}******")
    #print(f"****** memory_reserved : {torch.cuda.memory_reserved()}/{torch.cuda.max_memory_reserved()}******")

    print(torch.cuda.get_device_name())
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated()/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved()/1024**3,1), 'GB')

class PytrochLightningBase():

    def __init__(self):
        super().__init__()

        self.initial_params = {
                #""'early_stopping_rounds':50,
                #""'lr':0.001,#0.01,#0.005,
                #"batch_size":64 * 4,
                #"epochs":10000,
                #"loss_criterion_function": {"name":"mse", "func": my_loss_func},#nn.BCEWithLogitsLoss()} , #
    
                #"eval_metric":"val_MPE",
                "eval_max_or_min": "min",
                "val_every":1,
                "dataset_params":{},
                "random_seed_name__":"random_state",
                'num_class':1, #binary classification as regression with value between 0 and 1
                "use_gpu":1,
                'multi_gpu':False,
                }
        self.edit_params = {}

        self.best_iteration_ = 1

        self.reload_flag = False

        self.model = None
    #     self.setModel()

    # @abstractmethod
    # def setModel(self):
    #     pass  # あるいは raise NotImplementedError()




    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        params = prepareModelDir(params, self.__class__.__name__)

        self.edit_params = params
        pl.seed_everything(params[params["random_seed_name__"]]) 
        torch.backends.cudnn.enabled = True

        self.model.setParams(params)

        # def init_weights2(m):
        #     initrange = 0.1
        #     #print(f"m : {m}, type:{type(m)}")
        #     if type(m) == nn.Linear:
        #         m.weight.data.uniform_(-initrange, initrange)

        #         #m.weight.fill_(1.0)
        #         #print(m.weight)
        # self.model.apply(init_weights2)

        batch_size = params["batch_size"]
        if batch_size < 0:
            batch_size = X_train.shape[0]


        data_set_train = params["dataset_class"](X_train, y_train, params["dataset_params"], train_flag=True)
        dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True, collate_fn=params["collate_fn"],num_workers=params["num_workers"]) #, sampler=ImbalancedDatasetSampler(data_set_train))
        

        dataloader_val = None
        if X_valid is not None:
            data_set_val = params["dataset_class"](X_valid, y_valid, params["dataset_params"], train_flag=True)
            dataloader_val = torch.utils.data.DataLoader(data_set_val, batch_size=batch_size, shuffle=False, collate_fn=params["collate_fn"],num_workers=params["num_workers"])



        wandb_logger=None
        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            #wandb.init(config=params)
            wandb_run = wandb.init(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb_logger = WandbLogger(experment=wandb_run)
           # wandb_logger = WandbLogger(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb_logger.log_hyperparams(params)
            #
            #wandb.config.update(params,  allow_val_change=True)
            #wandb.watch(self.model, criterion, log=None)


        early_stop_callback = EarlyStopping(
                                monitor=f'val_{params["eval_metric"]}',
                                min_delta=0.00,
                                patience=params['early_stopping_rounds'],
                                verbose=True,
                                mode=params['eval_max_or_min']
                            )

        checkpoint_callback = ModelCheckpoint(
                                dirpath=PATH_TO_MODEL_DIR / params["model_dir_name"],
                                filename=params["path_to_model"].stem,
                                verbose=True,
                                monitor=f'val_{params["eval_metric"]}',                                
                                mode=params['eval_max_or_min'],
                                save_weights_only=True,
                                )

        # callbacks_list = []
        # if params['early_stopping_rounds'] > 0:
        #     callbacks_list = [early_stop_callback, checkpoint_callback]

                                

        self.trainer = pl.Trainer(
                        num_sanity_val_steps=0,
                        gpus=self.initial_params["use_gpu"], 
                        check_val_every_n_epoch=self.initial_params["val_every"],
                        max_epochs=self.initial_params["epochs"],
                        callbacks=[early_stop_callback, checkpoint_callback],
                        logger=wandb_logger,  

        )   

        self.trainer.fit(self.model, dataloader_train, dataloader_val)

        # if X_valid is not None:
        #     print(checkpoint_callback.best_model_path)
        #     pdb.set_trace()
        #     self.model.load_state_dict(torch.load(checkpoint_callback.best_model_path), strict=False) #self.model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
        #     #self.model.load_state_dict(torch.load(str(params["path_to_model"])+".ckpt"))
        # else:
        if X_valid is None:
            self.trainer.save_checkpoint(str(params["path_to_model"]))
            #torch.save(self.model.state_dict(), str(params["path_to_model"])+".ckpt")


        self.best_score_, self.best_iteration_, _ = self.model.getScoreInfo()
        logger.debug(self.best_score_)
        
        self.feature_importances_ = np.zeros(len(X_train.columns))
        #self.best_iteration_ = best_epoch#self.model.best_iteration_

        #if wandb_logger is not None:
            #wandb_logger.finalize("success")
            #wandb.finish()

    def predict(self, X_test, oof_flag=False):
        
        num_tta = self.edit_params["num_tta"]
        

        batch_size=self.edit_params["batch_size"] #if oof_flag else 1 #
        dummy_y = pd.DataFrame(np.zeros(X_test.shape), index=X_test.index)
        data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=(num_tta>1))
        dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, collate_fn=self.edit_params["collate_fn"],num_workers=self.edit_params["num_workers"])
        
        self.model.oof_prediction = oof_flag
        if self.model.oof_prediction==False:
            self.trainer.logger = None 


        tta_list = []
        for tta_i in range(num_tta):
            print(f"tta : {tta_i+1}th")
            if self.reload_flag:
                
                self.trainer.test(test_dataloaders=dataloader_test, model=self.model)
            else:
                
                self.trainer.test(test_dataloaders=dataloader_test, ckpt_path='best')
            final_preds = self.model.final_preds
            tta_list.append(final_preds)

        #pdb.set_trace()

        return np.array(tta_list).mean(axis=0)

    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        ppath_to_model = ppath_to_save_dir / f"model__{prefix}__{model_dir_name}__{self.__class__.__name__}.pkl"
        torch.save(self.model.state_dict(), str(ppath_to_model))

        print(f'Trained nn model was saved! : {ppath_to_model}')
        
        with open(str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json"), 'w') as fp:
            json.dump(bs, fp)


    def procLoadModel(self, model_dir_name, prefix, params):
        self.edit_params = params

        self.model.cuda()
        #show_cuda("init")
        
        if params["multi_gpu"]:
            self.model = nn.DataParallel(self.model)
        
        ppath_to_save_dir = PATH_TO_UPLOAD_MODEL_PARENT_DIR / model_dir_name
        print(f"ppath_to_save_dir : {ppath_to_save_dir}")
        print(list(ppath_to_save_dir.glob(f'model__{prefix}*')))
        #print(list(ppath_to_save_dir.iterdir()))
        
        name_list = list(ppath_to_save_dir.glob(f'model__{prefix}*'))
        if len(name_list)==0:
            print(f'[ERROR] Trained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]

        ppath_to_ckpt_model = searchCheckptFile(ppath_to_save_dir, ppath_to_model, prefix)
        
        #pdb.set_trace()
        self.model.load_state_dict(torch.load(str(ppath_to_ckpt_model))["state_dict"])
        #self.model.load_state_dict(torch.load(str(ppath_to_model)))
        print(f'Trained nn model was loaded! : {ppath_to_ckpt_model}')
        
        a = int(re.findall('iter_(\d+)__', str(ppath_to_model))[0])
        
        
   
        #print(self.model.best_iteration_ )
        #self.model.best_iteration_ 
        self.best_iteration_ = a
        
        path_to_json = str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json")
        if not os.path.exists(path_to_json):
            print(f'[ERROR] Trained nn json was NOT EXITS! : {path_to_json}')
            return -1
        with open(path_to_json) as fp:
            self.best_score_ = json.load(fp)
            #self.model._best_score = json.load(fp)

        

        self.trainer = pl.Trainer(
                        num_sanity_val_steps=0,
                        gpus=self.edit_params["use_gpu"], 
                        check_val_every_n_epoch=self.edit_params["val_every"],
                        max_epochs=self.edit_params["epochs"],
        )
        self.reload_flag = True

        return 0

class SSL_Wrapper(PytrochLightningBase):
    def __init__(self, img_size, num_out, regression_flag):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = SupConDataset
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {"img_size":img_size}
        
        self.model = SupConModel(base_name="efficientnet_b1") #num_out=num_out, regression_flag=regression_flag)

class ResNet_Wrapper(PytrochLightningBase):
    def __init__(self, img_size, num_out, regression_flag):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = MyDatasetResNet
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] =  {"img_size":img_size}
        
        self.model = myResNet(num_out=num_out, regression_flag=regression_flag)


class multiLabelNet(PytrochLightningBase):
    def __init__(self, img_size, num_out, regression_flag, tech_weight, material_weight):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = MyDatasetResNet
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {"img_size":img_size}
        
        #'', efficientnet_b1
        self.model = myMultilabelNet(base_name="vit_base_patch16_384", num_out=num_out, regression_flag=regression_flag, tech_weight=tech_weight, material_weight=material_weight)


        
class LastQueryTransformer_Wrapper(PytrochLightningBase):
    def __init__(self,
                #f_all, 
                sequence_features_list, 
                continuous_features_list,
                max_seq=1,
                #ninp=32,
                # nhead=1,
                # nhid=32,
                # nlayers=1,
                # dropout=0.1,
                pad_num=0,
                last_query_flag=False,
                ):


        super().__init__()

        bssid_features_list = getColumnsFromParts(["wb_"], sequence_features_list)
        fq_features_list = getColumnsFromParts(["wifi_frequency_"], sequence_features_list)

        self.initial_params["dataset_class"] = MyDatasetTransformer
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {
            "sequence_features_list":sequence_features_list, 
            "continuous_features_list":continuous_features_list,
            "bssid_features_list":bssid_features_list,
            "fq_features_list":fq_features_list,
            "last_query_flag":last_query_flag,
            "max_seq":max_seq,
            "pad_num":pad_num,
            "label_col":["x", "y", "floor"],

        }

        seq_dim_pairs_list = []
        print("sequence_features_list")
        for col in sequence_features_list:
            # if "wb_" in col:
            #     #dim = 29241 #29634
            #     #emb_dim = ninp
            #     continue
            # elif "wifi_frequency_" in col:
            #     continue
            if col == "site_id":
                dim = 24
                emb_dim = 2
            elif col == "site_floor":
                dim = 139+1
                emb_dim = 70
            else:
                continue
            #dim = int(df_all[col].nunique())
            pair = (dim, emb_dim)
            seq_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))
        

        self.model = SimpleTransformer(#LastLSTM(#LastQueryTransformer( #
                            n_numerical_features=len(continuous_features_list), 
                            emb_dim_pairs_list=seq_dim_pairs_list,
                            num_bssid_features=len(bssid_features_list),
                            num_fq_features=len(fq_features_list),
                            last_query_flag=last_query_flag,
                            
                            )


class LastQueryLSTM(PytrochLightningBase):
    def __init__(self,
                #f_all, 
                sequence_features_list, 
                continuous_features_list,
                weight_list,
                gl_norm_dict,
                max_seq=5,
                #ninp=32,
                # nhead=1,
                # nhid=32,
                # nlayers=1,
                # dropout=0.1,
                pad_num=0,
                last_query_flag=False,
                ):


        super().__init__()


        self.initial_params["dataset_class"] = MyDatasetLSTM2
        self.initial_params["collate_fn"] = None #collate_fn_Transformer
        
        use_feature_cols = sequence_features_list+continuous_features_list

        self.initial_params["dataset_params"] = {
            "sequence_features_list":sequence_features_list, 
            "continuous_features_list":continuous_features_list,
            "weight_list":weight_list,
            "use_feature_cols":use_feature_cols,
            "last_query_flag":last_query_flag,
            "max_seq":max_seq,
            "pad_num":pad_num,
            "label_col":["x", "y"],

        }

        prev_renew_cols = ["prev_x", "prev_y", "cum_rel_x", "cum_rel_y", "next_x", "next_y"]
        prev_renew_idx = [use_feature_cols.index(c) for c in prev_renew_cols]
        prev_renew_col_idx_dict = dict(zip(prev_renew_cols, prev_renew_idx))
       

        seq_dim_pairs_list = []
        print("sequence_features_list")
        for col in sequence_features_list:
            # if "wb_" in col:
            #     #dim = 29241 #29634
            #     #emb_dim = ninp
            #     continue
            # elif "wifi_frequency_" in col:
            #     continue
            if col == "floor":
                dim = gl_norm_dict["n_floor"]
                emb_dim = dim//2
            elif col == "site_id":
                dim = 24
                emb_dim = 12

            elif col == "site_floor":
                dim = 139+1
                emb_dim = 70
            else:
                continue
            #dim = int(df_all[col].nunique())
            pair = (dim, emb_dim)
            seq_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))
        

        self.model = LastLSTM(#LastQueryTransformer( #
                            n_numerical_features=len(continuous_features_list), 
                            emb_dim_pairs_list=seq_dim_pairs_list,
                            last_query_flag=last_query_flag,
                            prev_renew_col_idx_dict=prev_renew_col_idx_dict,
                            
                            )


def prepareModelDir(params, prefix):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / params["model_dir_name"]
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()

        params["path_to_model"] = ppath_to_save_dir / f"{prefix}_train_model.ckpt"

        return params

class DNN_Wrapper_Base(object):


    def __init__(self):
        self.model = None

        loss_f = nn.MSELoss()
        def my_loss_func(y_true, y_pred):
            
            #print(f"y_true:{y_true}")
            #print(f"y_pred:{y_pred}")
            
            #print(f"y_true:{y_true.shape}")
            #print(f"y_pred:{y_pred.shape}")
            
            #import pdb; pdb.set_trace()
            return loss_f(y_pred[:, :2].float(), y_true[:, :2].float()) ##+ loss_f(y_pred[:, 2:].float(), y_true[:, 2:].float())
            #return loss_f(y_pred.float(), y_true.float())

        
        self.initial_params = {
                #""'early_stopping_rounds':50,
                #""'lr':0.001,#0.01,#0.005,
                #"batch_size":64 * 4,
                #"epochs":10000,
                "loss_criterion_function": {"name":"mse", "func": my_loss_func},#nn.BCEWithLogitsLoss()} , #
    
                "eval_metric":"MPE",
                "eval_up":False,
                "val_every":1,
                "dataset_params":{},
                "random_seed_name__":"random_state",
                'num_class':1, #binary classification as regression with value between 0 and 1
                'multi_gpu':False,
                }
        self.edit_params = {}

        self.best_iteration_ = 1

    


    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        params = prepareModelDir(params, self.__class__.__name__)

        self.edit_params = params
        torch.manual_seed(params[params["random_seed_name__"]]) 
        torch.backends.cudnn.enabled = True


        

        

        self.model.cuda()
        #show_cuda("init")
        
        if params["multi_gpu"]:
            self.model = nn.DataParallel(self.model)

        #Define loss criterion
        criterion = params["loss_criterion_function"]["func"] #
        criterion_name = params["loss_criterion_function"]["name"]
        #Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer, mode='max', factor=0.5,
        #         patience=params["early_stopping_rounds"]-2, verbose=True, min_lr=1e-6)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        batch_size = params["batch_size"]
        if batch_size < 0:
            batch_size = X_train.shape[0]

        #label_list = list(y_train.value_counts().to_dict().values())

        #weights = 1 / torch.Tensor(label_list)
        #samples_weights = weights[y_train.values]
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

        data_set_train = params["dataset_class"](X_train, y_train, params["dataset_params"], train_flag=True)
        dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True, collate_fn=params["collate_fn"]) #, sampler=ImbalancedDatasetSampler(data_set_train))
        #show_cuda("prepare dataloader_train")
        
        val_hold_dataloader={}
        if X_valid is not None:
            data_set_val = params["dataset_class"](X_valid, y_valid, params["dataset_params"], train_flag=True)
            dataloader_val = torch.utils.data.DataLoader(data_set_val, batch_size=batch_size, shuffle=False, collate_fn=params["collate_fn"])
            val_hold_dataloader["valid"] = dataloader_val


        if X_holdout is not None:
            data_set_hold = params["dataset_class"](X_holdout, y_holdout, params["dataset_params"], train_flag=True)
            dataloader_hold = torch.utils.data.DataLoader(data_set_hold, batch_size=batch_size, shuffle=False, collate_fn=params["collate_fn"])
            val_hold_dataloader["hold"] = dataloader_hold

        best_score_dict={}
        eval_metric_func_dict = params["eval_metric_func_dict__"]

        n_epoch = params["epochs"]
        if params["eval_up"]:
            last_score = -10000000000000
        else:
            last_score=10000000000000

        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            #wandb.init(config=params)
            wandb.init(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb.config.update(params,  allow_val_change=True)
            wandb.watch(self.model, criterion, log=None)
        
        num_seen = 0
        from tqdm import tqdm
        for i in range(1, n_epoch+1):
            #print('Training epoch {}'.format(i))
            #scheduler.step()

            score_dict={}

            self.model.train()

            train_batch_loss_list = []
            eval_score_dict_list = []
            
            for batch_i, (batch_x_list_, batch_y) in enumerate(tqdm(dataloader_train)):
                #print(f"batch {batch_i} / {len(dataloader_train)}")
                batch_x_list = [b.cuda() for b in batch_x_list_]
                batch_y = batch_y.cuda()
                
                

                #Clear the previous gradients
                optimizer.zero_grad()

                #Precit the output for Given input
                y_pred = self.model(batch_x_list)

                #import pdb; pdb.set_trace()


                #print(batch_y.data.cpu().numpy().shape)
                #print(y_pred.data.cpu().numpy().shape)

                #Compute  loss
                loss = criterion(y_true=batch_y, y_pred=y_pred)
                
                train_batch_loss_list.append(loss.detach().item())
                train_batch_eval_score_dict=calcEvalScoreDict(y_true=batch_y.data.cpu().detach().numpy(), y_pred=y_pred.data.cpu().detach().numpy(), eval_metric_func_dict=eval_metric_func_dict)
                #train_batch_eval_score_dict=calcEvalScoreDict(batch_y, y_pred, eval_metric_func_dict)
                eval_score_dict_list.append(train_batch_eval_score_dict)
                
                del batch_x_list_
                del batch_x_list
                del batch_y
                del y_pred
                #torch.cuda.empty_cache()
                #gc.collect()

                
                #Compute gradients
                loss.backward()
                #Adjust weights
                optimizer.step()
                
                num_seen+=batch_size
                # if not ON_KAGGLE: 
                #     wandb.log({"epoch": i, "loss": loss}, step=num_seen)
                
            #scheduler.step()
                    
            #print(train_batch_loss_list)
            train_loss = np.array(train_batch_loss_list).mean()
            score_dict["train"]=calcBatchMeanEvalScoreDictFromEvalScoreDictList(eval_score_dict_list)
            #best_score_dict["train"]["loss"] = train_loss

            print_text = "Epoch {}, train_loss {}, ".format(i, train_loss)
            for name, score in score_dict["train"].items():
                print_text += "train_{} {}, ".format(name, score)
                if (not ON_KAGGLE) and (params["no_wandb"]==False):
                    wandb.log({name: score})
                
            

            if (i % params["val_every"] == 0):
                for val_hold_name in val_hold_dataloader:
                

                    self.model.eval()
                    with torch.no_grad():


                        val_batch_loss_list = []
                        val_eval_score_dict_list = []

                        for val_batch_x_list_, val_batch_y in val_hold_dataloader[val_hold_name]:
                            val_batch_x_list = [b.cuda() for b in val_batch_x_list_]
                            val_batch_y = val_batch_y.cuda()

                    

                            y_pred_valid = self.model(val_batch_x_list)
                            #print(y_pred_valid.shape) #torch.Size([318, 68, 5])
    
                            loss_val = criterion(y_true=val_batch_y, y_pred=y_pred_valid)
                            val_batch_loss_list.append(loss_val.detach().item())


                            val_batch_eval_score_dict=calcEvalScoreDict(y_true=val_batch_y.data.cpu().numpy(), y_pred=y_pred_valid.data.cpu().numpy(), eval_metric_func_dict=eval_metric_func_dict)
                            #val_batch_eval_score_dict=calcEvalScoreDict(val_batch_y, y_pred_valid, eval_metric_func_dict)
                            val_eval_score_dict_list.append(val_batch_eval_score_dict)

                            del val_batch_x_list_
                            del val_batch_x_list
                            del val_batch_y
                            del y_pred_valid
                            torch.cuda.empty_cache()
                            gc.collect()

                    val_loss = np.array(val_batch_loss_list).mean()            
                    score_dict[val_hold_name]=calcBatchMeanEvalScoreDictFromEvalScoreDictList(val_eval_score_dict_list)
                    #best_score_dict[val_hold_name]["loss"] = val_loss

                    print_text += "{}_loss {}, ".format(val_hold_name, val_loss)
                    for name, score in score_dict[val_hold_name].items():
                        print_text += "{}_{} {}, ".format(val_hold_name, name, score)
                        if (not ON_KAGGLE) and (params["no_wandb"]==False):
                            wandb.log({name: score})

                logger.debug(print_text)

                if X_valid is not None:
                    eval_metric_score = score_dict[val_hold_name][params["eval_metric"]]
                    #scheduler.step(eval_metric_score)
                    

                    if ((params["eval_up"]==True) and (last_score < eval_metric_score)) or ((params["eval_up"]==False) and (last_score > eval_metric_score)):
                        last_score = eval_metric_score
                        best_score_dict = score_dict

                        best_epoch = i
                        self.best_iteration_=best_epoch
                        es_rounds = params["early_stopping_rounds"]
                        
                        if params["multi_gpu"]:
                            self.model.module.setSaveParams(best_epoch, last_score)
                        else:
                            self.model.setSaveParams(best_epoch, last_score)
                        print("save model")
                        torch.save(self.model.state_dict(), str(params["path_to_model"]))
                        # with open(str(params["path_to_model"]), 'wb') as f:
                        #     cloudpickle.dump(self.model, f)

                    else:
                        if es_rounds > 0:
                            es_rounds -=1
                        else:
                            print('EARLY-STOPPING !')
                            print('Best epoch found: {}'.format(best_epoch))
                            print('Exiting. . .')
                            break

        print("VALIDATION score:{}".format(last_score))

        #show_cuda("before del")

        del data_set_train
        del dataloader_train
        del val_hold_dataloader

        if X_valid is not None:
            del data_set_val
            del dataloader_val
        if X_holdout is not None:
            del data_set_hold
            del dataloader_hold
        
        torch.cuda.empty_cache()
        gc.collect()

        #show_cuda("after del")


        if X_valid is not None:
            self.model.load_state_dict(torch.load(str(params["path_to_model"])))
        else:
            torch.save(self.model.state_dict(), str(params["path_to_model"]))


        #show_cuda("after model re load")

        self.best_score_ = best_score_dict
        logger.debug(self.best_score_)
        
        self.feature_importances_ = np.zeros(len(X_train.columns))
        #self.best_iteration_ = best_epoch#self.model.best_iteration_

    def predict(self, X_test):

        dummy_y = pd.DataFrame(np.zeros(X_test.shape), index=X_test.index)
        data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=False)
        dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=self.edit_params["batch_size"], shuffle=False, collate_fn=self.edit_params["collate_fn"])
        
        final_preds = []#np.array([])

        self.model.eval()
        with torch.no_grad():
            for test_batch_x_list_, test_batch_y_dummy in dataloader_test:
                test_batch_x_list = [b.cuda() for b in test_batch_x_list_]
                #print(test_batch_x_list[0].shape)

                y_pred_test = self.model(test_batch_x_list)
                y_pred_test = y_pred_test.data.cpu().numpy()
                #print(y_pred_test.shape)
                final_preds.append(y_pred_test)
                #final_preds = np.concatenate((final_preds, y_pred_test), axis=None)
                

                del test_batch_x_list_
                del test_batch_x_list
                del test_batch_y_dummy
                torch.cuda.empty_cache()
                gc.collect()
        

        del dummy_y
        del data_set_test
        del dataloader_test
        
        torch.cuda.empty_cache()
        gc.collect()

        show_cuda("pred after del")
        
        final_preds = np.concatenate(final_preds)
        #print(final_preds.shape)
        #import pdb; pdb.set_trace()
        return final_preds

    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        ppath_to_model = ppath_to_save_dir / f"model__{prefix}__{model_dir_name}__{self.__class__.__name__}.pkl"
        torch.save(self.model.state_dict(), str(ppath_to_model))

        print(f'Trained nn model was saved! : {ppath_to_model}')
        
        with open(str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json"), 'w') as fp:
            json.dump(bs, fp)

    def procLoadModel(self, model_dir_name, prefix, params):
        self.edit_params = params

        self.model.cuda()
        #show_cuda("init")
        
        if params["multi_gpu"]:
            self.model = nn.DataParallel(self.model)
        
        ppath_to_save_dir = PATH_TO_UPLOAD_MODEL_PARENT_DIR / model_dir_name
        print(f"ppath_to_save_dir : {ppath_to_save_dir}")
        print(list(ppath_to_save_dir.glob(f'model__{prefix}*')))
        #print(list(ppath_to_save_dir.iterdir()))
        
        name_list = list(ppath_to_save_dir.glob(f'model__{prefix}*'))
        if len(name_list)==0:
            print(f'[ERROR] Trained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]


        self.model.load_state_dict(torch.load(str(ppath_to_model)))
        print(f'Trained nn model was loaded! : {ppath_to_model}')
        
        a = int(re.findall('iter_(\d+)__', str(ppath_to_model))[0])
        
        
   
        #print(self.model.best_iteration_ )
        #self.model.best_iteration_ 
        self.best_iteration_ = a
        
        path_to_json = str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json")
        if not os.path.exists(path_to_json):
            print(f'[ERROR] Trained LGB json was NOT EXITS! : {path_to_json}')
            return -1
        with open(path_to_json) as fp:
            self.best_score_ = json.load(fp)
            #self.model._best_score = json.load(fp)

        return 0
    
class DNN_Wrapper(DNN_Wrapper_Base):
    def __init__(self, init_x_num):
        super().__init__()
        self.model = MyDNNmodel(init_x_num)
        #self.initial_params["path_to_model"] = PATH_TO_MODEL_DIR / "DNN_train_model.pkl"
        self.initial_params["dataset_class"] = MyDataset
        

class EmbeddingDNN_Wrapper(DNN_Wrapper_Base):
    def __init__(self, df_all, continuous_features_list, embedding_features_list, emb_dropout_rate=0.25):
        super().__init__()

        emb_dim_pairs_list = []
        print("embedding_features_list")
        for col in embedding_features_list:
            dim = int(df_all[col].nunique())
            pair = (dim, min(50, (dim + 1) // 2))
            emb_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))
        #cat_dims = [int(df_all[col].nunique()) for col in embedding_features_list]
        #emb_dim_pairs_list = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

        
        #continuous_features_list = [col for col in all_columns if col not in embedding_features_list ]
        #print(continuous_features_list)
        num_cont_features = len(continuous_features_list)

        self.model = EmbeddingDNN(emb_dim_pairs_list, num_cont_features, emb_dropout_rate)
        #self.initial_params["path_to_model"] = PATH_TO_MODEL_DIR / "DNN_Emb_train_model.pkl"

        self.initial_params["dataset_params"] = {
            "category_features_list":embedding_features_list, 
            "continuous_features_list":continuous_features_list
            }

        self.initial_params["dataset_class"] = MyDatasetEmbedding







class Transformer_Wrapper(DNN_Wrapper_Base):
    def __init__(self,
                #f_all, 
                sequence_features_list, 
                continuous_features_list,
                max_seq=100,
                ninp=32,
                nhead=1,
                nhid=32,
                nlayers=1,
                dropout=0.1,
                pad_num=0,

                ):


        super().__init__()


        


        self.initial_params["dataset_class"] = MyDatasetTransformer
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        #self.initial_params["path_to_model"] = PATH_TO_MODEL_DIR / "Transformer_train_model.pkl"

        self.initial_params["dataset_params"] = {
            "sequence_features_list":sequence_features_list, 
            "continuous_features_list":continuous_features_list,
            "max_seq":max_seq,
            "pad_num":pad_num,
            "label_col":["x", "y", "floor"],

        }

        seq_dim_pairs_list = []
        print("sequence_features_list")
        for col in sequence_features_list:
            if "wb_" in col:
                dim = 29634
            
            
            #dim = int(df_all[col].nunique())
            pair = (dim, ninp)
            seq_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))
            
        #pair = (44119+1, ninp)
        #pair = (54433, ninp)
        #pair = (61307+1, ninp)
        #pair = (29634, ninp)

        
        
        
        seq_dim_pairs_list.append(pair)
        
        # pair = (24+1, ninp)
        # seq_dim_pairs_list.append(pair)

        self.model = Transformer_DNN(ninp=ninp, nhead=nhead, nhid=nhid, nlayers=nlayers,
                emb_dim_pairs_list=seq_dim_pairs_list, num_continuout_features=len(continuous_features_list), dropout=dropout,
        )

    

    # def predict(self, X_test):

    #     dummy_y = pd.DataFrame(np.zeros(X_test.shape), index=X_test.index)
    #     data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=False)
    #     dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=X_test.shape[0], shuffle=False, collate_fn=self.edit_params["collate_fn"])
        
    #     final_preds_row_id = []
    #     final_preds = []#np.array([])
    #     #np.array([])

    #     self.model.eval()
    #     with torch.no_grad():
    #         for test_batch_x_list_, test_batch_y_dummy in dataloader_test:
    #             test_batch_x_list = [b.cuda() for b in test_batch_x_list_]
    #             #print(test_batch_x_list[0].shape)

    #             y_pred_test = self.model(test_batch_x_list)
    #             y_pred_test = y_pred_test.data.cpu().numpy()

    #             # if self.initial_params["multi_gpu"]:
    #             #     row_id = self.model.module.current_row_id.cpu().numpy()
    #             # else:
    #             #     row_id = self.model.current_row_id.cpu().numpy()
    #             row_id = test_batch_x_list[-2]
    #             src_key_padding_mask = test_batch_x_list[-1]
    #             row_id = row_id[~src_key_padding_mask].data.cpu().numpy()

    #             #print(y_pred_test.shape)
    #             final_preds.append(y_pred_test)
    #             final_preds_row_id.append(row_id)
                

    #             del test_batch_x_list_
    #             del test_batch_x_list
    #             del test_batch_y_dummy
    #             torch.cuda.empty_cache()
    #             gc.collect()
        

    #     del dummy_y
    #     del data_set_test
    #     del dataloader_test
        
    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     show_cuda("pred after del")
        
    #     final_preds = np.concatenate(final_preds)
    #     final_preds_row_id = np.concatenate(final_preds_row_id)
    #     final_preds = pd.DataFrame(final_preds, index=final_preds_row_id)

        
    #     #import pdb; pdb.set_trace()
    #     return final_preds
    








 

class LSTM_Wrapper(DNN_Wrapper_Base):
    def __init__(self, df_all, 
                 continuous_features_list, 
                 embedding_category_features_list,
                 sequence_features_list,
                 sequence_index_col,
                 input_sequence_len_col,
                 output_sequence_len_col,
                 weight_col,
                 num_target,
                 emb_dropout_rate=0.25):
        super().__init__()

        emb_dim_pairs_list = []
        print("embedding_category_features_list")
        for col in embedding_category_features_list:
            dim = int(df_all[col].nunique())
            pair = (dim, min(50, (dim + 1) // 2))
            emb_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))


        seq_dim_pairs_list = []
        print("sequence_features_list")
        for col in sequence_features_list:
            #dim = int(df_all.explode(col)[col].nunique())
            dim = int(df_all[col].nunique())
            pair = (dim, min(50, (dim + 1) // 2))
            #pair = (dim, 100)
            seq_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))


        num_cont_features = len(continuous_features_list)

        self.model = LSTM_DNN(seq_dim_pairs_list, emb_dim_pairs_list, num_cont_features, num_target, emb_dropout=emb_dropout_rate)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")  
        #self.initial_params["path_to_model"] = PATH_TO_MODEL_DIR / f"LSTM_train_model_{now}.pkl"

        self.initial_params["dataset_params"] = {
            "sequence_features_list":sequence_features_list, 
            "embedding_category_features_list":embedding_category_features_list, 
            "continuous_features_list":continuous_features_list,
            "transform_class":SequenceTransformer,
            "sequence_index_col":sequence_index_col,
            "input_sequence_len_col":input_sequence_len_col,
            "output_sequence_len_col":output_sequence_len_col,
            "input_sequence_dummy_len_col":"seq_length_dummy",
            "output_sequence_dummy_len_col":"seq_scored_dummy",
            "length_validation":False,
            'weight_col':weight_col,
            }

        self.initial_params["dataset_class"] = MyDatasetLSTM

        self.initial_params['lr'] = 0.01
        self.initial_params["batch_size"]= 32#64
        self.initial_params['early_stopping_rounds']=50
        self.initial_params["epochs"]=10000
        
    
    def predict(self, X_test):

        dummy_y = pd.DataFrame(np.zeros(X_test.shape), index=X_test.index)
        data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=False)
        dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=X_test.shape[0], shuffle=False)
        
        
        final_preds = []#np.array([])

        self.model.eval()
        with torch.no_grad():
            for test_batch_x_list_, test_batch_y_dummy in dataloader_test:
                test_batch_x_list = [b.cuda() for b in test_batch_x_list_]
                #print(test_batch_x_list[0].shape)

                y_pred_test = self.model(test_batch_x_list)
                y_pred_test = y_pred_test.data.cpu().numpy()
                
                #transform for mRNA dataset
                for batch_i in range(y_pred_test.shape[0]):
                    output_len = test_batch_x_list_[1][batch_i].numpy()
                    input_len = test_batch_x_list_[2][batch_i].numpy()
                    
                    tmp = y_pred_test[batch_i]
                    final_preds.append(tmp)
                    np_zero = np.zeros(((input_len - output_len), tmp.shape[1])) 
                    final_preds.append(np_zero)
                    

                #final_preds.append(y_pred_test)
                #final_preds = np.concatenate((final_preds, y_pred_test), axis=None)
                

                del test_batch_x_list_
                del test_batch_x_list
                del test_batch_y_dummy
                torch.cuda.empty_cache()
                gc.collect()

        del dummy_y
        del data_set_test
        del dataloader_test
        
        torch.cuda.empty_cache()
        gc.collect()

        #show_cuda("pred after del")
        
        final_preds = np.concatenate(final_preds) 

        
        return final_preds


class Graph_Wrapper(DNN_Wrapper_Base):
    def __init__(self, df_all, 
                 node_feature_list,
                 edge_adjacent_feature_list, 
                 edge_connect_feature_list,
                 node_index_col,
                 edge_connect_list,
                 sequence_index_col,
                 input_sequence_len_col,
                 output_sequence_len_col,
                 weight_col,
                 num_target,
                 emb_dropout_rate=0.25):
        super().__init__()

        

        #self.model = Graph_DNN(num_node_features=len(node_feature_list), num_target=num_target, emb_dropout=emb_dropout_rate)
        self.model = MyDeeperGCN(
                            num_node_features=(len(node_feature_list)+1),
                            num_edge_features=(len(edge_adjacent_feature_list)+2+len(edge_connect_feature_list)+2+4),
                            num_classes=num_target)


        #now = datetime.now().strftime("%Y%m%d_%H%M%S")  
        #self.initial_params["path_to_model"] = PATH_TO_MODEL_DIR / f"Graph_train_model_{now}.pkl"

        self.initial_params["dataset_params"] = {
            "node_feature_list":node_feature_list, 
            "edge_adjacent_feature_list":edge_adjacent_feature_list,
            "edge_connect_feature_list":edge_connect_feature_list,
            "node_index_col":node_index_col, 
            "edge_connect_list":edge_connect_list,
            "sequence_index_col":sequence_index_col,
            "input_sequence_len_col":input_sequence_len_col,
            "output_sequence_len_col":output_sequence_len_col,
            "input_sequence_dummy_len_col":"seq_length_dummy",
            "output_sequence_dummy_len_col":"seq_scored_dummy",
            "length_validation":False,
            'weight_col':weight_col,
            "ppath_to_saved_data_dir": PROC_DIR / "pair_sequence",
            }

        self.initial_params["dataset_class"] = MyDatasetGraph

        self.initial_params['lr'] = 0.001

        self.initial_params["batch_size"]= 64
        self.initial_params['early_stopping_rounds']=50
        self.initial_params["epochs"]=10000

    
    def predict(self, X_test):
        #show_cuda("pred init")
        dummy_y = pd.DataFrame(np.zeros(X_test.shape), index=X_test.index)
        data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=False)
        
        dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=X_test.shape[0], shuffle=False)
        #show_cuda("prepare test dataloader")
        final_preds = []#np.array([])

        self.model.eval()
        #show_cuda("prepare test dataloader")
        with torch.no_grad():
            for test_batch_x_list_, test_batch_y_dummy in dataloader_test:
                test_batch_x_list = [b.cuda() for b in test_batch_x_list_]
                #print(test_batch_x_list[0].shape)
                #show_cuda("prepare batch")

                y_pred_test = self.model(test_batch_x_list)
                #show_cuda("pred y_pred_test1")
                y_pred_test = y_pred_test.data.cpu().numpy()

                #show_cuda("pred y_pred_test2")

                
                #transform for mRNA dataset
                for batch_i in range(y_pred_test.shape[0]):
                    output_len = test_batch_x_list_[1][batch_i].detach().item()
                    input_len = test_batch_x_list_[2][batch_i].detach().item()

                    #show_cuda("output_len, input_len")
                    
                    tmp = y_pred_test[batch_i]
                    final_preds.append(tmp)
                    np_zero = np.zeros(((input_len - output_len), tmp.shape[1])) 
                    final_preds.append(np_zero)
                    

                #final_preds.append(y_pred_test)
                #final_preds = np.concatenate((final_preds, y_pred_test), axis=None)
                

                del test_batch_x_list_
                del test_batch_x_list
                del test_batch_y_dummy
                torch.cuda.empty_cache()
                gc.collect()

        del dummy_y
        del data_set_test
        del dataloader_test
        
        torch.cuda.empty_cache()
        gc.collect()

        #show_cuda("pred after del")
        
        final_preds = np.concatenate(final_preds) 

        
        return final_preds        

     

        
        
# class datasetDT(object):
    
#     def __init__(self, X, y):
        
#         self.data = [X, y]

        
#     def __len__(self):
#         return len(self.data)
    
#     def __call__(self, idx):
        
#         return self.data[idx]
    
    

# class DeepTable_Wrapper(object):
    
#     def __init__(self, continuous_features_list, embedding_features_list, 
#                  emb_dropout_rate=0.25, regression_flag=True, loss_func="auto", metric_func=['RootMeanSquaredError']):
        
        
        
#         self.conf = ModelConfig(
#             dnn_params={
#                 'hidden_units':((300, 0.3, True),(300, 0.3, True),), #hidden_units
#                 'dnn_activation':'relu',
#             },
#             fixed_embedding_dim=True,
#             embeddings_output_dim=20,
#             nets =['linear','cin_nets','dnn_nets'],
#             #nets=WideDeep,
#             loss = loss_func,
            
#             stacking_op = 'add',
#             output_use_bias = False,
#             auto_encode_label=True,
#             cin_params={
#                 'cross_layer_size': (200, 200),
#                 'activation': 'relu',
#                 'use_residual': False,
#                 'use_bias': True,
#                 'direct': True, 
#                 'reduce_D': False,
#             },
#             metrics= metric_func,  #['RootMeanSquaredError'], #metrics=['AUC'],
#             earlystopping_patience=100,
#             monitor_metric="val_root_mean_squared_error", #
#             #monitor_metric="val_auc",
     
            
#             )

#         self.model = DeepTable(config=self.conf)
        
#         self.initial_params = {

#                 "n_estimators":1000,
#                 "random_seed_name__":"random_state",
#                 }
#         self.edit_params = {}
        
#         self.embedding_features_list = embedding_features_list

#         self.best_iteration_ = 1
        
#         self.regression_flag_ = regression_flag
#         self.ppath_to_weight_dir = PATH_TO_MODEL_DIR / f'DeepTable_modmel_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         os.makedirs(self.ppath_to_weight_dir, exist_ok=True)
#         self.ppath_to_weight = self.ppath_to_weight_dir / 'dt_model.h5'
    
#     def fit(self, X_train, y_train, X_valid, y_valid, X_holdout=None, y_holdout=None, params=None):
        
 
        
#         best_score_dict={}
#         eval_metric_func_dict = params["eval_metric_func_dict__"]
#         print(eval_metric_func_dict)
#         val_name = f"val_{list(eval_metric_func_dict.keys())[0]}"

#         if X_valid is not None:

#             val_Data = datasetDT(X_valid, y_valid)

          

#             # ModelCheckPoint
#             model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#                 str(self.ppath_to_weight), monitor=val_name, mode='min', verbose=1, save_best_only=True, save_weights_only=False)


            
#             depp_model, history = self.model.fit(X_train, y_train, validation_data=val_Data, validation_split=0, 
#                                             batch_size=1024, epochs=self.initial_params["n_estimators"], callbacks=[model_checkpoint])
            
#             depp_model = self.model.load_deepmodel(str(self.ppath_to_weight))
#             # depp_model = load_model(str(self.ppath_to_weight), dt_custom_objects)
#             # for x in inspect.getmembers(self.model, inspect.ismethod):
#             #     print (x[0])
#             self.model._DeepTable__set_model('val', "name", depp_model, history.history)
#             if self.regression_flag_:
#                 y_pred_valid = self.model.predict(X_valid)
#             else:
            
#                 y_pred_valid = self.model.predict_proba(X_valid)
            
#             print(y_pred_valid)
#             print(y_pred_valid.shape)
            
        
#             best_score_dict["train"]  = {}
#             best_score_dict["valid"] = {}
#             for name, f in eval_metric_func_dict.items():
#                 #score  = f(y_valid.values, y_pred_valid.ravel())
#                 score = np.sqrt(mean_squared_error(y_valid.values, y_pred_valid.ravel()))
#                 print(f"y_pred_valid {name} : {score}")
#                 # eval_score_dict[name] = score
#                 print(history.history[f"val_{name}"])
#                 min_index = np.array(history.history[f"val_{name}"]).argmin()

#                 train_score = history.history[name][min_index]
#                 val_score = history.history[f"val_{name}"][min_index]
#                 best_score_dict["train"][name] = train_score
#                 best_score_dict["valid"][name] = val_score
#                 print(f"ep {min_index+1} is min : {best_score_dict}")
#                 self.best_iteration_ = min_index+1
        
#         else:
#             model, history = self.model.fit(X_train, y_train, batch_size=1024, epochs=self.initial_params["n_estimators"])


#             best_score_dict["train"]  = {}
#             for name, f in eval_metric_func_dict.items():
#                 #score  = f(y_valid.values, y_pred_valid.ravel())
#                 #print(f"y_pred_valid {name} : {score}")
#                 # eval_score_dict[name] = score
#                 min_index = self.initial_params["n_estimators"] - 1

#                 train_score = history.history[name][min_index]

#                 best_score_dict["train"][name] = train_score

#                 print(f"ep {min_index+1} is min : {best_score_dict}")
#                 self.best_iteration_ = min_index+1
        
        
#         self.best_score_ = best_score_dict
#         logger.debug(self.best_score_)
#         self.feature_importances_ = np.zeros(len(X_train.columns))
        
       


    
#     def predict(self, X_test):

        
        
#         if self.regression_flag_:
#             y_pred = self.model.predict(X_test)
#         else:
        
#             y_pred = self.model.predict_proba(X_test)
        
#         return y_pred

class LGBWrapper_Base(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = None
        
        self.initial_params = {
            
            
                # 'early_stopping_rounds':50, #50,
                # 'learning_rate': 0.1,#0.5,
                #'feature_fraction': 0.1, #col_sumple_by_tree for xgb
                #'min_data_in_leaf' : 16, #min_child_weight for xgb
                #'bagging_fraction': 0.9, #subsample for xgb [0-1]
                

                # 'reg_alpha': 0, #1, #alpha for xgb
                # 'reg_lambda': 1, #lambda for xgb
                # 'n_jobs': -1,
                # 'n_estimators' : 10000,
                # 'boosting_type': 'gbdt',
                # 'verbose': 1,
                # "random_seed_name__":"random_state",
                # "importance_type":"gain",
                # "min_data":3,
                # "num_leaves":128,
                # "is_training_metric":True,

                'n_jobs': -1,
                #"device":"gpu",

                'boosting_type': 'gbdt',
                "random_seed_name__":"random_state",
                "deal_numpy":False,
                "first_metric_only": True,
                
                'max_depth': 2,
                #'max_bin': 300,
                #'bagging_fraction': 0.9,
                #'bagging_freq': 1, 
                'colsample_bytree': 0.9,
                #'colsample_bylevel': 0.3,
                #'min_data_per_leaf': 2,
                "min_child_samples":8,
                
                'num_leaves': 4,#240,#120,#32, #3000, #700, #500, #400, #300, #120, #80,#300,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                
                # 'max_depth': -1, #3,
                # 'max_bin': 300,
                # 'bagging_fraction': 0.95,#0.85,
                # 'bagging_freq': 1, 
                # 'colsample_bytree': 0.85,
                # 'colsample_bynode': 0.85,
                # 'min_data_per_leaf': 16,#5,#16,#25,
                # 'num_leaves': 300,#240,#120,#32, #3000, #700, #500, #400, #300, #120, #80,#300,
                # 'lambda_l1': 0.5,
                # 'lambda_l2': 0.5,
                
                # "num_leaves": 500,
                # "max_depth": 13,
                # "min_child_weight": 16,
                # "feature_fraction": 0.7829154495406087,
                # "bagging_fraction": 0.6733724601370413,
                # "bagging_freq": 6,
                # "min_child_samples": 33,
                # "lambda_l1": 0.019722059110181,
                # "lambda_l2": 0.4010542626363833,

                
                }
        
        

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        self.edit_params=params
        metric_name = list(params["eval_metric_func_dict__"].keys())[0]
        print(f"metric_name : {metric_name}")
        eval_metric = params["eval_metric_func_dict__"][metric_name]  #["wrmsse"] #params["metric"]
        params["metric"] = "None"
        print(f"----------eval_metric:{callable(eval_metric)}")

        eval_set = [(X_train.values, y_train.values)] if isinstance(X_train, pd.DataFrame) else [(X_train, y_train)]
        eval_names = ['train']
        
        #self.model = self.model.set_params(**extractModelParameters(params, self.model))
        self.model = self.model.set_params(**extractModelParametersWithStr(params, exclude_str="__"))

        if X_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                eval_set.append((X_valid.values, y_valid.values))  
            else:
                eval_set.append((X_valid, y_valid))  

            eval_names.append('valid')


        if X_holdout is not None:
            if isinstance(X_holdout, pd.DataFrame):
                eval_set.append((X_holdout.values, y_holdout.values))
            else:
                eval_set.append((X_holdout, y_holdout))  
            eval_names.append('holdout')


        # if 'cat_cols' in params.keys():
        #     cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
        #     if len(cat_cols) > 0:
        #         categorical_columns = params['cat_cols']
        #     else:
        #         categorical_columns = 'auto'
        # else:
        categorical_columns = 'auto'


        call_back_list = []
        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            wandb.init(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb.config.update(params,  allow_val_change=True)
            call_back_list.append(wandb_callback())


        
        self.model.fit(X=X_train, y=y_train, #X=X_train.values, y=y_train.values,
                    eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                    verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                    categorical_feature=categorical_columns,
                    callbacks=call_back_list,
                    )
        print(self.model)
        self.best_score_ = self.model.best_score_
        logger.debug(self.best_score_)
        self.feature_importances_ = self.model.feature_importances_
        self.best_iteration_ = self.model.best_iteration_

    def predict(self, X_test, oof_flag=True):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_).reshape(-1, 1)
    
    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        ppath_to_model = ppath_to_save_dir / f"model__{prefix}__{model_dir_name}__{self.__class__.__name__}.pkl"
        pickle.dump(self.model, open(ppath_to_model, 'wb'))
        print(f'Trained LGB model was saved! : {ppath_to_model}')
        
        with open(str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json"), 'w') as fp:
            json.dump(bs, fp)
        
    def procLoadModel(self, model_dir_name, prefix):
        
        ppath_to_save_dir = PATH_TO_UPLOAD_MODEL_PARENT_DIR / model_dir_name
        print(f"ppath_to_save_dir : {ppath_to_save_dir}")
        print(list(ppath_to_save_dir.glob(f'model__{prefix}*')))
        #print(list(ppath_to_save_dir.iterdir()))
        
        name_list = list(ppath_to_save_dir.glob(f'model__{prefix}*'))
        if len(name_list)==0:
            print(f'[ERROR] Trained LGB model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]
        # if not os.path.exists(ppath_to_model):
        #     print(f'[ERROR] Trained LGB model was NOT EXITS! : {ppath_to_model}')
        #     return -1

        self.model = pickle.load(open(ppath_to_model, 'rb'))
        print(f'Trained LGB model was loaded! : {ppath_to_model}')
        
        a = int(re.findall('iter_(\d+)__', str(ppath_to_model))[0])
        
        
        self.model._best_iteration= a
        #print(self.model.best_iteration_ )
        #self.model.best_iteration_ 
        self.best_iteration_ = self.model.best_iteration_
        
        path_to_json = str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json")
        if not os.path.exists(path_to_json):
            print(f'[ERROR] Trained LGB json was NOT EXITS! : {path_to_json}')
            return -1
        with open(path_to_json) as fp:
            self.best_score_ = json.load(fp)
            #self.model._best_score = json.load(fp)

        return 0

    
class LGBWrapper_regr(LGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        self.model = lgb.LGBMRegressor()
        print(f"lgb version : {lgb.__version__}")
        
        self.initial_params['objective'] = 'regression' #torch_rmse #'regression'
        self.initial_params['metric'] = 'mae'
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        if isinstance(y_train, pd.DataFrame):
            assert y_train.shape[1] == 1
            y_train = y_train.iloc[:,0]
        
        
        if y_valid is not None:
            if isinstance(y_valid, pd.DataFrame):
                assert y_valid.shape[1] == 1
                y_valid = y_valid.iloc[:,0]
                
        if y_holdout is not None:
            if isinstance(y_holdout, pd.DataFrame):
                assert y_holdout.shape[1] == 1
                y_holdout = y_holdout.iloc[:,0]
                
        
        super().fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_holdout=X_holdout, y_holdout=y_holdout, params=params)



class LGBWrapper_cls(LGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        
        self.model = lgb.LGBMClassifier()
        #self.model = lgb

        self.initial_params['objective'] = 'multiclass'
        self.initial_params['num_class'] = 4
        #self.initial_params['metric'] = 'binary_logloss'
        #self.initial_params['is_unbalance'] = True

    def proc_predict(self, X_test, oof_flag=False):

        if oof_flag:

            pred = self.model.predict(X_test, num_iteration=self.model.best_iteration_)
        else:
            pred = self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
            
            
            # X_test["path"] = X_test.index.map(lambda x: x.split("_")[1])

            # df_pred = X_test[["path"]]
            # df_pred["floor"] = pred
            # df_pred.groupby("path")["floor"].apply(lambda x: x.mode())

            pdb.set_trace()
        return pred



    #def predict_proba(self, X_test):
    def predict(self, X_test, oof_flag=False):
        if (self.model.objective == 'binary') :
            #print("X_test b:", X_test)
            #print("X_test:shape b", X_test.shape)
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            #return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
            #pred = self.proc_predict(X_test, oof_flag)

            return self.model.predict(X_test, num_iteration=self.model.best_iteration_)
    
    def predict_proba(self, X_test):
        #print("X_test:", X_test)
        #print("X_test:shape", X_test.shape)
        return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
    
class XGBWrapper_Base(object):
    
    def __init__(self):
        self.model = None
        
        
        self.initial_params={
                'eta':0.1,
                'max_depth':9, #3, 4, 5, 6, 7, 8, 9
                "min_child_weight":8.0, #0.1 - 10.0
                "gamma":0.0, #1e-8 - 1.0
                "colsample_bytree": 0.8,
                "subsample":0.8,
                "alpha":0.0,
                "lambda":1.0,
                "random_seed_name__":"seed",
                'verbose':1,
                'early_stopping_rounds':200,
                "n_estimators":10000,
                }
        
       
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        self.edit_params=params
        
        eval_metric_func_dict = params["eval_metric_func_dict__"]
        eval_metric = params["eval_metric"]

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        
        self.model = self.model.set_params(**extractModelParameters(params, self.model))
        #self.model = self.model.set_params(**extractModelParametersWithStr(params, exclude_str="__"))

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')


        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        best_score_dict={}
        
        
        self.model.fit(X=X_train, y=y_train,
                    eval_set=eval_set, eval_metric=eval_metric,
                    verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])
        print(self.model)
        
        y_train_pred = self.predict(X_train)
        
        
        
            
        best_score_dict["train"]=calcEvalScoreDict(y_true=y_train, y_pred=y_train_pred, eval_metric_func_dict=eval_metric_func_dict)

        if X_valid is not None:
            #y_valid_pred = self.model.predict(X_valid)
            y_valid_pred = self.predict(X_valid)
            best_score_dict["valid"] = calcEvalScoreDict(y_true=y_valid, y_pred=y_valid_pred, eval_metric_func_dict=eval_metric_func_dict)
            

        if X_holdout is not None:
            
            #y_holdout_pred = self.model.predict(X_holdout)
            y_holdout_pred = self.predict(X_holdout)
            best_score_dict["holdout"] = calcEvalScoreDict(y_true=y_holdout, y_pred=y_holdout_pred, eval_metric_func_dict=eval_metric_func_dict)


        self.best_score_ = best_score_dict
        logger.debug(self.best_score_)
        self.feature_importances_ = np.zeros(len(X_train.columns))
        
        self.best_iteration_ = self.model.get_booster().best_iteration

        self.best_iteration_ = self.model.get_booster().best_iteration


    def predict(self, X_test):
        
        return self.model.predict(X_test)
    

    
    
class XGBWrapper_regr(XGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        self.model = xgb.XGBRegressor()


        self.initial_params['objective'] = 'reg:squarederror'
        self.initial_params['eval_metric'] = 'rmse'
    
    
class XGBWrapper_cls(XGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier()
        

        self.initial_params['objective'] = 'binary:logistic'
        self.initial_params['eval_metric'] = 'auc'
        
    
    def predict(self, X_test):
        if (self.model.objective == 'binary:logistic') :
            #print("X_test b:", X_test)
            #print("X_test:shape b", X_test.shape)
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict_proba(X_test)
    
    def predict_proba(self, X_test):
        #print("X_test:", X_test)
        #print("X_test:shape", X_test.shape)
        return self.model.predict_proba(X_test)
    
class Cat_Base(object):
    
    def __init__(self):
        self.model = None
        
        
        self.initial_params={
                'learning_rate':0.1,
                'depth':8, #3, 4, 5, 6, 7, 8,
                # "min_child_weight":8.0, #0.1 - 10.0
                # "gamma":0.0, #1e-8 - 1.0
                # "colsample_bytree": 0.8,
                # "subsample":0.8,
                # "alpha":0.0,
                # "lambda":1.0,
                
                "random_seed_name__":"random_seed",
                'verbose':100,
                'early_stopping_rounds':200,
                "n_estimators":10000,
                "task_type":"GPU",
                "use_best_model":True,
                
                #"calc_feature_importance":True,
                
                }
        
       
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        self.edit_params=params
        
        eval_metric_func_dict = params["eval_metric_func_dict__"]
        #eval_metric = params["eval_metric"]

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        
        #self.model = self.model.set_params(**extractModelParameters(params, self.model))
        self.model = self.model.set_params(**extractModelParametersWithStr(params, exclude_str="__"))
        print(self.model.get_params())

        if X_valid is not None:
            eval_set = [(X_valid, y_valid)]
            eval_names = ['valid']


        if X_holdout is not None:
            eval_set = [(X_holdout, y_holdout)]
            eval_names = ['holdout']

        best_score_dict={}
        
        
        self.model.fit(X=X_train, y=y_train,
                    eval_set=eval_set,  #eval_metric=eval_metric,
                    verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds']
                    )
        print(self.model)
        
        y_train_pred = self.predict(X_train)
        
        
        
            
        best_score_dict["train"]=calcEvalScoreDict(y_true=y_train, y_pred=y_train_pred, eval_metric_func_dict=eval_metric_func_dict)

        if X_valid is not None:
            #y_valid_pred = self.model.predict(X_valid)
            y_valid_pred = self.predict(X_valid)
            best_score_dict["valid"] = calcEvalScoreDict(y_true=y_valid, y_pred=y_valid_pred, eval_metric_func_dict=eval_metric_func_dict)
            

        if X_holdout is not None:
            
            #y_holdout_pred = self.model.predict(X_holdout)
            y_holdout_pred = self.predict(X_holdout)
            best_score_dict["holdout"] = calcEvalScoreDict(y_true=y_holdout, y_pred=y_holdout_pred, eval_metric_func_dict=eval_metric_func_dict)


        self.best_score_ = best_score_dict
        logger.debug(self.best_score_)
        self.feature_importances_ = np.zeros(len(X_train.columns))
        
        self.best_iteration_ = self.model.get_best_iteration()


    def predict(self, X_test):
        
        return self.model.predict(X_test)
        
class Cat_regr(Cat_Base):
    """
    A wrapper for catboost model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        self.model = CatBoostRegressor()

        self.initial_params['loss_function'] = 'RMSE'
        self.initial_params['eval_metric'] = 'RMSE'    