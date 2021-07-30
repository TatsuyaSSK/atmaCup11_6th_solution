# coding:utf-8

import os
import pathlib
import numpy as np
import pandas as pd

import lightgbm as lgb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger




from DNNmodel import *
from utils import *



def extractModelParameters(original_param, model):
    
    model_params_keys = model.get_params().keys()
    model_params = {}
    for k, v in original_param.items():
        if k in model_params_keys:
            model_params[k] = v
    print(model_params)
    return model_params

def extractModelParametersWithStr(original_param, exclude_str="__"):

    
    model_params = {}
    for k, v in original_param.items():
        if not exclude_str in k:
            model_params[k] = v
    print(model_params)
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
            
            #for atam11, commentout!
            if PROJECT_NAME == "atma11":
                df_pred = pd.read_csv(str(f.parent/pred_f_name))[self.target_col]
            else:
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
        print(self.best_score_)
        
        self.setFeatureImportance(X_train.columns)

    def predict(self, X_test, oof_flag=True):
        
        if self.rate_list_ == None:
            print(f"X_test : {X_test}")
            print(f"X_test.mean(axis=1) : {X_test.mean(axis=1)}")
            return X_test.mean(axis=1).values
        else:
            pred = np.zeros(len(X_test))
            for c, r in zip(X_test.columns, self.rate_list_):
                pred += (X_test[c] * r).values
            return pred.reshape(-1, 1)

    def setFeatureImportance(self, columns_list):
        self.feature_importances_ = np.zeros(len(columns_list))
    

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
        print(self.best_score_)
        
        self.feature_importances_ = np.zeros(len(X_train.columns))
        #self.best_iteration_ = best_epoch#self.model.best_iteration_

        #if wandb_logger is not None:
            #wandb_logger.finalize("success")
            #wandb.finish()

    def predict(self, X_test, oof_flag=False):
        
        num_tta = self.edit_params["num_tta"]
        

        batch_size=self.edit_params["batch_size"] #if oof_flag else 1 #
        dummy_y = pd.DataFrame(np.zeros((X_test.shape[0], 1)), index=X_test.index)
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
    def __init__(self, base_name, img_size, num_out, regression_flag, salient_flag):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = SupConDataset
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {"img_size":img_size, "salient_flag":salient_flag}
        in_channels=4 if salient_flag else 3
        #self.model = SupConModel(base_name="resnet18", in_channels=in_channels,) #num_out=num_out, regression_flag=regression_flag)
        self.model = SSLsimSiam(base_name=base_name, in_channels=in_channels,)
class ResNet_Wrapper(PytrochLightningBase):
    def __init__(self, base_name, img_size, num_out, regression_flag, salient_flag):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = MyDatasetResNet
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {"img_size":img_size, "salient_flag":salient_flag, "regression_flag":regression_flag}
        
        self.model = myResNet(base_name=base_name, num_out=num_out, regression_flag=regression_flag)


class multiLabelNet(PytrochLightningBase):
    def __init__(self, base_name, img_size, num_out, regression_flag, salient_flag, tech_weight, material_weight):
        
        super().__init__()
        
        self.initial_params["dataset_class"] = MyDatasetResNet
        self.initial_params["collate_fn"] = None #collate_fn_Transformer

        self.initial_params["dataset_params"] = {"img_size":img_size, "salient_flag":salient_flag, "regression_flag":regression_flag}
        
        in_channels=4 if salient_flag else 3
        #'efficientnet_b1', "resnet18"  , "resnet18d"
        self.model = myMultilabelNet(base_name=base_name, in_channels=in_channels,num_out=num_out, regression_flag=regression_flag, tech_weight=tech_weight, material_weight=material_weight)




def prepareModelDir(params, prefix):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / params["model_dir_name"]
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()

        params["path_to_model"] = ppath_to_save_dir / f"{prefix}_train_model.ckpt"

        return params





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
                
                'max_depth': 3,
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
        print(self.best_score_)
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
                
        #pdb.set_trace()
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
 