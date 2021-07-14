# coding:utf-8

import argparse

import os
import pathlib
import numpy as np
import pandas as pd

from datetime import datetime

#from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss



from utils import *
from atma11_utils import * 

#from dt_utils import wrmsse_simple_tf
#from m5_utils import my_squared_error, loss_wrmsse, torch_wrmsse, wrmsse_simple, test3, custom_asymmetric_train, my_loss_rmse, visualizeCompLV
from model_wrappers import *
from eda import procEDA
from MyFoldSplit import TournamentGroupKFold, DummyKfold, SeasonKFold, myStratifiedKFold, myStratifiedKFoldWithGroupID, siteStratifiedPathGroupKFold, StratifiedKFoldWithGroupID
#from preprocess import procTest





from log_settings import MyLogger

my_logger = MyLogger()
logger = my_logger.generateLogger("train", LOG_DIR+"/train.log").getChild(__file__)

if not ON_KAGGLE:
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")






def predictNanfromModel(_df, col:str, drop_cols:list=[], nan_pos_flag = False):

    df = _df.drop(columns=drop_cols)
    print(df.columns)

    num_class = df[col].dropna().nunique()
    print("num_class : {}".format(num_class))



    if nan_pos_flag:
        df_train = df.loc[df["{}_nan_pos".format(col)] == 0]
        df_train = df_train.drop("{}_nan_pos".format(col), axis=1)
        df_test = df.loc[df["{}_nan_pos".format(col)] == 1]
        df_test = df_test.drop(columns=[col, "{}_nan_pos".format(col)])
    else:
        df_train = df.loc[df[col].isnull()==False]
        df_test = df.loc[df[col].isnull()].drop(col, axis=1)

    df_X_train=df_train.drop(col, axis=1)
    df_y_train=df_train[col]
    print(df_train.shape)
    print(df_test.shape)
    print(df_X_train.shape)
    print(df_y_train.shape)
    print(df_y_train.value_counts())



    n_fold = 10
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    mt = MainTransformer()
    ft = FeatureTransformer()
    transformers = {'ft': ft}






    use_columns = df_X_train.columns

    model_wrapper = LGBWrapper_cls()

    params=model_wrapper.initial_params
    params["verbose"]=0


    if num_class == 2:
        eval_metric_name = "auc"
        params["eval_metric_func_dict__"] = {eval_metric_name:roc_auc_score}
        params["objective"] = "binary"
        params['metric'] = eval_metric_name
    else:
        #eval_metric_name = "multi_logloss"
        eval_metric_name = "multi_error"
        params["eval_metric_func_dict__"] = {eval_metric_name:accuracy_score} # , "acc":accuracy_score}
        params["objective"] = "multiclass"
        params['metric'] = eval_metric_name
        params["num_class"] = num_class


    params["n_estimators"]=100
    params['reg_alpha'] = 0.0
    params['reg_lambda'] = 0.0
    params['max_depth'] = -1


    model = RegressorModel(model_wrapper=model_wrapper, columns=use_columns)
    model.fit(X=df_X_train, y=df_y_train, X_holdout=None, y_holdout=None, folds=folds, eval_metric=eval_metric_name, params=params, preprocesser=mt, transformers=transformers)



    y_pred_proba = model.predict(df_test, proba=True)
    df_test["pred_nan_{}".format(col)] = np.argmax(y_pred_proba, axis=1)

    logger.debug("after test nan:")
    logger.debug(df_test["pred_nan_{}".format(col)].value_counts(dropna=False))



    _df["pred_nan_{}".format(col)] = _df[col]

    logger.debug("before df with nan:")
    logger.debug(_df["pred_nan_{}".format(col)].value_counts(dropna=False))
    _df.loc[df_test.index, "pred_nan_{}".format(col)] = df_test["pred_nan_{}".format(col)]

    logger.debug("after df with nan:")
    logger.debug(_df["pred_nan_{}".format(col)].value_counts(dropna=False))

    return _df






class RegressorModel(object):


    def __init__(self, columns: list = None, model_wrapper=None):



        """

        :param original_columns:
        :param model_wrapper:
        """
        self.columns = columns
        self.model_wrapper = model_wrapper
        self.result_dict = {}
        self.train_one_fold = False
        self.preprocesser = None
        self.random_seed_list = [2020, 73, 28, 2018, 334]


        self.deal_numpy = model_wrapper.initial_params.get("deal_numpy" , False)

    def fit(self, X: pd.DataFrame, y,
            X_holdout: pd.DataFrame = None, y_holdout=None,
            folds=None,
            params: dict = {"verbose":1},
            eval_metric='rmse',
            cols_to_drop: list = None,
            preprocesser=None,
            transformers: dict = None,
            post_processor=None,
            adversarial: bool = False,
            plot: bool = True,
            seed_averaging:bool = True,
            permutation_feature:bool = False,
            ):

        logger.debug("===========================================")
        logger.debug("[MODEL] ::: {}".format(self.model_wrapper.__class__.__name__))
        logger.debug("[Parameters] ::: {}".format(params))

        """
        Training the model.

        :param X: training data
        :param y: training target
        :param X_holdout: holdout data
        :param y_holdout: holdout target
        :param folds: folds to split the data. If not defined, then model will be trained on the whole X
        :param params: training parameters
        :param eval_metric: metric for validataion
        :param cols_to_drop: list of columns to drop (for example ID)
        :param preprocesser: preprocesser class
        :param transformers: transformer to use on folds
        :param adversarial
        :return:
        """

        if folds is None:
            folds = DummyKfold(n_splits=params["fold2"], random_state=42)
            self.train_one_fold = True

        self.columns = X.columns if self.columns is None else self.columns
        self.feature_importances = pd.DataFrame(columns=['feature', 'importance'])
        self.permutation_feature_df_list=[]

        self.trained_transformers = {k: [] for k in transformers} if transformers != None else {}
        self.transformers = transformers
        self.post_processor = post_processor
        self.models = []
        self.best_iterations = []
        self.folds_dict = {}
        self.eval_metric = eval_metric
        n_target = params["num_class"] if "num_class" in params.keys() else 1
        #self.oof = np.zeros((len(X), n_target)) if n_target==1 else pd.DataFrame(index=X.index, columns= y.columns)



        self.n_target = n_target
        self.valid_indices = []
        self.target_name = params["target_name"]
        self.target_name_idx = params["target_name_idx"]

        # cpu_stats("before X[self.columns]")
        # drop_cols = [col for col in X.columns if not col in self.columns ]
        # #X = X.loc[:, self.columns]
        # X.drop(columns=drop_cols, inplace=True)
        # cpu_stats("after X[self.columns]")

        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            wb_run_name_base = wandb.util.generate_id()



        if params["pred_only"]:


            plot=False
            self.cols_to_drop=None

            self.df_oof = pd.DataFrame(index=[0],columns=self.target_name)


            for fold_n in range(folds.n_splits):

                self.folds_dict[fold_n] = {}

                model = copy.deepcopy(self.model_wrapper)
                model.procLoadModel(params["model_dir_name"], prefix=f"fold_{fold_n}", params=params)

                self.folds_dict[fold_n]['scores'] = model.best_score_

                #model.feature_importances_ = np.zeros(1)
                self.models.append(model)

                #if hasattr(model, "best_iteration_"):
                self.best_iterations.append(model.best_iteration_)
                print(f"add best_iterations : {self.best_iterations}")


        else:

            self.df_oof = pd.DataFrame(index=X.index, columns= y.columns)

            if X_holdout is not None:
                X_holdout = X_holdout[self.columns]


            if preprocesser is not None:
                cpu_stats("before preprocesser")
                self.preprocesser = preprocesser
                self.preprocesser.fit(X, y)
                X = self.preprocesser.transform(X, y)
                #self.columns = X.columns.tolist()
                if X_holdout is not None:
                    X_holdout = self.preprocesser.transform(X_holdout)

                cpu_stats("after preprocesser")



            print(f"FOLD {folds.__class__.__name__}")

            group=None

            if folds.__class__.__name__ == "GroupKFold":
                group = X['Season']

            fold_y = y
            #if folds.__class__.__name__ == "StratifiedKFoldWithGroupID":
            #    fold_y = X['user_id']

            if params["mid_save"]:
                X.to_pickle(PROC_DIR / f'mid_save_X.pkl')
                X = pd.DataFrame(index=X.index)
                #del X
                gc.collect()
                cpu_stats(f"del X for mid save")
                #X = tmp_X

            cpu_stats(f"before split")
            for fold_n, (train_index, valid_index) in enumerate(folds.split(X, fold_y, group)):

                cpu_stats(f"in fold {fold_n+1}")


                if seed_averaging:
                    params[params["random_seed_name__"]] = self.random_seed_list[fold_n % len(self.random_seed_list)]
                else:
                    params[params["random_seed_name__"]] = self.random_seed_list[0]

                if X_holdout is not None:
                    X_hold = X_holdout.copy()
                else:
                    X_hold = None

                print('Fold {} started at {}'.format(fold_n + 1, time.ctime()))
                self.folds_dict[fold_n] = {}

                if self.train_one_fold:

                    cpu_stats("before X_train = X[self.columns]")

                    # if set(X.columns) == set(self.columns):
                    #     X_train = X.to_numpy()
                    #     cpu_stats("after X_train = X[self.columns]")
                    # else:
                    if self.deal_numpy:
                        X_train = X[self.columns].to_numpy()
                        y_train = y.to_numpy()
                    else:
                        X_train = X[self.columns]
                        y_train = y
                    cpu_stats("after  y_train = y.to_numpy()")
                    X_valid = None
                    y_valid = None

                    
                else:


                    if set(X.columns) == set(self.columns):
                        print("set(X.columns) == set(self.columns)")
                        if self.deal_numpy:

                            X_train, X_valid = X.iloc[train_index].to_numpy(), X.iloc[valid_index].to_numpy()
                        else:
                            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
                    else:
                        if self.deal_numpy:
                            X_train, X_valid = X[self.columns].iloc[train_index].to_numpy(), X[self.columns].iloc[valid_index].to_numpy()
                        else:
                            X_train, X_valid = X[self.columns].iloc[train_index], X[self.columns].iloc[valid_index]
                    if self.deal_numpy:
                        y_train, y_valid = y.iloc[train_index].to_numpy(), y.iloc[valid_index].to_numpy()
                    else:
                        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

               


                    self.valid_indices.extend(valid_index)


                cpu_stats("fold split")

                
                datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_hold, 'y_train': y_train}
                cpu_stats("datasets make")
                X_train, X_valid, X_hold = self.transform_(datasets, cols_to_drop)
                cpu_stats("after transform")

                self.folds_dict[fold_n]['columns'] = X_train.columns.tolist()
                logger.debug(self.folds_dict[fold_n]['columns'])

                print("y_train")
                print(y_train.mean())

                if X_valid is not None:
                    print("y_valid")
                    print(y_valid.mean())


                print('Fold {} started at {}'.format(fold_n + 1, time.ctime()))

                model = copy.deepcopy(self.model_wrapper)

                # if adversarial:
                #     X_new1 = X_train.copy()
                #     if X_valid is not None:
                #         X_new2 = X_valid.copy()
                #     elif X_holdout is not None:
                #         X_new2 = X_holdout.copy()
                #     X_new = pd.concat([X_new1, X_new2], axis=0)
                #     y_new = np.hstack((np.zeros((X_new1.shape[0])), np.ones((X_new2.shape[0]))))
                #     X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_new)

                ret = -1
                if params["use_old_file"]:
                    ret = model.procLoadModel(params["model_dir_name"], prefix=f"fold_{fold_n}")

                    model.feature_importances_ = np.zeros(len(X_train.columns)) #X_train.columns))

                if ret < 0:
                    if (not ON_KAGGLE) and (params["no_wandb"]==False):
                        params["wb_run_name"] = wb_run_name_base + f"_fold_{fold_n}"

                    model.fit(X_train, y_train, X_valid, y_valid, X_hold, y_holdout, params=params)
                    model.procModelSaving(params["model_dir_name"], prefix=f"fold_{fold_n}__iter_{model.best_iteration_}", bs=model.best_score_)

                self.folds_dict[fold_n]['scores'] = model.best_score_
                if self.df_oof.shape[0] != len(X):
                    self.df_oof = pd.DataFrame(np.zeros((X.shape[0], self.oof.shape[1])), index=X.index, columns= y.columns)
                if not adversarial:
                    if X_valid is not None:

                        tmp_pred= model.predict(X_valid, oof_flag=True)
                        if isinstance(tmp_pred, np.ndarray):
                            tmp_pred = tmp_pred.reshape(-1, n_target)
                            print(tmp_pred.shape)


                        if len(valid_index) == tmp_pred.shape[0]:

                            self.df_oof.iloc[valid_index] = tmp_pred
                        else:
                            #import pdb; pdb.set_trace()
                            self.df_oof.loc[tmp_pred.index] = tmp_pred

                fold_importance = pd.DataFrame(list(zip(self.folds_dict[fold_n]['columns'], model.feature_importances_)),
                                               columns=['feature', 'importance'])
                self.feature_importances = self.feature_importances.append(fold_importance)

                if permutation_feature:


                    permutation_importance = PermutationImportance(model.model, random_state=self.random_seed_list[fold_n % len(self.random_seed_list)])
                    permutation_importance.fit(X_valid, y_valid)
                    #print(f"Permutation importance fold {fold_n}")

                    exp_df = eli5.explain_weights_df(permutation_importance, feature_names = X_train.columns.values)
                    #print(exp_df)
                    exp_df=exp_df.set_index("feature")
                    #exp_df.to_csv(f'{PATH_TO_FEATURES_DIR}/permutation_feature_imp_fold{fold_n}_' + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
                    self.permutation_feature_df_list.append(exp_df)


                self.models.append(model)

                if hasattr(model, "best_iteration_"):
                    self.best_iterations.append(model.best_iteration_)
                    print(f"add best_iterations : {self.best_iterations}")


                cpu_stats(f"before fold{fold_n+1} del")
                del datasets, X_train, X_valid, y_train, y_valid
                gc.collect()
                cpu_stats(f"after fold{fold_n+1} del")


            self.feature_importances['importance'] = self.feature_importances['importance'].astype(int)

            if permutation_feature:
                df_permutation_total = calcPermutationWeightMean(self.permutation_feature_df_list)
                df_permutation_total.to_csv(f'{PATH_TO_FEATURES_DIR}/permutation_feature_imp_' + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", index=True)
                logger.debug("permutation feature importance weight under 0")
                logger.debug(df_permutation_total.loc[df_permutation_total["weight_mean"]<=0].index)

        # if params['verbose']:
        self.calc_scores_()

        if self.post_processor is not None:
            cpu_stats("before post_preprocesser")

            self.post_processor.fit(X, y)
            X = self.post_processor.transform(X, y)
            if X_holdout is not None:
                X_holdout = self.post_processor.transform(X_holdout)

            cpu_stats("after post_preprocesser")


        if (plot==True) & (ON_KAGGLE==False):

            if self.n_target  == 1:

                #category_cols=[ "part", ]
                #visualizeComp(X.iloc[self.valid_indices], y.iloc[self.valid_indices].values, self.df_oof.iloc[self.valid_indices].values, category_cols=category_cols)

                compPredTarget(self.df_oof.iloc[self.valid_indices].values, y.iloc[self.valid_indices].values, index_list=y.iloc[self.valid_indices].index, title_str="oof_diff")
                # print(classification_report(y, self.oof.argmax(1)))
                fig, ax = plt.subplots(figsize=(16, 12))
                plt.subplot(2, 2, 1)
                self.plot_feature_importance(top_n=20)
                plt.subplot(2, 2, 2)
                self.plot_metric()
                plt.subplot(2, 2, 3)
                plt.hist(y.values.reshape(-1, 1) - self.df_oof.values.reshape(-1, 1))
                plt.title('Distribution of errors')
                plt.subplot(2, 2, 4)
                plt.hist(self.df_oof.values.reshape(-1, 1))
                plt.title('Distribution of oof predictions');

                plt.savefig(os.path.join(str(PATH_TO_GRAPH_DIR), "{}_result_fig.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
                plt.close()


    def loadModels(self, params, eval_metric, folds):


        self.models = []
        self.folds_dict = {}
        self.eval_metric = eval_metric

        self.target_name = params["target_name"]
        self.target_name_idx = params["target_name_idx"]

        for fold_n in range(folds.n_splits):

            self.folds_dict[fold_n] = {}

            model = copy.deepcopy(self.model_wrapper)
            model.procLoadModel(params["model_dir_name"], prefix=f"fold_{fold_n}")

            self.folds_dict[fold_n]['scores'] = model.best_score_

            #model.feature_importances_ = np.zeros(1)
            self.models.append(model)

            #if hasattr(model, "best_iteration_"):
            self.best_iterations.append(model.best_iteration_)
            print(f"add best_iterations : {self.best_iterations}")

        self.calc_scores_()





    def transform_(self, datasets, cols_to_drop):
        if self.transformers is not None:
            for name, transformer in self.transformers.items():

                if name == "targetEncoding":
                    print(name)
                    datasets['X_train'] = transformer.fit(datasets['X_train'], datasets['y_train'])
                    #showNAN(datasets['X_train'])
                else:
                    transformer.fit(datasets['X_train'], datasets['y_train'])
                    datasets['X_train'] = transformer.transform(datasets['X_train'])

                if datasets['X_valid'] is not None:
                    datasets['X_valid'] = transformer.transform(datasets['X_valid'])
                if datasets['X_holdout'] is not None:
                    datasets['X_holdout'] = transformer.transform(datasets['X_holdout'])
                self.trained_transformers[name].append(transformer)
        if cols_to_drop is not None:
            cols_to_drop = [col for col in cols_to_drop if col in datasets['X_train'].columns]

            datasets['X_train'].drop(cols_to_drop, axis=1, inplace=True)
            if datasets['X_valid'] is not None:
                datasets['X_valid'].drop(cols_to_drop, axis=1, inplace=True)
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'].drop(cols_to_drop, axis=1, inplace=True)
        self.cols_to_drop = cols_to_drop

        return datasets['X_train'], datasets['X_valid'], datasets['X_holdout']

    def calc_scores_(self):
        logger.debug("\n")
        datasets = [k for k, v in [v['scores'] for k, v in self.folds_dict.items()][0].items() if len(v) > 0]
        self.scores = {}
        for d in datasets:
            scores = [v['scores'][d][self.eval_metric] for k, v in self.folds_dict.items()]
            print_text = "[{} : {}] CV mean score on {}: {:.4f} +/- {:.4f} std. ::: {}".format(self.target_name_idx, self.target_name, d, np.mean(scores), np.std(scores), scores)
            logger.debug(self.model_wrapper.__class__.__name__)
            logger.debug(print_text)
            if not ON_KAGGLE:
                slack.notify(text=self.model_wrapper.__class__.__name__)
                slack.notify(text=print_text)
            self.scores[d] = np.mean(scores)

    def predict(self, X_test, proba:bool = False, averaging: str = 'usual', arg_max=False):
        """
        Make prediction

        :param X_test:
        :param averaging: method of averaging
        :return:
        """

        # for col in self.columns:
        #     if col in  X_test.columns:
        #         print(f"in : {col}")
        #     else:
        #         print(f"not : {col}")


        X_test = X_test.loc[:,self.columns]


        full_prediction = np.zeros((X_test.shape[0], self.df_oof.shape[1]))

        #with timer("preprocesser"):
        if self.preprocesser is not None:
            #logger.debug("before preprocesser")
            #logger.debug(X_test)
            X_test = self.preprocesser.transform(X_test)
            #logger.debug("after preprocesser")
            #logger.debug(X_test)


        for i in range(len(self.models)):
            if not ON_KAGGLE:
                print(f"prediction : model {i}")

            #with timer(f"copy {i}"):
            X_t = X_test.copy()

            #with timer(f"transformers {i}"):
            for name, transformers in self.trained_transformers.items():
                #print("before transformers : {}".format(name))
                #print(X_t)
                params = {"idx":i}
                X_t = transformers[i].transform(X_t, y=None, params=params)
                #print("after transformers : {}".format(name))
                #print(X_t)

            if self.cols_to_drop is not None:
                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]
                X_t = X_t.drop(cols_to_drop, axis=1)


            #with timer(f"predict {i}"):
            if proba:
                
                y_pred = self.models[i].predict_proba(X_t.to_numpy()).reshape(full_prediction.shape[0], -1)
                if full_prediction.shape[1] != y_pred.shape[1]:
                    full_prediction = np.zeros((y_pred.shape[0], y_pred.shape[1]))


            else:
                if self.deal_numpy:
                    y_pred = self.models[i].predict(X_t.to_numpy()).reshape(-1, full_prediction.shape[1])
                else:
                    y_pred = self.models[i].predict(X_t).reshape(-1, full_prediction.shape[1])


            # del self.models[i]
            # gc.collect()

            # if case transformation changes the number of the rows
            if full_prediction.shape[0] != len(y_pred):
                full_prediction = np.zeros((y_pred.shape[0], self.df_oof.shape[1]))

            if averaging == 'usual':
                full_prediction += y_pred
            elif averaging == 'rank':
                full_prediction += pd.Series(y_pred).rank().values

        full_prediction = full_prediction / len(self.models)

        if arg_max:
            #pdb.set_trace()
            full_prediction = np.argmax(full_prediction, axis=1)

        return full_prediction

    def plot_feature_importance(self, drop_null_importance: bool = True, top_n: int = 10):
        """
        Plot default feature importance.

        :param drop_null_importance: drop columns with null feature importance
        :param top_n: show top n columns
        :return:
        """

        top_feats = self.get_top_features(drop_null_importance, top_n)
        if len(top_feats) > 0:
            feature_importances = self.feature_importances.loc[self.feature_importances['feature'].isin(top_feats)]
            feature_importances['feature'] = feature_importances['feature'].astype(str)
            top_feats = [str(i) for i in top_feats]

            sns.barplot(data=feature_importances, x='importance', y='feature', orient='h', order=top_feats)
            plt.title('Feature importances')

    def get_top_features(self, drop_null_importance: bool = True, top_n: int = 10):
        """
        Get top features by importance.

        :param drop_null_importance:
        :param top_n:
        :return:
        """
        grouped_feats = self.feature_importances.groupby(['feature'])['importance'].mean()
        sort_f = grouped_feats.sort_values(ascending=False)

        logger.debug("Feature Importance")
        logger.debug(sort_f)
        for i in range(len(sort_f)):
            logger.debug("{} : {}".format(sort_f.index[i], sort_f[i]))

        logger.debug("zero Importance")
        logger.debug("{}".format(grouped_feats[grouped_feats == 0].index))

        if drop_null_importance:
            grouped_feats = grouped_feats[grouped_feats != 0]
        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]

    def plot_metric(self):
        """
        Plot training progress.
        Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html

        :return:
        """
        full_evals_results = pd.DataFrame()

        try:
            for model in self.models:
                evals_result = pd.DataFrame()
                for k in model.model.evals_result_.keys():
                    evals_result[k] = model.model.evals_result_[k][self.eval_metric]
                evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
                full_evals_results = full_evals_results.append(evals_result)

            full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,
                                                                                                'variable': 'dataset'})
            sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')
            plt.title('Training progress')
        except Exception as e:
            pass



class MainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, label_decode_cols_dict={}, change_category_cols:list=[], minMax_scaler_cols:list=[], standard_scaler_cols:list = [], label_encoding_cols:list=[], log_list=[], convert_cyclical: bool = False, create_interactions: bool = False, n_interactions: int = 20):
        """
        Main transformer for the data. Can be used for processing on the whole data.

        :param convert_cyclical: convert cyclical features into continuous
        :param create_interactions: create interactions between features
        """

        self.convert_cyclical = convert_cyclical
        self.create_interactions = create_interactions
        self.feats_for_interaction = None
        self.n_interactions = n_interactions

        self.minMax_scaler_cols_ = minMax_scaler_cols
        if len(self.minMax_scaler_cols_) > 0:
            self.mm_sc_ = preprocessing.MinMaxScaler()
        else:
            self.mm_sc_ = None

        self.standard_scaler_cols_ = standard_scaler_cols
        if len(self.standard_scaler_cols_) > 0:
            self.sc_ = StandardScaler()
        else:
            self.sc_ = None

        self.label_encoding_cols_= label_encoding_cols
        if len(self.label_encoding_cols_) > 0:
            self.lbl_ = preprocessing.LabelEncoder()
        else:
            self.lbl_ = None

        self.log_list_ = log_list

        self.change_category_cols_ = change_category_cols
        self.label_decode_cols_dict_ = label_decode_cols_dict

    def fit(self, X, y=None):

        #if self.sc_ != None:
        #    self.sc_.fit(X[self.standard_scaler_cols_])

        if self.create_interactions:
            self.feats_for_interaction = [col for col in X.columns if 'sum' in col
                                          or 'mean' in col or 'max' in col or 'std' in col
                                          or 'attempt' in col]
            self.feats_for_interaction1 = np.random.choice(self.feats_for_interaction, self.n_interactions)
            self.feats_for_interaction2 = np.random.choice(self.feats_for_interaction, self.n_interactions)

        return self

    def transform(self, X, y=None, inplace=True):
        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)

        if len(self.label_decode_cols_dict_) > 0:

            for col, replace_dict in self.label_decode_cols_dict_.items():
                if col in data.columns:
                    data[col].replace(replace_dict, inplace=True)
                    print(f"replace {col} using {replace_dict}")


        if len(self.change_category_cols_) > 0:
            for col in self.change_category_cols_:
                data.loc[:, col] = data.loc[:, col].astype("category")
                print(f"transform ; {col} to {data[col].dtype}")

        if len(self.log_list_) > 0:
            for col in self.log_list_:
                if col in data.columns:
                    data.loc[:, col] = np.log1p(data.loc[:, col])
                    print(f"log transform: {col}")


        if self.mm_sc_ != None:
            data.loc[:, self.minMax_scaler_cols_] = data.loc[:, self.minMax_scaler_cols_].astype("float64")
            for col in self.minMax_scaler_cols_:
                self.mm_sc_.fit(data.loc[:, col].values.reshape(-1, 1))
                data.loc[:, col] = self.mm_sc_.transform(data.loc[:, col].values.reshape(-1, 1))
            #showNAN(data)

        if self.sc_ != None:
            #logger.debug(data[self.standard_scaler_cols_])
            data.loc[:, self.standard_scaler_cols_] = data.loc[:, self.standard_scaler_cols_].astype("float64")
            for col in self.standard_scaler_cols_:
                self.sc_.fit(data.loc[:, col].values.reshape(-1, 1))
                data.loc[:, col] = self.sc_.transform(data.loc[:, col].values.reshape(-1, 1))
                print(f"transform sc_ ; {col} : {data[col].dtype} ")


            #data[self.standard_scaler_cols_] = pd.DataFrame(self.sc_.transform(data[self.standard_scaler_cols_]), columns=self.standard_scaler_cols_)
            #ogger.debug(data[self.standard_scaler_cols_])

        if self.lbl_ != None:
            for col in self.label_encoding_cols_:
                self.lbl_.fit(list(data.loc[:, col].unique()))
                data.loc[:, col] = self.lbl_.transform(list(data.loc[:, col].values))
                #print(sorted(list(data[col].unique())))

        if self.create_interactions:
            for col1 in self.feats_for_interaction1:
                for col2 in self.feats_for_interaction2:
                    data.loc[:, '{}_int_{}'.format(col1, col2)] = data.loc[:, col1] * data.loc[:, col2]

        if self.convert_cyclical:
            data['timestampHour'] = np.sin(2 * np.pi * data['timestampHour'] / 23.0)
            data['timestampMonth'] = np.sind(2 * np.pi * data['timestampMonth'] / 23.0)
            data['timestampWeek'] = np.sin(2 * np.pi * data['timestampWeek'] / 23.0)
            data['timestampMinute'] = np.sin(2 * np.pi * data['timestampMinute'] / 23.0)

#         data['installation_session_count'] = data.groupby(['installation_id'])['Clip'].transform('count')
#         data['installation_duration_mean'] = data.groupby(['installation_id'])['duration_mean'].transform('mean')
#         data['installation_title_nunique'] = data.groupby(['installation_id'])['session_title'].transform('nunique')

#         data['sum_event_code_count'] = data[['2000', '3010', '3110', '4070', '4090', '4030', '4035', '4021', '4020', '4010', '2080', '2083', '2040', '2020', '2030', '3021', '3121', '2050', '3020', '3120', '2060', '2070', '4031', '4025', '5000', '5010', '2081', '2025', '4022', '2035', '4040', '4100', '2010', '4110', '4045', '4095', '4220', '2075', '4230', '4235', '4080', '4050']].sum(axis=1)

        # data['installation_event_code_count_mean'] = data.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #data = reduce_mem_usage(data)
        return data

    def fit_transform(self, X, y=None, inplace=True, **fit_params):

        if inplace:
            data = X
        else:
            data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)




class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, main_cat_features: list = None, num_cols: list = None):
        """

        :param main_cat_features:
        :param num_cols:
        """
        self.main_cat_features = main_cat_features
        self.num_cols = num_cols

    def fit(self, X, y=None):

#         self.num_cols = [col for col in X.columns if 'sum' in col or 'mean' in col or 'max' in col or 'std' in col
#                          or 'attempt' in col]


        return self

    def transform(self, X, y=None, params={}, inplace=True):
        if inplace==True:
            data= X
        else:
            data = copy.deepcopy(X)
#         for col in self.num_cols:
#             data[f'{col}_to_mean'] = data[col] / data.groupby('installation_id')[col].transform('mean')
#             data[f'{col}_to_std'] = data[col] / data.groupby('installation_id')[col].transform('std')

        return data

    def fit_transform(self, X, y=None, inplace=True, **fit_params):
        if inplace==True:
            data= X
        else:
            data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)

class binnigTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, encoding_dict):
        self.encoding_dict = encoding_dict

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None, params={}, inplace=True):
        if inplace==True:
            data= X
        else:
            data = copy.deepcopy(X)

        for col, label_num in self.encoding_dict.items():
            data.loc[:, col] = pd.qcut(data.loc[:, col], label_num, labels=False, duplicates='drop')
            print("col : {}, label:{}".format(col, label_num))
            print(data[col].value_counts())
        return data

    def fit_transform(self, X, y=None, inplace=True, **fit_params):
        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)

class DropFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, drop_columns:list):
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None, params={}, inplace=True):
        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)

        data.drop(columns=self.drop_columns, inplace=True)
        #print(f"drop trainsorm: {self.drop_columns}")


        return data

    def fit_transform(self, X, y=None, inplace=True, **fit_params):
        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)


class TargetEncodingTransormer(BaseEstimator, TransformerMixin):

    def __init__(self, encoding_cols: list):
        self.encoding_cols = encoding_cols
        self.current_idx = -1
        self.transorm_target_mean_list = []
        self.ave_target = -999

    def fit(self, X, y=None, inplace=True):
        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)
        self.ave_target = y.mean()

        df_train_X, dict_all_train_target_mean = targetEncoding(data, y, encoding_cols=self.encoding_cols, _n_splits=4)
        self.transorm_target_mean_list.append(dict_all_train_target_mean)
        self.current_idx += 1


        return df_train_X

    def transform(self, X, y=None, params={}, inplace=True):

        if inplace==True:
            data=X
        else:
            data = copy.deepcopy(X)

        idx = self.current_idx
        if "idx" in params:
            idx = params["idx"]
        dict_all_train_target_mean=self.transorm_target_mean_list[idx]

        for c in self.encoding_cols:
            data[c] = data[c].map(dict_all_train_target_mean[c])

            idx_1_unique = dict_all_train_target_mean[c].unique()
            idx_2_unique = data[c].unique()
            for c2 in idx_2_unique:
                if not c2 in idx_1_unique:
                    print("TARGET ENCORDING ERROR {}: {}".format(c, c2))
            data[c].fillna(self.ave_target , inplace=True)

        return data

    def fit_transform(self, X, y=None, **fit_params):
        print("*****NOT IMPLEMENT!!!*****")
        sys.exit(-1)
        return

#class CategoricalTransformer(BaseEstimator, TransformerMixin):
#
#    def __init__(self, cat_cols=None, drop_original: bool = False, encoder=OrdinalEncoder()):
#        """
#        Categorical transformer. This is a wrapper for categorical encoders.
#
#        :param cat_cols:
#        :param drop_original:
#        :param encoder:
#        """
#        self.cat_cols = cat_cols
#        self.drop_original = drop_original
#        self.encoder = encoder
#        self.default_encoder = OrdinalEncoder()
#
#    def fit(self, X, y=None):
#
#        if self.cat_cols is None:
#            kinds = np.array([dt.kind for dt in X.dtypes])
#            is_cat = kinds == 'O'
#            self.cat_cols = list(X.columns[is_cat])
#        self.encoder.set_params(cols=self.cat_cols)
#        self.default_encoder.set_params(cols=self.cat_cols)
#
#        self.encoder.fit(X[self.cat_cols], y)
#        self.default_encoder.fit(X[self.cat_cols], y)
#
#        return self
#
#    def transform(self, X, y=None):
#        data = copy.deepcopy(X)
#        new_cat_names = ['{}_encoded'.format(col) for col in self.cat_cols]
#        encoded_data = self.encoder.transform(data[self.cat_cols])
#        if encoded_data.shape[1] == len(self.cat_cols):
#            data[new_cat_names] = encoded_data
#        else:
#            pass
#
#        if self.drop_original:
#            data = data.drop(self.cat_cols, axis=1)
#        else:
#            data[self.cat_cols] = self.default_encoder.transform(data[self.cat_cols])
#
#        return data
#
#    def fit_transform(self, X, y=None, **fit_params):
#        data = copy.deepcopy(X)
#        self.fit(data)
#        return self.transform(data)


def simplePredictionSet(df_train, df_test, target_col_list:list,
                        use_columns:list, _folds, setting_params:dict,
                        main_transformer, transformer_dict:dict, post_processor,  model_wrapper,
                        eval_metric_name:str, eval_metric_func_dict:dict, hold_flag=False, permutation_feature=False):

    if setting_params["pred_only"]:
        df_X_train = None
        df_y_train = None

    else:

        cpu_stats("simplePredictionSet")
        
        

        df_y_train=df_train.loc[:, target_col_list]
        cpu_stats("simplePredictionSet : df_y_train create")

        df_train.drop(columns=target_col_list, inplace=True)

        cpu_stats("simplePredictionSet : df_train drop target")
        df_X_train=df_train
        cpu_stats("simplePredictionSet : df_X_train")


    if hold_flag:
        df_X_train, df_X_hold, df_y_train, df_y_hold = train_test_split(df_train.drop(target_col_list, axis=1), df_train.loc[:, target_col_list], test_size=0.3, random_state=2020, stratify=df_train.loc[:, target_col_list])
    else:
        df_X_hold=None
        df_y_hold=None

    cpu_stats("before df_train del")
    del df_train
    gc.collect()
    cpu_stats("after df_train del")

    params=model_wrapper.initial_params
    params["eval_metric_func_dict__"] = eval_metric_func_dict
    params["eval_metric"] = eval_metric_name
    params["target_name"] = target_col_list
    params["target_name_idx"] = setting_params["target_idx"]
    params["n_estimators"] = setting_params["epochs"]
    params["epochs"] = setting_params["epochs"]
    params["early_stopping_rounds"] = setting_params["early_stopping_rounds"]
    params["learning_rate"] = setting_params["learning_rate"]
    params["batch_size"] = setting_params["batch_size"]
    params["num_workers"] = setting_params["num_workers"]
    params["verbose"] = setting_params["verbose"]
    params["model_dir_name"] = setting_params["model_dir_name"]
    params["pred_only"] = setting_params["pred_only"]
    params["pred2_only"] = setting_params["pred2_only"]
    params["mid_save"] = setting_params["mid_save"]
    params["use_old_file"] = setting_params["use_old_file"]
    params["fold2"] = setting_params["fold2"]
    params["num_class"] = setting_params["num_class"]
    params["no_wandb"] = setting_params["no_wandb"]

    params["use_columns"] = use_columns

    if (not ON_KAGGLE) and (params["no_wandb"]==False):
    
        # initialize a new wandb project
        params["wb_group_name"] = params["model_dir_name"]#wandb.util.generate_id()


    model = RegressorModel(model_wrapper=model_wrapper, columns=use_columns)



    model.fit(X=df_X_train, y=df_y_train, X_holdout=df_X_hold, y_holdout=df_y_hold, folds=_folds, 
            eval_metric=eval_metric_name, params=params, preprocesser=main_transformer, transformers=transformer_dict, 
            post_processor=post_processor, permutation_feature=permutation_feature, 
            plot=setting_params["train_plot"])

    df_oof = model.df_oof.iloc[model.valid_indices]

    print(f"df_oof : {df_oof}")



    valid_score = model.scores["valid"]

    model_name = model_wrapper.__class__.__name__

    if (setting_params["mode"] == "ave") | (setting_params["mode"]=="stack"):

        y_pred = model.predict(df_test)
        df_y_pred = pd.DataFrame(y_pred, index=df_test.index, columns=target_col_list)
        print(f"df_y_pred : {df_y_pred}")

    else:

        if _folds.__class__.__name__ == "SeasonKFold":
        #####################
        # train again with full train data using mean of best iterations



            params["n_estimators"] = int(np.array(model.best_iterations).mean())
            params["epochs"] = params["n_estimators"]
            print(f"mean best iterations : {params['n_estimators']}")
            params["early_stopping_rounds"] = params["n_estimators"]+1
            params["model_dir_name"] += "-full"


            model2 = RegressorModel(model_wrapper=model_wrapper, columns=use_columns)
            model2.fit(X=df_X_train, y=df_y_train, X_holdout=df_X_hold, y_holdout=df_y_hold, folds=None, eval_metric=eval_metric_name, params=params, preprocesser=main_transformer, transformers=transformer_dict, post_processor=post_processor, permutation_feature=False, plot=False)

            del model
            gc.collect()

            model = model2



        cpu_stats("before df_X_train, df_y_train  del")
        del df_X_train, df_y_train
        gc.collect()
        cpu_stats("after df_X_train, df_y_train  del")

        if (setting_params["mode"] == "lgb"):
            y_pred = model.predict(df_test, proba=True, arg_max=True)
        else:

            y_pred = model.predict(df_test)
        df_y_pred = pd.DataFrame(y_pred, index=df_test.index, columns=target_col_list)
        print(f"df_y_pred : {df_y_pred}")


    return df_y_pred, df_oof, valid_score, model_name

class StackingWrapper(object):


    def __init__(self, df_train, df_test, target_col:str, _folds, eval_metric_name:str, eval_metric_func_dict:dict,
                stacking_model_wrppers:list, use_columns_list:list, mt_list:list, tf_dict_list:list):

        self.stacking_model_wrppers_ = stacking_model_wrppers
        self.use_columns_list_ = use_columns_list
        self.mt_list_=mt_list
        self.tf_dict_list_ = tf_dict_list
        self.meta_model = self.stacking_model_wrppers_[-1]

        self.initial_params = self.meta_model.initial_params


        self.modelsPrediction(df_train, df_test, target_col, _folds, eval_metric_name, eval_metric_func_dict)


    def modelsPrediction(self, df_train, df_test, target_col:str, _folds, eval_metric_name:str, eval_metric_func_dict:dict):

        y_pred_list=[]
        oof_list = []
        for i, model in enumerate(self.stacking_model_wrppers_[:-1]):
            print("stacking {}: {}".format(i, model))
            y_pred, oof, _, _ = simplePredictionSet(df_train, df_test, target_col,
                            use_columns=self.use_columns_list_[i], _folds=_folds,
                            main_transformer=self.mt_list_[i], transformer_dict=self.tf_dict_list_[i],  model_wrapper=model,
                            eval_metric_name=eval_metric_name, eval_metric_func_dict=eval_metric_func_dict, hold_flag=False)
            y_pred_list.append(y_pred)
            oof_list.append(oof)

        self.setMeta(df_train, df_test, target_col, y_pred_list, oof_list)
#        np_y_pred = np.concatenate(y_pred_list, 1)
#        np_oof = np.concatenate(oof_list, 1)
#        print(np_y_pred.shape)
#        print(np_oof.shape)
#        self.df_meta_test = pd.DataFrame(np_y_pred, index=df_test.index)
#        self.df_meta_train = pd.DataFrame(np_oof, index=df_train.index)
#
#        self.df_meta_train[target_col] = df_train[target_col]


    def setMeta(self, df_train, df_test, target_col:str, y_pred_list:list, oof_list:list):

        np_y_pred = np.concatenate(y_pred_list, 1)
        np_oof = np.concatenate(oof_list, 1)
        print(np_y_pred.shape)
        print(np_oof.shape)
        self.df_meta_test = pd.DataFrame(np_y_pred, index=df_test.index)
        self.df_meta_train = pd.DataFrame(np_oof, index=df_train.index)

        self.df_meta_train[target_col] = df_train[target_col]




    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        #X_train_meta = self.df_meta_train.loc[X_train.index]
        #X_valid_meta = self.df_meta_train.loc[X_valid.index]

        ##BE careful ,this repalce X_valid
        #X_valid = X_valid_meta

        print("X_train:{}".format(X_train.shape))
        print(X_train.columns)
        print("y_train:{}".format(y_train.shape))
        print("X_valid:{}".format(X_valid.shape))
        print(X_valid.columns)
        print("y_valid:{}".format(y_valid.shape))

        self.meta_model.fit(X_train, y_train, X_valid, y_valid, X_holdout, y_holdout, params)
        self.best_score_ = self.meta_model.best_score_
        logger.debug(self.best_score_)
        self.feature_importances_ = self.meta_model.feature_importances_

    def predict(self, X_test):
        print("X_test:{}".format(X_test.shape))

        return self.meta_model.predict(X_test)

class SimpleStackingWrapper(StackingWrapper):


    def __init__(self, df_train, df_test, target_col_list, meta_model, path_to_meta_feature_dir):


        self.meta_model = meta_model
        self.target_col_list = target_col_list
        self.initial_params = self.meta_model.initial_params

        self.df_meta_train, self.df_meta_test = self.setMetaFromFiles(path_to_meta_feature_dir)

        idx1 = set(df_train.index)
        print(df_train)

        self.df_meta_train[target_col_list] = df_train.loc[self.df_meta_train.index, target_col_list]

        idx2 = set(self.df_meta_train.index)
        print(self.df_meta_train)
        print(f"idx1:{len(idx1)}")
        print(f"idx2:{len(idx2)}")

        #for mRNAdata
        meta_train_mol_ids = df_train.loc[self.df_meta_train.index,"id"].unique()
        scored_length = df_train.loc[self.df_meta_train.index,"seq_scored"].values[0]
        print(f"meta_train_mol_ids : {len(meta_train_mol_ids)}")
        print(f"scored_length:{scored_length}")
        scored_mol_pos_ids = [f"{mol_id}_{i}" for mol_id in meta_train_mol_ids for i in range(scored_length)]
        print(f"scored_mol_pos_ids : {len(scored_mol_pos_ids)}")

        self.df_meta_train = self.df_meta_train.loc[scored_mol_pos_ids, :]
        print(self.df_meta_train)

        #print(idx1-idx2)
        #print(idx2-idx1)
        #sys.exit()

        #for mRNA


        #self.df_meta_train = self.df_meta_train.loc[self.df_meta_train.index.isin(df_train.index),:]




    def setMetaFromFiles(self, path_to_meta_feature_dir):
        pp_dir = pathlib.Path(path_to_meta_feature_dir)

        y_pred_list=[]
        oof_list = []
        for f in pp_dir.glob('*--_oof.csv'):
            oof_f_name = f.name
            print(oof_f_name)

            df_oof = pd.read_csv(str(f.parent/oof_f_name), index_col=0)[self.target_col_list]
            df_oof.sort_values('id_seqpos',inplace=True, ascending=True)
            print(f"df_oof : {df_oof.shape}")
            #oof_list.append(df_oof[self.target_col].values.reshape(-1, 1))
            oof_list.append(df_oof)




            pred_f_name = oof_f_name.replace("oof", "submission")
            print(pred_f_name)

            df_pred = pd.read_csv(str(f.parent/pred_f_name), index_col=0)[self.target_col_list]
            df_pred.sort_values('id_seqpos',inplace=True, ascending=True)
            print(f"df_pred : {df_pred.shape}")
            #y_pred_list.append(df_pred[self.target_col].values.reshape(-1, 1))
            y_pred_list.append(df_pred)

        df_oof = pd.concat(oof_list, axis=1)
        df_oof.columns=[i for i in range(0, len(df_oof.columns))]

        df_pred = pd.concat(y_pred_list, axis=1)
        df_pred.columns=[i for i in range(0, len(df_pred.columns))]



        return df_oof, df_pred

def interpolationSpline(df_y_pred, df_oof, true_y):
    
    dat = list(zip(df_oof.values.flatten(), true_y))
    dat = sorted(dat, key = lambda x: x[0])
    
    
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    
    from scipy.interpolate import UnivariateSpline
    spline_model = UnivariateSpline(list(datdict.keys()), list(datdict.values()))
    
 
    spline_fit = np.clip(spline_model(np.clip(df_oof.values.flatten().astype(float), -30, 30)), 0.025, 0.975)
    
    
    print_text = 'spline fit logloss is {:.5f}'.format(log_loss(true_y, spline_fit))
    slack.notify(text=print_text)
    print(print_text) 
    
    y_pred_fit = np.clip(spline_model(np.clip(df_y_pred.values.flatten().astype(float), -30, 30)), 0.025, 0.975)
    
    return y_pred_fit, spline_fit

def calcWeight(df_train, df_test):

    gp = df_train.groupby("target")["art_series_id"].agg({"count", "nunique"})
    #pdb.set_trace()
    gp["weight_nunique"] = gp["count"].sum()/gp["count"]

    df_train["loss_weight"] = df_train["target"].map(gp["weight_nunique"])
    df_test["loss_weight"] = 1.0



    return df_train, df_test


gl_norm_dict = {}

def preproc(df_train, df_test, target_col_list, setting_params):

    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        pass
    elif (setting_params["mode"]=="lgb"):

        pass

    else:

        df_train = drop_art_series(df_train)

        target_col = target_col_list[0]

        if target_col == "target":

            df_train[target_col] = df_train[target_col].astype(float) / 3.0
        elif target_col == "year_bin50":
            df_train[target_col] = df_train[target_col].astype(float) / 11.0
        

        df_train, df_test = calcWeight(df_train, df_test)

    return df_train, df_test

def postproc(df_y_pred, df_oof, df_train, df_test, setting_params):
    


    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        pass
    elif (setting_params["mode"]=="lgb"):
        pass
    else:

        target_col = setting_params["target"][0]
        if target_col == "target":
            df_y_pred["target"] = df_y_pred["target"] * 3
            df_oof["target"] = df_oof["target"]* 3
            
            def f(x):

                if x < 0:
                    return 0
                elif x > 3:
                    return 3
                
                return x
            
            df_y_pred["target"] = df_y_pred["target"].map(lambda x: f(x))
            df_oof["target"] = df_oof["target"].map(lambda x: f(x))

        elif target_col == "year_bin50":
            df_y_pred[target_col] = df_y_pred[target_col] * 11.0
            df_oof[target_col] = df_oof[target_col]* 11.0

            def f(x):

                if x <= 3:
                    return 0
                elif (x > 3) and (x <= 5):
                    return 1
                elif (x > 5) and (x <= 7):
                    return 2
                else:
                    return 3
                
                
            
            df_y_pred[target_col] = df_y_pred[target_col].map(lambda x: f(x))
            df_oof[target_col] = df_oof[target_col].map(lambda x: f(x))

            #pdb.set_trace()

    return df_y_pred, df_oof


def trainMain(df_train, df_test, target_col_list, setting_params):


    ########################
    #  preprocess
    ##########################

    #df_train.replace([np.inf, -np.inf], -1, inplace=True)
    #df_test.replace([np.inf, -np.inf], -1, inplace=True)

    df_train, df_test = preproc(df_train, df_test, target_col_list, setting_params)
    #nan_cols = showNAN(df_train)
    #print(nan_cols)
    #sys.exit()

    permutation_feature_flag=setting_params["permutation_feature_flag"]#False

    #procEDA(df_train, df_test)

    ########################
    #  set fold
    ##########################




    n_fold = setting_params["fold"]
    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        folds = KFold(n_fold)

    elif setting_params["mode"]=="lgb":
        #folds = StratifiedKFoldWithGroupID(n_splits=n_fold, group_id_col="path")
        #folds = myStratifiedKFoldWithGroupID(n_splits=n_fold, shuffle=False, group_id_col="path", stratified_target_id="floor")
        folds = siteStratifiedPathGroupKFold(df_test=df_test, n_splits=n_fold, group_id_col="path", stratified_target_id="site_id")


    else:

        #folds = myStratifiedKFold(n_splits=n_fold, shuffle=False, random_state=2020, stratified_col="site_id")
        
        #folds = Year2KFold(date_column='point_of_deal', test_years=2, clipping=False, split_type=2, n_splits=n_fold)

        if target_col_list[0]=="year_bin50":
            folds = myStratifiedKFoldWithGroupID(n_splits=n_fold, shuffle=False, group_id_col="art_series_id", stratified_target_id="target")
        elif target_col_list[0] == "target":
            folds = StratifiedKFoldWithGroupID(n_splits=n_fold, group_id_col="art_series_id")

        #folds = KFold(n_fold)

       # folds = GroupKFold(n_fold)

        #folds = TournamentGroupKFold(n_splits=n_fold)
        #folds = TimeStampNewUserSplit(n_splits=n_fold, group_id_col="user_id")

       

        #folds=None


    ##################################
    #         set eval
    ##################################

    from sklearn.metrics import roc_auc_score


    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        
        
    
        
        def my_eval(y_pred, y_true):
            
            preds = []
            for i in y_pred:
                preds.append([1-i, i])

            return log_loss(y_true, preds)
        
        eval_metric_name = 'binary_logloss'
        eval_metric_func_dict= {eval_metric_name:my_eval}

    elif (setting_params["mode"]=="lgb"):

        if setting_params["type"]=="classification":
            #use defalut lgb eval metric
            eval_metric_name = 'multi_logloss'
            eval_metric_func_dict= {eval_metric_name:eval_metric_name}

            # def my_eval(y_true, y_pred):

            #     ans = np.argmax(y_pred.reshape(y_true.shape[0], -1), axis=1)
                
                
            #     ret =  (np.abs(ans-y_true)).mean()
            #     return 'comp_eval', ret, False
            # eval_metric_name = 'comp_eval'
            # eval_metric_func_dict= {eval_metric_name:my_eval}
           



        else:
            eval_metric_name =  'l1'
            eval_metric_func_dict= {eval_metric_name:eval_metric_name}
            

    elif (setting_params["mode"]=="nn"):
        
        # def comp_metric(xhat, yhat, fhat, x, y, f):

            
   
        #     diff_x = (xhat-x) * gl_norm_dict["x_norm"] 
        #     diff_y = (yhat-y) * gl_norm_dict["y_norm"] 

        #     # print(f"norms : {gl_norm_dict['x_norm']}, {gl_norm_dict['y_norm']}")
        #     # print(f"diff_x : {x}")
        #     # print(f"diff_y : {y}")

        #     intermediate = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2))# + 15 * np.abs(fhat-f)
        #     return intermediate.sum()/xhat.shape[0]
        
        from sklearn.metrics import mean_squared_error
        def my_eval(y_pred, y_true):

            if setting_params["target"][0] == "target":
            
                y_pred = y_pred*3.0
                y_true = y_true*3.0
                y_pred = np.where(y_pred < 0, 0, y_pred)
                y_pred = np.where(y_pred > 3, 3, y_pred)

            elif setting_params["target"][0] == "year_bin50":

               
                

                y_pred = y_pred*11.0
                y_true = y_true*11.0

                y_pred = np.where(y_pred <= 3, 0, y_pred)
                y_pred = np.where((y_pred > 3) & (y_pred <= 5), 1, y_pred)
                y_pred = np.where((y_pred > 5) & (y_pred <= 7), 2, y_pred)
                y_pred = np.where(y_pred > 7, 3, y_pred)

                y_true = np.where(y_true <= 3, 0, y_true)
                y_true = np.where((y_true > 3) & (y_true <= 5), 1, y_true)
                y_true = np.where((y_true > 5) & (y_true <= 7), 2, y_true)
                y_true = np.where(y_true > 7, 3, y_true)

            
            return np.sqrt(mean_squared_error(y_true, y_pred))

            
        eval_metric_name = 'rmse'
        eval_metric_func_dict= {eval_metric_name:my_eval}



    else:

        pass



    


    #eval_metric_name = 'root_mean_squared_error'
    #eval_metric_func_dict= {eval_metric_name:"RootMeanSquaredError"}#'RootMeanSquaredError'}



    #eval_metric_name = 'rmse'
    #eval_metric_func_dict= {eval_metric_name: "rmse"}

    #eval_metric_name = 'MCRMSE'
    #eval_metric_func_dict= {eval_metric_name: my_eval}




    ##################################
    #         set loss
    ##################################
    custom_loss_func = False


    # def func_loss_wrmsse(y_true, y_pred):
    #     #return torch_wrmsse(y_true, y_pred, weight_mat_csr=np_weight_mat_csr, weight1=np_weight1, weight2_div_12=weight2_div_12, num_items=num_items, num_days=num_days)
    #     return loss_wrmsse(y_true, y_pred, weight_mat_csr=np_weight_mat_csr, sw=sw, num_items=num_items, num_days=num_days)

    # #custom_loss_func = func_loss_wrmsse

    # _weight_type ="full_linear" #
    # def func_weight_loss(y_true, y_pred):
    #     #return custom_asymmetric_train(y_pred=y_pred, y_true=y_true, w_type=_weight_type, num_items=num_items)
    #     return my_squared_error(y_pred=y_pred, y_true=y_true, w_type=_weight_type, num_items=num_items)
    # #custom_loss_func = func_weight_loss


    ########################
    #  prepare columns lists
    ##########################



    if (setting_params["mode"]=="lgb") | (setting_params["mode"]=="lgb2"):

        drop_cols=[
            'timestamp', 
            'near_floor',
            
             'first_ts', 'last_ts', 

       
            ]

     


        use_columns=list(df_test.columns)



        for col in drop_cols:
            if col in use_columns:
                use_columns.remove(col)

        #groupK_id_list = ["user_id"]
        id_list = ['path', "site_id", ]#['team1ID', 'team2ID',]
        time_list = ['first_ts', 'last_ts', 'timestamp',  ]

        rssi_list = getColumnsFromParts(["wifi_rssi", "ib_rssi"], use_columns)
        #rssi_list = getColumnsFromParts(["ib_rssi"], use_columns)
        #for col in rssi_list:
            #if df_train[col].dtype=="object":
            #    print(f"{col} : {df_train[col].dtype}")
            #df_train[col] = df_train[col].astype(int)
            #df_test[col] = df_test[col].astype(int)

        #pdb.set_trace()
        embedding_features_list= id_list #+ #['brand', 'model', 'android_name', 'api_level', 'version_name', 'version_code'] + id_list
        continuous_features_list = rssi_list +time_list
                                # + [
                                #    #'wifi_ssid', 'wifi_bssid', 'wifi_rssi', 'wifi_frequency', 'wifi_lastseen_ts', 
                                #    # "mean_max_floor", "mean_max_floor_by_path", "count_max_floor", "count_max_floor_by_path",
                                    
                                    

                                #     ]+ time_list# + has_floor_list



        scale_features = continuous_features_list
        inter_list =  [

            ]
        #inter_list+=getColumnsFromParts(["concat"], use_columns)

        target_encode_features =  [

                                   ] + inter_list + embedding_features_list




        #use_columns_graph = node_feature_list + id_list + other_list + edge_adjacent_feature_list+ edge_connect_feature_list + edge_connect_list
        #use_columns_dnn = embedding_features_list+continuous_features_list + id_list + other_list
        use_columns_lgb = target_encode_features + continuous_features_list 
        # use_columns_logreg = sclae_featrues + oh_nom_list + bin_list + nan_pos_list + oh_time_list + sqr_nom_list+ ["nom_6"]

        # use_columns_list = [use_columns_dnn, use_columns_lgb, use_columns_logreg, use_columns_logreg]
        final_use_columns =  use_columns_lgb #use_columns


    elif setting_params["mode"]=="nn":

        use_columns=list(df_test.columns)

        id_list = ["image_name"]
        time_list = []

        sequence_list= []
        continuous_features_list = []
        scale_features = time_list
        target_encode_features=[]

        if setting_params["target"][0] == "target":
            final_use_columns = id_list + ["loss_weight"]
        elif setting_params["target"][0] == "year_bin50":
            final_use_columns = id_list + ["loss_weight"]

        # if setting_params["pred_only"]==False:
        #     dropcols = [col for col in df_train.columns if col not in final_use_columns]
        #     dropcols.remove(setting_params["target"][0])
        #     df_train.drop(columns=dropcols, inplace=True)

            
        #     #df_train.to_pickle(PROC_DIR/"df_proc_train_nn.pkl")


    elif (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        #use_columns=list(df_test.columns)
        final_use_columns=None

    #######################
    # set main transformer
    #######################

    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        mt_none = MainTransformer()

        final_mt=mt_none

    elif (setting_params["mode"]=="lgb") :
        
        mt_none = MainTransformer()

        final_mt=mt_none
    else:


        mt_none = MainTransformer()
        # mt_log = MainTransformer(log_list=["koujika_mean_by_point_of_deal_year",
        #                                 "koujika_mean_by_sichoson_point_of_deal_year",
        #                                 #"koujika_mean_by_area_sichoson_point_of_deal_year"
        #                                 ])

        # mt_log_stack = MainTransformer(log_list=[0, 1, 2, 3])

        # mt_minmax = MainTransformer(minMax_scaler_cols = scale_featrues)
        # mt_std_scaler = MainTransformer(standard_scaler_cols = scale_featrues)

        mt_DNN = MainTransformer(
            #log_list=scale_features,
            #minMax_scaler_cols=scale_features,
            #standard_scaler_cols = rssi_list+scale_features,
            #label_encoding_cols=embedding_features_list
            )

        # mt_list=[mt_DNN, mt_none, mt_std_scaler, mt_none]

        #mt_DT = MainTransformer(change_category_cols=embedding_features_list, log_list=[], standard_scaler_cols=sclae_featrues)
        #mt_cat = MainTransformer(label_encoding_cols=embedding_features_list)


        final_mt = mt_DNN




    #######################
    # set post_prosessor
    #######################



    #dec_dict = pickle_load(PROC_DIR / 'decode_dict.pkl')
    #post_decode = MainTransformer(label_decode_cols_dict=dec_dict)
    post_None = MainTransformer()

    final_post=post_None


    #######################
    # set feature transformer
    #######################

    if (setting_params["mode"]=="ave") | (setting_params["mode"]=="stack"):
        ft_none = FeatureTransformer()
        transformers0 = {'ft': ft_none}
        final_tf_dict = transformers0
    elif (setting_params["mode"]=="lgb") :
        ft = DropFeatureTransformer(drop_columns=["path"])
        transformers1 = {'ft': ft}
        final_tf_dict = transformers1

    else:

        ft_none = FeatureTransformer()
        transformers0 = {'ft': ft_none}

        #target_encoding_cols = target_encode_features #getColumnsFromParts(["nom_", "inter_"], nom_list)

        #tet = TargetEncodingTransormer(target_encoding_cols)
        #transformers3 = {'targetEncoding': tet}

        ft = DropFeatureTransformer(drop_columns=["t1_wifi"])
        transformers1 = {'ft': ft}
        #transformers4 = {'ft': ft, 'targetEncoding': tet}


        # = TargetEncodingTransormer(["nom_6"])
        #bin_ft = binnigTransformer({"nom_6":3})
        #transformers2 = {'targetEncoding': tet_nom_6, "bining":bin_ft}

        #tf_dict_list=[transformers0, transformers1, transformers0, transformers0]

        final_tf_dict = transformers0

    ##################################
    #         select model
    ##################################


    #model_wrapper = LGBWrapper_regr()
    #model_wrapper = XGBWrapper_regr()

    #if callable(custom_loss_func):
    #    model_wrapper.initial_params['objective'] = custom_loss_func

    # model_wrapper = Cat_regr()
    # model_wrapper.initial_params['cat_features'] = embedding_features_list

    # model_wrapper = DeepTable_Wrapper(continuous_features_list=continuous_features_list,
    #                                     embedding_features_list=embedding_features_list,
    #                                     emb_dropout_rate=0.15, regression_flag=True,
    #                                     metric_func=[eval_metric_func_dict[eval_metric_name]]
    #                                     )


    #model_wrapper= LogisticRegression_Wrapper()
    #model_wrapper= RandomForestClassifier_Wrapper()


    #model_wrapper=LGBWrapper_regr()
    #model_wrapper = ElasticNetRegression_Wrapper()
    #model_wrapper = Lasso_Wrapper()
    #model_wrapper = Ridge_Wrapper()
    #model_wrapper = SVR()
   # model_wrapper=DNN_Wrapper(init_x_num=len(use_columns))


    # df_all = pd.concat([df_train, df_test], sort=False)
    # model_embeddingdnn_wrapper=EmbeddingDNN_Wrapper(df_all=df_all, continuous_features_list=continuous_features_list, embedding_features_list=embedding_features_list, emb_dropout_rate=0.15)
    # model_wrapper = model_embeddingdnn_wrapper

    if (setting_params["mode"]=="lgb") & (setting_params["type"]=="classification"):
        
        
        model_wrapper = LGBWrapper_cls()
        #model_wrapper = XGBWrapper_cls()
        
    if (setting_params["mode"]=="lgb") & (setting_params["type"]=="regression"):
        
        
        model_wrapper = LGBWrapper_regr()
       

    if (setting_params["mode"]=="nn"):

        # df_all = pd.concat([df_train, df_test], sort=False)
        # model_lstm_wrapper=LSTM_Wrapper(df_all=df_all, sequence_features_list=sequence_list, continuous_features_list=continuous_features_list, embedding_category_features_list=embedding_category_list, num_target=len(target_col_list),
        #                                 sequence_index_col="id", input_sequence_len_col="seq_length", output_sequence_len_col="seq_scored", weight_col="weight",emb_dropout_rate=0.5)
        model_wrapper = ResNet_Wrapper()

        #model_wrapper = Transformer_Wrapper(sequence_features_list=sequence_list, continuous_features_list=continuous_features_list,)
        #model_wrapper = LastQueryTransformer_Wrapper(sequence_features_list=sequence_list, continuous_features_list=continuous_features_list,)

        





    if (setting_params["mode"]=="graph"):
        df_all = pd.concat([df_train, df_test], sort=False)
        model_graph_wrapper=Graph_Wrapper(df_all=df_all, node_feature_list=node_feature_list, node_index_col="structure_id",  edge_connect_list=edge_connect_list,
                                        edge_adjacent_feature_list=edge_adjacent_feature_list, edge_connect_feature_list=edge_connect_feature_list, num_target=len(target_col_list),
                                        sequence_index_col="id", input_sequence_len_col="seq_length", output_sequence_len_col="seq_scored", weight_col="weight", emb_dropout_rate=0.5)
        model_wrapper = model_graph_wrapper



    #stacking_model_wrppers=[model_embeddingdnn_wrapper, LGBWrapper_cls(), LogisticRegression_Wrapper(),  LogisticRegression_Wrapper()]
    #model_wrapper = StackingWrapper(df_train, df_test, "target", _folds=folds, eval_metric_name=eval_metric_name, eval_metric_func_dict=eval_metric_func_dict,
    #                stacking_model_wrppers=stacking_model_wrppers, use_columns_list=use_columns_list,
    #                mt_list=mt_list, tf_dict_list=tf_dict_list)


    if (setting_params["mode"]=="stack"):
        meta_model = Ridge_Wrapper()
        model_wrapper = SimpleStackingWrapper(df_train, df_test, target_col_list,  meta_model, str(OUTPUT_DIR / setting_params["stacking_dir_name"]))
        df_train = model_wrapper.df_meta_train
        df_test = model_wrapper.df_meta_test


    if (setting_params["mode"]=="ave"):
        rate_list=None#[0.6, 0.2, 0.2]
        model_wrapper = Averaging_Wrapper(df_train, df_test, target_col_list, str(OUTPUT_DIR / setting_params["stacking_dir_name"]), rate_list=rate_list)
        df_train = model_wrapper.df_meta_train
        df_test = model_wrapper.df_meta_test


    df_y_pred, df_oof, valid_score, model_name = simplePredictionSet(df_train.copy(deep=True), df_test, target_col_list,
                                                                use_columns=final_use_columns, _folds=folds, setting_params=setting_params,
                                                                main_transformer=final_mt, transformer_dict=final_tf_dict, post_processor=final_post, model_wrapper=model_wrapper,
                                                                eval_metric_name=eval_metric_name, eval_metric_func_dict=eval_metric_func_dict, permutation_feature=permutation_feature_flag)


    #########################
    #post processing
    #########################

    df_y_pred, df_oof = postproc(df_y_pred, df_oof, df_train, df_test, setting_params)


    return df_y_pred, df_oof, valid_score, model_name





def saveSubmission(path_to_output_dir, df_train, df_test, target_col, df_y_pred, df_oof, valid_score, model_name, setting_params):

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "{}_{}--{:.6f}--".format(now, model_name,valid_score)


    df_save_oof = df_oof.reset_index()
    
    path_to_output_oof = path_to_output_dir/"oof.csv" if ON_KAGGLE else path_to_output_dir / "{}_oof.csv".format(prefix)
    df_save_oof.to_csv(path_to_output_oof, index=False)


    #only in atma11
    #df_final_pred = df_y_pred.reset_index()
    df_final_pred = df_y_pred

    path_to_output = path_to_output_dir/"submission.csv" if ON_KAGGLE else path_to_output_dir / "{}_submission.csv".format(prefix)
    df_final_pred.to_csv(path_to_output, index=False)

   

    logger.debug("df_save_oof:{}".format(df_save_oof.shape))
    logger.debug("df_submit:{}".format(df_final_pred.shape))



def createOutputDir(setting_params):
    label_str = "_".join([h[0] for h in setting_params["label_list"]])
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{now}_{label_str}_{setting_params['comment']}"

    dir_name = dir_name+"_p" if setting_params["pseudo"] else dir_name

    ppath_to_out = OUTPUT_DIR / dir_name

    os.mkdir(ppath_to_out)

    return ppath_to_out

def createStackDir(stack_dir_list, data_label, ppath_to_out):

    new_label_dir = ppath_to_out / f"{data_label}"
    os.mkdir(new_label_dir)

    for dir_name in stack_dir_list:
        ppath = OUTPUT_DIR / dir_name
        print(ppath)


        for f in ppath.glob(f'*_{data_label}*.csv'):
            print(f)
            shutil.copy2(f, new_label_dir / f.name)

    return new_label_dir

def procSkipTarget(target_col, df_oof, df_y_pred, df_train_targets):

    df_oof[target_col] = 0
    df_y_pred[target_col] = 0

    df_oof_each = pd.DataFrame(df_oof[target_col])
    df_y_pred_each = pd.DataFrame(df_y_pred[target_col])

    valid_score_each = log_loss(df_train_targets[target_col], df_oof[target_col])

    return df_y_pred_each, df_oof_each, valid_score_each, "skip"


def main(setting_params):

    mode=setting_params["mode"]
    setting_params["index"] = "object_id"
    
    target_cols= ["target"] #year_bin50
   
    setting_params["num_class"] = len(target_cols)



    if setting_params["pred_only"]==False:
        
        if mode == "ave":
            mode = "nn"

        df_train = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
        df_train.index.name = setting_params["index"]



        logger.debug("df_train:{}".format(df_train.shape))
    else:
        df_train = None



    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test.pkl')

   
    
    df_submit = pd.read_csv(INPUT_DIR/f"atmaCup#11_sample_submission.csv")
    df_submit.index = df_test.index



    logger.debug("df_test:{}".format(df_test.shape))
    logger.debug("df_submit:{}".format(df_submit.shape))

    cpu_stats("after df initial data load")


    df_y_pred = df_submit



    

    setting_params["target_idx"] = 0
    setting_params["target"] = target_cols

    if (mode  == "nn")| (mode  == "lgb")| (mode == "ave") | (mode == "stack"):

        df_y_pred_each, df_oof, valid_score, model_name = trainMain(df_train, df_test, target_cols, setting_params)
        

        if setting_params["debug"]:

            df_y_pred.loc[df_y_pred_each.index, target_cols] = df_y_pred_each[target_cols]
        else:
            df_y_pred = df_y_pred_each#[target_cols]

        if target_cols[0]=="year_bin50":
            df_oof = df_oof.rename(columns={target_cols[0]:"target"})
            df_y_pred = df_y_pred.rename(columns={target_cols[0]:"target"})

        saveSubmission(OUTPUT_DIR, df_train, df_test, target_cols, df_y_pred, df_oof, valid_score, model_name, setting_params)


def argParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="nn", choices=['lgb','nn','graph', 'ave', 'stack'] )
    parser.add_argument('-t', '--type', default="regression", choices=['classification','regression'] )
    parser.add_argument('-stack_dir', '--stacking_dir_name', type=str, )
    parser.add_argument('-ep', '--epochs', type=int, default=4)
    parser.add_argument('-es', '--early_stopping_rounds', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
    parser.add_argument('-batch', '--batch_size', type=int, default=128)
    parser.add_argument('-num_workers', '--num_workers', type=int, default=2)
    parser.add_argument('-v', '--verbose', type=int, default=1)
    parser.add_argument('-f', '--fold', type=int, default=5)
    parser.add_argument('-f2', '--fold2', type=int, default=3)
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-d2', '--debug2', action="store_true")
    parser.add_argument('-u', '--use_old_file', action="store_true")
    parser.add_argument('-train_plot', '--train_plot', action="store_true")

    parser.add_argument('-no_wb', '--no_wandb', action="store_true")

    parser.add_argument('-model_dir', '--model_dir_name', type=str)
    parser.add_argument('-pred', '--pred_only', action="store_true")
    parser.add_argument('-pred2', '--pred2_only', action="store_true")
    parser.add_argument('-mid_save', '--mid_save', action="store_true")
    parser.add_argument('-permu', '--permutation_feature_flag', action="store_true")


    args=parser.parse_args()

    setting_params= vars(args)

    if setting_params["model_dir_name"] is None:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        setting_params["model_dir_name"] = f"{now}"



    return setting_params

if __name__ == '__main__':



    setting_params=argParams()
    
    main(setting_params = setting_params)


