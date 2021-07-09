# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:14:29 2020

@author: p000526841
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
import inspect

from utils import *
from log_settings import MyLogger

my_logger = MyLogger()
logger = my_logger.generateLogger("EDA", LOG_DIR+"/eda.log")

plt.rcParams['font.family'] = 'IPAexGothic'

def showCorr(df, str_value_name, show_percentage=0.6):
        corrmat = df.corr()
    
        num_of_col = len(corrmat.columns)
        cols = corrmat.nlargest(num_of_col, str_value_name)[str_value_name]
        tmp = cols[(cols >= show_percentage) | (cols <= -show_percentage)]

        logger.debug("*****[ corr : " + str_value_name + " ]*****")
        logger.debug(tmp)
        logger.debug("*****[" + str_value_name + "]*****")
        logger.debug("\n")
        
        #print(tmp[0])

def showBoxPlot(df, str_val1, str_va2):
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=90, size='small')
    
    #neigh_median = df.groupby([str_val1],as_index=False)[str_va2].median().sort_values(str_va2)
    #print(neigh_median)
    #col_order = neigh_median[str_val1].values
    #sns.boxplot(x=df[str_val1], y =df[str_va2], order=col_order)

    sns.boxplot(x=df[str_val1], y =df[str_va2])
    plt.tight_layout()
    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_box_plot_{}.png".format(str_val1))
    #print("save: ", path_to_save)
    plt.savefig(path_to_save)
    plt.show(block=False) 
    plt.close()


def showValueCount(df_train, df_test, str_value, str_target, debug=True, regression_flag=1, _fig_size=(20,10)):

        if str_value == str_target:
            df_test[str_value] = np.nan
        
        df = pd.concat([df_train, df_test])
        if not str_value in df.columns:
            logger.debug(str_value, " is not inside columns")
            return 
        
        
        se_all = df[str_value]
        se_train = df_train[str_value]
        se_test = df_test[str_value]
        
        all_desc = se_all.describe()
        train_desc = se_train.describe()
        test_desc = se_test.describe()
        df_concat_desc = pd.concat([train_desc, test_desc, all_desc], axis=1, keys=['train', 'test', "all"])
        
        if debug:
            logger.debug("***[" + str_value + "]***")
            logger.debug("describe :")
            logger.debug(df_concat_desc)
        
        num_nan_all = se_all.isna().sum()
        num_nan_train = se_train.isna().sum()
        num_nan_test = se_test.isna().sum()
        df_concat_num_nan = pd.DataFrame([num_nan_train, num_nan_test, num_nan_all], columns=["num_of_nan"], index=['train', 'test', "all"]).transpose()
        
        if debug:
            logger.debug("Num of Nan : ")
            logger.debug(df_concat_num_nan)
        
        df_value = se_all.value_counts(dropna=False)
        df_value_percentage = (df_value / df_value.sum()) * 100
        
        
        df_value_train = se_train.value_counts(dropna=False)
        df_value_train_percentage = (df_value_train / df_value_train.sum()) * 100
        
        df_value_test = se_test.value_counts(dropna=False)
        df_value_test_percentage = (df_value_test / df_value_test.sum()) * 100
        
        df_concat = pd.concat([df_value_train, df_value_train_percentage, df_value_test, df_value_test_percentage, df_value, df_value_percentage], axis=1, keys=['train', "train rate", 'test', "test rate", "all", "all rate"], sort=True)
        
        train_values = set(se_train.unique())
        test_values = set(se_test.unique())

        xor_values = test_values - train_values 
        if xor_values:
            #print(f'Replace {len(xor_values)} in {col} column')
            logger.debug(f'{xor_values} is only found in test, not train!!!')
            
            #full_data.loc[full_data[col].isin(xor_values), col] = 'xor'
            
        xor_values_train = train_values - test_values
        if xor_values_train:
            #print(f'Replace {len(xor_values)} in {col} column')
            logger.debug(f'{xor_values_train} is only found in train, not test!!!' )
            
            #full_data.loc[full_data[col].isin(xor_values), col] = 'xor'
        
        
        if debug:
            logger.debug("value_counts :")
            logger.debug(df_concat)
            
            plt.figure(figsize=_fig_size)
            df_concat[['train', 'test', "all"]].plot.bar(figsize=_fig_size)

            
            plt.ylabel('Number of each element', fontsize=12)
            plt.xlabel(str_value, fontsize=12)
            plt.xticks(rotation=90, size='small')
            plt.tight_layout()
            path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_num_each_elments_{}.png".format(str_value))
            #print("save: ", path_to_save)
            plt.savefig(path_to_save)
            plt.show(block=False) 
            plt.close()
        
            plt.figure(figsize=_fig_size)
            df_concat[['train rate', 'test rate', "all rate"]].plot.bar(figsize=_fig_size)
            plt.ylabel('rate of each element', fontsize=12)
            plt.xlabel(str_value, fontsize=12)
            plt.xticks(rotation=90, size='small')
            plt.tight_layout()
            path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_rate_each_elments_{}.png".format(str_value))
            #print("save: ", path_to_save)
            plt.savefig(path_to_save)
            plt.show(block=False) 
            plt.close()

        
        
        
        if str_value != str_target and str_target in df.columns:
            
            if regression_flag == 1:
                if debug:
                    showBoxPlot(df_train, str_value, str_target)
            
            else:
                
                df_train_small = df.loc[df[str_target].isnull() == False, [str_value, str_target]]
                df_stack = df_train_small.groupby(str_value)[str_target].value_counts().unstack()
                
                if debug:
                    logger.debug("---")
                
                col_list = []
                df_list = []
                
                if debug:
                    plt.figure(figsize=_fig_size)
                    g = sns.countplot(x=str_value, hue = str_target, data=df, order=df_stack.index)
                    plt.xticks(rotation=90, size='small')
                    ax1 = g.axes
                    ax2 = ax1.twinx()
                    
                for col in df_stack.columns:
                    col_list += [str(col), str(col)+"_percent"]
                    df_percent = (df_stack.loc[:, col] / df_stack.sum(axis=1))
                    
                    df_list += [df_stack.loc[:, col], df_percent]
                    
                    if debug:
                        #print(df_percent.index)
                        xn = range(len(df_percent.index))
                        sns.lineplot(x=xn, y=df_percent.values, ax=ax2)
                        #sns.lineplot(data=df_percent, ax=ax2)
                        #sns.lineplot(data=df_percent, y=(str(col)+"_percent"), x=df_percent.index)
                    
                df_conc = pd.concat(df_list, axis=1, keys=col_list)
                
                if debug:
                    logger.debug(df_conc.T)
                    #print(df_stack.columns)
                    #print(df_stack.index)

                    #plt.tight_layout()
                    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_count_line_{}.png".format(str_value))
                    #print("save: ", path_to_save)
                    plt.savefig(path_to_save)
                    plt.show(block=False) 
                    plt.close()
        
                
        if debug:
            logger.debug("******\n")
        
        del df
        gc.collect()
        
        return df_concat

def showJointPlot(_df_train, _df_test, str_value, str_target, debug=True, regression_flag=1, corr_flag=False, empty_nums=[], log_flag=1, _fig_size=(20, 10)):
        print("now in function : ", inspect.getframeinfo(inspect.currentframe())[2])
        
        df_train = _df_train.copy()
        df_test = _df_test.copy()
        
        if str_value == str_target:
            df_test[str_target] = np.nan
        
        if len(empty_nums) >0:
            for e in empty_nums:
                df_train[str_value] = df_train[str_value].replace(e, np.nan)
                df_test[str_value] = df_test[str_value].replace(e, np.nan)
        
        if log_flag==1:
            df_train[str_value] = np.log1p(df_train[str_value])
            df_test[str_value] = np.log1p(df_test[str_value])
        
        df = pd.concat([df_train, df_test])

        if not str_value in df.columns:
            logger.debug(str_value + " is not inside columns")
            return 
        

        se_all = df[str_value]
        se_train = df_train[str_value]
        se_test = df_test[str_value]
        
        all_desc = se_all.describe()
        train_desc = se_train.describe()
        test_desc = se_test.describe()
        df_concat_desc = pd.concat([train_desc, test_desc, all_desc], axis=1, keys=['train', 'test', "all"])
        
        logger.debug("***[" + str_value + "]***")
        logger.debug("describe :")
        logger.debug(df_concat_desc)
        
        num_nan_all = se_all.isna().sum()
        num_nan_train = se_train.isna().sum()
        num_nan_test = se_test.isna().sum()
        df_concat_num_nan = pd.DataFrame([num_nan_train, num_nan_test, num_nan_all], columns=["num_of_nan"], index=['train', 'test', "all"]).transpose()
        
        logger.debug("Num of Nan : ")
        logger.debug(df_concat_num_nan)
        
        skew_all = se_all.skew()
        skew_train = se_train.skew()
        skew_test = se_test.skew()
        df_concat_skew = pd.DataFrame([skew_train, skew_test, skew_all], columns=["skew"], index=['train', 'test', "all"]).transpose()
        
        
        logger.debug("skew : ")
        logger.debug(df_concat_skew)
        
        if corr_flag==True:
            showCorr(df, str_value)
        
        
               
        #tmp_se = pd.Series( ["_"] * len(df_dist), columns=["dataset"] )
        #print(tmp_se.values)
        #df_dist.append(tmp_se)
        #df_dist["dataset"].apply(lambda x: "train" if pd.isna(x[self.str_target_value_]) == False else "test")
        #df_dist.plot(kind="kde", y=df_dist["dataset"])
        
        plt.figure(figsize=_fig_size)
        sns.distplot(df_train[str_value].dropna(),kde=True,label="train")
        sns.distplot(df_test[str_value].dropna(),kde=True,label="test")

        plt.title('distplot by {}'.format(str_value),size=20)
        plt.xlabel(str_value)
        plt.ylabel('prob')
        plt.legend() #実行させないと凡例が出ない。
        plt.tight_layout()
        path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_distplot_{}.png".format(str_value))
        #print("save: ", path_to_save)
        plt.savefig(path_to_save)
        plt.show(block=False) 
        plt.close()
        #sns.distplot(df_dist[str_value], hue=df_dist["dataset"])
        
        #visualize_distribution(df[str_value].dropna())
        #visualize_probplot(df[str_value].dropna())
        
#        plt.figure(figsize=(10,5))
#        
#        sns.distplot()
#        plt.show()
 
        if (str_value != str_target) and (str_target in df.columns):
            #plt.figure(figsize=(10,5))
            if regression_flag == 1:
                sns.jointplot(str_value, str_target, df_train)
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_jointplot_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
                df_train.plot.hexbin(x=str_value, y=str_target, gridsize=15, sharex=False)
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_hexbin_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()

                
            #plt.show()
            else:
                
                df_small = df_train[[str_value, str_target]]
                logger.debug(df_small.groupby(str_target)[str_value].describe().T)
                
                type_val = df_small[str_target].unique()
                #print(type_val)
                plt.figure()
                for i, val in enumerate(type_val):
                    sns.distplot(df_small.loc[df_small[str_target]==val, str_value].dropna(),kde=True,label=str(val)) #, color=mycols[i%len(mycols)])
                plt.legend() #実行させないと凡例が出ない。
                plt.tight_layout()
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_distplot_target_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
                
                plt.figure(figsize=_fig_size)
                plt.xlabel(str_value, fontsize=9)
                for i, val in enumerate(type_val):
                    sns.kdeplot(df_small.loc[df_small[str_target] == val, str_value].dropna().values, bw=0.5,label='Target: {}'.format(val))
                    
                sns.kdeplot(df_test[str_value].dropna().values, bw=0.5,label='Test')
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_kde_target_{}.png".format(str_value))
                print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
        
        logger.debug("******\n")
        
        del df, df_train, df_test
        gc.collect()
        
        return
    
    
def showDetails(df_train, df_test, new_cols, target_val, corr_flag=True):
    for new_col in new_cols:
        try:
            if df_train[new_col].dtype == "object":
                showValueCount(df_train, df_test, new_col, target_val, debug=True, regression_flag=0)
            else:
                showJointPlot(df_train, df_test, new_col, target_val, debug=True, regression_flag=0, corr_flag=corr_flag, empty_nums=[-999], log_flag=0)
        except Exception as e:
            logger.debug(e)
            logger.debug("******[error col : {}]******".format(new_col))
        
def interEDA(df_train, df_test, inter_col, new_cols, target_val, _fig_size=(10, 5)):

    df = pd.concat([df_train, df_test])
    elements = df[inter_col].unique()
    type_val = df[target_val].unique()

    for col in new_cols:
        plt.figure(figsize=_fig_size)
        plt.title('interaction kde of {}'.format(inter_col),size=20)
        plt.xlabel(col, fontsize=9)
        for e in elements:

            df_small = df_train.loc[df_train[inter_col] == e]
            for i, val in enumerate(type_val):
                sns.kdeplot(df_small.loc[df_small[target_val] == val, col].dropna().values, bw=0.5,label='Inter:{}, Target: {}'.format(e, val))
                
            sns.kdeplot(df_test.loc[df_test[inter_col]==e, col].dropna().values, bw=0.5,label='Inter:{}, Test'.format(e))
        path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_inter_kde_{}_vs_{}.png".format(inter_col, col))
        print("save: ", path_to_save)
        plt.savefig(path_to_save)
        plt.show(block=False) 
        plt.close()




def procEDA_(df_train, df_test):
    
    #df_train = df_train[df_train["y"] < (90)]
    
    new_col=["延床面積（㎡）"]#
    #new_col=df_train.columns
    showDetails(df_train, df_test, new_col, "y", corr_flag=False)
    
    sys.exit()
    #df["nom"]
    #print(df_train.loc[df_train["auto__nom_8_count_agg_by_nom_7"] > 30000, "auto__nom_8_count_agg_by_nom_7"])
    
    #showValueCount(df_train, df_test, "ord_5", "target", debug=True, regression_flag=0)
    for i in range(6):
        

        col_name = "ord_{}".format(i)
        df_train[col_name] /= df_train[col_name].max()  # for convergence
        df_test[col_name] /= df_test[col_name].max()
        
        
        new_name = "{}_sqr".format(col_name)
        df_train[new_name] = 4*(df_train[col_name] - 0.5)**2
        df_test[new_name] = 4*(df_test[col_name] - 0.5)**2
    #
    new_col=["ord_3", "ord_3_sqr"]#getColumnsFromParts(["ord_3"], df_train.columns)
    #showDetails(df_train, df_test, new_col, "target", corr_flag=False)
    for col in new_col:
        showJointPlot(df_train, df_test, col, "target", debug=True, regression_flag=0, corr_flag=False, empty_nums=[-999], log_flag=1)
    
        
    # new_cols=list(df_train.columns.values)
    # new_cols.remove("bin_3")
    # new_cols.remove("target")
    # #new_cols=["null_all_count"]
    # #new_cols = getColumnsFromParts(["bin_3"], df_train.columns)
    # #showDetails(df_train, df_test, new_cols, "target")
    # for col in new_cols:
    #     try:
    #         interEDA(df_train, df_test, col, ["bin_3"], "target")
    #     except Exception as e:
    #         logger.debug(e)
    #         logger.debug("******[inter error col : {}]******".format(col))

    sys.exit(0)
    
    
    # colums_parts=[]
    
    # parts_cols = getColumnsFromParts(colums_parts, df_train.columns)
    
    # new_cols = list(set(new_cols + parts_cols))

    use_columns=list(df_test.columns)
    bin_list = ["bin_{}".format(i) for i in range(5)]
    
    ord_list = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "ord_5_1", "ord_5_2"]

    nom_list = ["nom_{}".format(i) for i in range(10)] #getColumnsFromParts(["nom_"] , use_columns)
    #oh_nom_list = getColumnsFromParts(["OH_nom"] , use_columns)
    time_list = ["day", "month"]

    nan_pos_list = getColumnsFromParts(["_nan_pos"] , use_columns)
    count_list = getColumnsFromParts(["_count"], use_columns)
    #inter_list = getColumnsFromParts(["inter_"], use_columns)
    additional_list = ["day_cos", "day_sin", "month_cos", "month_sin"]


    embedding_features_list=time_list + ord_list + nom_list + bin_list
    continuous_features_list =  additional_list+count_list +nan_pos_list
    final_cols = embedding_features_list+continuous_features_list

    #adversarialValidation(df_train[final_cols], df_test[final_cols], drop_cols=[])
    adv2(df_train[final_cols], df_test[final_cols], drop_cols=[])
    
    return

def loadRaw():

    df_train = pd.read_pickle(INPUT_DIR / 'train.pkl')
    df_test = pd.read_pickle(INPUT_DIR / 'test.pkl')

    return df_train, df_test

def loadProc(stage=1, decode_flag=False):

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_lgb_stage{stage}.pkl')
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_lgb_stage{stage}.pkl')
    
    if decode_flag:
        dec_dict = pickle_load(PROC_DIR / 'decode_dict.pkl')
        for col in df_test.columns:
            if col in dec_dict.keys():
                df_train[col].replace(dec_dict[col], inplace=True)
                df_test[col].replace(dec_dict[col], inplace=True)
            


    return df_train, df_test

def procSave():
    
    df_train, df_test = loadRaw()
    
    logger.debug("df_train_interm:{}".format(df_train.shape))
    logger.debug("df_test_interm:{}".format(df_test.shape))
    

    df_train["part"] = "train"
    df_test["part"] = "test"
    
    df = pd.concat([df_train, df_test])
    
    
    syurui_list = list(df["種類"].unique())

    
    for w in syurui_list:
        
        df_csv = df.loc[(df["種類"]==w)]
        df_csv.to_csv(PROC_DIR /"syurui_{}.csv".format(w), encoding='utf_8_sig')
        
        
def edaSeasonMatch(df_train, df_test):
    
    print(df_train.groupby("Season")["total_match_team1"].mean())    
    print(df_test.groupby("Season")["total_match_team1"].mean())    
    
    sys.exit()
    
    
def compare_sample(df_train, df_test):
    
    df_train_sample = pd.read_csv(PROC_DIR/"df_train_sample.csv", index_col=0)
    df_train_sample["org_ID"] = [f"{s}_{w}_{l}" for s, w, l in zip(df_train_sample["Season"], df_train_sample["TeamIdA"], df_train_sample["TeamIdB"])]
    df_train_sample.set_index("org_ID", inplace=True)
    df_train_sample = df_train_sample.loc[df_train_sample["Season"]<2015]
    
    
    df_train =df_train.loc[df_train["Season"]>=2003]
    # df_swap = df_train[["team1ID", "team2ID", "seed_num_diff", "Season"]]
    # df_swap["team2ID"] = df_train["team1ID"]
    # df_swap["team1ID"] = df_train["team2ID"]
    # df_swap["seed_num_diff"] = -1 * df_train["seed_num_diff"]
    
    
    # df_train = pd.concat([df_train[["team1ID", "team2ID", "seed_num_diff", "Season"]],df_swap])
    df_train["org_ID"] = [f"{s}_{w}_{l}" for s, w, l in zip(df_train["Season"], df_train["team1ID"], df_train["team2ID"])]
    df_train.set_index("org_ID", inplace=True)
    
    df_train_sample["diff"] = df_train_sample["SeedDiff"]- df_train["seed_num_diff"]
    print(df_train_sample["diff"].value_counts())
    
    print(df_train_sample["SeedDiff"].equals(df_train["seed_num_diff"]))
    print(df_train.shape)
    print(df_train_sample.shape)
    
    sys.exit()
    
def procEDA2(df_train, df_test):

    
    #compare_sample(df_train, df_test)
    #print(df_train.loc[(df_train["team1ID"]==1411)&(df_train["team2ID"]==1421)])
    #sys.exit()
    
    target_col="Pred"
    
    # other_list = ['FGM','FGA','FGM3','FGA3','FTM','FTA',
    #                                      'OR','DR','Ast','TO','Stl','Blk','PF',
    #                                       'goal_rate', '3p_goal_rate',
    #                                      'ft_goal_rate']
    # new_col= getColumnsFromParts(other_list, df_train.columns)
    
    new_col = getColumnsFromParts(["pair", ], df_train.columns)
    
    object_flag = 0
    
    if object_flag:
        for col in new_col:
            for tmp_df in [df_train, df_test]:
                tmp_df[col] = tmp_df[col].astype("object")

    
    #new_col=df_train.columns
    showDetails(df_train, df_test, new_col, target_col, corr_flag=False)  
    
def tmp1(df_train, df_test):
    
    area = ["千代田区", "新宿区", "江戸川区", "神津島村", "町田市"]
    tmp = df_train.loc[df_train["市区町村名"].isin(area), :].groupby("市区町村名")["地区名"].value_counts(dropna=False)
    print(tmp)
    
    tmp = df_train.loc[df_train["地区名"].isnull(), ["市区町村名", "種類", "地域"]]
    print(tmp)
    
    
    
def procAdv(df_train, df_test):
    
    drop_cols=['ranksystem_max_BNZ_team1',
    'ranksystem_max_RTH_team1',
    'ranksystem_max_7OT_team1',
    'ranksystem_max_ARG_team1',
    'ranksystem_max_BIH_team1',
    'ranksystem_max_7OT_team2',
    'ranksystem_max_ADE_team1',
    'ranksystem_max_ACU_team1',
    'ranksystem_max_BWE_team1',
    'ranksystem_max_COL_team1',
    'ranksystem_max_FAS_team1',
    'ranksystem_std_TPR_team1',
    'ranksystem_max_BBT_team1',
    'ranksystem_max_BIH_team2',
    'ranksystem_max_DAV_team1',
    'ranksystem_max_DII_team1',
    'ranksystem_max_DOL_team1',
    'ranksystem_std_KPK_team1',
    'ranksystem_max_RWP_team1',
    'ranksystem_max_KRA_team1',
    'ranksystem_max_KPK_team1',
    'ranksystem_max_DC_team1',
    'ranksystem_max_BOB_team1',
    'ranksystem_max_DUN_team2',
    'ranksystem_max_ARG_team2',
    'ranksystem_max_SMS_team2',
    'ranksystem_max_SEL_team1',
    'ranksystem_min_BIH_team1',
    'ranksystem_max_CPA_team1',
    'ranksystem_std_JJK_team2',
    'ranksystem_mean_MOR_team1',
    'ranksystem_max_DOL_team2',
    'ranksystem_max_DCI_team1',
    'ranksystem_max_AP_team1',
    'ranksystem_std_WOL_team1',
    'ranksystem_min_LEF_team1',
    'ranksystem_min_INC_team1',
    'ranksystem_min_BNZ_team1',
    'ranksystem_min_7OT_team1',
    'ranksystem_max_WLK_team1',
    'ranksystem_max_SMS_team1',
    'ranksystem_max_PIG_team1',
    'ranksystem_max_NET_team1',
    'ranksystem_max_DAV_team2',
    'ranksystem_mean_BNZ_team2',
    'ranksystem_max_BUR_team1',
    'ranksystem_max_CWL_team2',
    'ranksystem_max_JNG_team2',
    'ranksystem_max_BWE_team2',
    'ranksystem_std_WLK_team1',
    'ranksystem_max_EMK_team1',
    'ranksystem_max_AWS_team1',
    'ranksystem_std_JNG_team2',
    'ranksystem_std_DOL_team2',
    'ranksystem_std_DC_team2',
    'ranksystem_std_BWE_team2',
    'ranksystem_mean_RTP_team2',
    'ranksystem_max_SEL_team2',
    'ranksystem_max_SAG_team1',
    'ranksystem_max_JJK_team1',
    'ranksystem_max_FDM_team1',
    'ranksystem_max_ENT_team1',
    'ranksystem_std_AP_team1',
    'ranksystem_std_KCX_team1',
    'ranksystem_max_TPR_team2',
    'ranksystem_std_MAS_team2',
    'ranksystem_max_RWP_team2',
    'ranksystem_std_DAV_team1',
    'ranksystem_min_CBR_team2',
    'ranksystem_max_WIL_team2',
    'ranksystem_max_DUN_team1',
    'ranksystem_max_DES_team1',
    'ranksystem_mean_WLK_team2',
    'ranksystem_max_CBR_team2',
    'ranksystem_mean_BOB_team1',
    'ranksystem_max_FDM_team2',
    'ranksystem_max_LAW_team2',
    'ranksystem_std_LMC_team1',
    'ranksystem_std_COL_team2',
    'ranksystem_max_PGH_team1',
    'ranksystem_max_BNZ_team2',
    'ranksystem_max_MAS_team2',
    'ranksystem_std_RTH_team1',
    'ranksystem_min_PGH_team1',
    'ranksystem_mean_JNG_team2',
    'ranksystem_max_BBT_team2',
    'ranksystem_std_DOK_team1',
    'ranksystem_mean_DOL_team2',
    'ranksystem_max_RTP_team1',
    'ranksystem_max_NOL_team2',
    'ranksystem_max_CRO_team1',
    'ranksystem_max_CNG_team1',
    'ranksystem_std_USA_team2',
        'ranksystem_max_CBR_team1',
    'ranksystem_max_WOL_team1',
    'ranksystem_max_AUS_team1',
    'ranksystem_max_BCM_team1',
    'ranksystem_max_BLS_team1',
    'ranksystem_max_DOK_team1',
    'ranksystem_max_JNG_team1',
    'ranksystem_max_CWL_team1',
    'ranksystem_max_EBP_team1',
    'ranksystem_max_INC_team1',
    'ranksystem_max_CPR_team1',
    'ranksystem_max_COL_team2',
    'ranksystem_max_BPI_team1',
    'ranksystem_max_LEF_team1',
    'ranksystem_max_BNT_team1',
    'ranksystem_max_CJB_team1',
    'ranksystem_max_AP_team2',
    'ranksystem_max_BD_team1',
    'ranksystem_min_DAV_team1',
    'ranksystem_max_HAS_team1',
    'ranksystem_max_TRK_team1',
    'ranksystem_max_DCI_team2',
    'ranksystem_max_KCX_team1',
    'ranksystem_max_MOR_team1',
    'ranksystem_max_LMC_team1',
    'ranksystem_max_NOL_team1',
    'ranksystem_min_SEL_team2',
    'ranksystem_std_DUN_team2',
    'ranksystem_std_CBR_team1',
    'ranksystem_min_DCI_team1',
    'ranksystem_min_COL_team1',
    'ranksystem_max_SE_team1',
    'ranksystem_max_REW_team1',
    'ranksystem_max_DII_team2',
    'ranksystem_max_LYN_team1',
    'ranksystem_max_DC2_team1',
    'ranksystem_max_D1A_team1',
    'ranksystem_std_TRK_team2',
    'ranksystem_max_MAS_team1',
    'ranksystem_max_EBP_team2',
    'ranksystem_min_FDM_team1',
    'ranksystem_mean_FAS_team1',
    'ranksystem_min_KRA_team2',
    'ranksystem_max_BP5_team1',
    'ranksystem_max_FAS_team2',
    'ranksystem_std_SRS_team2',
    'ranksystem_std_INC_team1',
    'ranksystem_std_DOL_team1',
    'ranksystem_std_CWL_team1',
    'ranksystem_min_RPI_team1',
    'ranksystem_max_WIL_team1',
    'ranksystem_max_LEF_team2',
    'ranksystem_max_JCI_team1',
    'ranksystem_std_BIH_team1',
    'ranksystem_max_ECK_team1',
    'ranksystem_max_SRS_team1',
    'ranksystem_mean_DOL_team1',
    'ranksystem_min_DES_team2',
    'ranksystem_min_BBT_team2',
    'ranksystem_max_RTP_team2',
    'ranksystem_max_RPI_team1',
    'ranksystem_max_LMC_team2',
    'ranksystem_max_LAW_team1',
    'ranksystem_std_WOL_team2',
    'ranksystem_std_COL_team1',
    'ranksystem_max_RT_team1',
    'ranksystem_max_POM_team1',
    'ranksystem_std_BNZ_team2',
    'ranksystem_max_KRA_team2',
    'ranksystem_max_JJK_team2',
    'ranksystem_max_KPK_team2',
    'ranksystem_std_DUN_team1',
    'ranksystem_std_SFX_team1',
    'ranksystem_max_LOG_team2',
    'ranksystem_min_ARG_team1',
    'ranksystem_std_LAW_team2',
    'ranksystem_min_JNG_team2',
    'ranksystem_std_CNG_team2',
    'ranksystem_min_RT_team1',
    'ranksystem_max_SFX_team1',
    'ranksystem_max_EMK_team2',
    'ranksystem_std_BBT_team2',
    'ranksystem_max_RTR_team1',
    'ranksystem_max_NET_team2',
    'ranksystem_max_LOG_team1',
    'ranksystem_std_WLK_team2',
    'ranksystem_mean_WOL_team1',
    'ranksystem_std_LEF_team2',
    'ranksystem_std_JNG_team1',
    'ranksystem_min_CBR_team1',
    'ranksystem_min_RTH_team1',
    'ranksystem_max_BKM_team1',
    'ranksystem_max_PIR_team1',
    'ranksystem_max_BNM_team1',
    'ranksystem_max_DWH_team1',
    'ranksystem_max_BRZ_team1',
    'ranksystem_max_BOW_team1',
    'ranksystem_max_SPR_team1',
    'ranksystem_max_TPR_team1',
    'ranksystem_max_DDB_team1',
    'ranksystem_min_CPR_team1',
    'ranksystem_max_EBB_team1',
    'ranksystem_max_ESR_team1',
    'ranksystem_max_GRN_team1',
    'ranksystem_max_DOK_team2',
    'ranksystem_min_BBT_team1',
    'ranksystem_max_KOS_team1',
    'ranksystem_max_WOB_team1',
    'ranksystem_min_DOK_team1',
    'ranksystem_max_MB_team1',
    'ranksystem_max_PGH_team2',
    'ranksystem_max_DES_team2',
    'ranksystem_max_SPW_team1',
    'ranksystem_max_REI_team1',
    'ranksystem_max_KLK_team1',
    'ranksystem_max_INC_team2',
    'ranksystem_max_TRP_team1',
    'ranksystem_max_HAS_team2',
    'ranksystem_max_KCX_team2',
    'ranksystem_min_CNG_team1',
    'ranksystem_min_DUN_team1',
    'ranksystem_min_WOL_team1',
    'ranksystem_std_BOB_team1',
    'ranksystem_max_SAG_team2',
    'ranksystem_max_CMV_team1',
    'ranksystem_std_DCI_team1',
    'ranksystem_std_LAW_team1',
    'ranksystem_std_DII_team1',
    'ranksystem_std_CBR_team2',
    'ranksystem_min_RTP_team1',
    'ranksystem_min_PIG_team1',
    'ranksystem_min_MAS_team1',
    'ranksystem_max_HKB_team1',
    'ranksystem_std_MOR_team1',
    'ranksystem_std_FDM_team1',
    'ranksystem_max_MOR_team2',
    'ranksystem_std_DES_team2',
    'ranksystem_min_RWP_team2',
    'ranksystem_std_BNZ_team1',
    'ranksystem_min_SPR_team2',
    'ranksystem_max_FSH_team1',
    'ranksystem_max_COX_team1',
    'ranksystem_min_REW_team1',
    'ranksystem_mean_BIH_team1',
    'ranksystem_max_STH_team1',
    'ranksystem_std_DOK_team2',
    'ranksystem_max_PIR_team2',
    'ranksystem_min_WLK_team2',
    'ranksystem_min_POM_team1',
    'ranksystem_std_WOB_team2',
    'ranksystem_std_EMK_team1',
    'ranksystem_min_WOB_team1',
    'ranksystem_std_RPI_team1',
    'ranksystem_min_LOG_team1',
    'ranksystem_std_RTP_team2',
    'ranksystem_std_SFX_team2',
    'ranksystem_min_ARG_team2',
    'ranksystem_mean_JJK_team2',
    'ranksystem_mean_FDM_team1',
    'ranksystem_max_WOB_team2',
    'ranksystem_max_USA_team1',
    'ranksystem_max_SCR_team1',
    'ranksystem_max_REW_team2',
    'ranksystem_max_RT_team2',
    'ranksystem_std_PGH_team1',
    'ranksystem_min_FAS_team1',
    'ranksystem_min_BNZ_team2',
    'ranksystem_std_PGH_team2',
    'ranksystem_mean_WOB_team2',
    'ranksystem_max_HER_team1',
    'ranksystem_min_EMK_team1',
    'ranksystem_min_LAW_team2',
    'ranksystem_std_TPR_team2',
    'ranksystem_std_SRS_team1',
    'ranksystem_std_LYN_team1',
    'ranksystem_min_LAW_team1',
    'ranksystem_min_JJK_team2',
    'ranksystem_max_SMN_team1',
    'ranksystem_min_RTP_team2',
    'ranksystem_min_EBP_team2',
    'ranksystem_std_7OT_team2',
    'ranksystem_std_PIG_team1',
    'ranksystem_max_WMV_team1',
    'ranksystem_max_JEN_team1',
    'ranksystem_std_FAS_team2',
    'ranksystem_std_BBT_team1',
    'ranksystem_mean_COL_team1',
    'ranksystem_max_SFX_team2',
    'ranksystem_max_POM_team2',
    'ranksystem_min_CWL_team1',
'ranksystem_mean_RTH_team1',
'ranksystem_max_CRW_team1',
'ranksystem_min_AP_team1',
'ranksystem_max_FMG_team1',
'ranksystem_min_BWE_team1',
'ranksystem_max_ISR_team1',
'ranksystem_max_ERD_team1',
'ranksystem_min_LMC_team1',
'ranksystem_min_CRO_team1',
'ranksystem_max_SPR_team2',
'ranksystem_max_MSX_team1',
'ranksystem_min_EBP_team1',
'ranksystem_max_GRS_team1',
'ranksystem_min_KPK_team1',
'ranksystem_min_DII_team1',
'ranksystem_max_RTB_team1',
'ranksystem_max_SIM_team1',
'ranksystem_max_SP_team1',
'ranksystem_max_CTL_team1',
'ranksystem_std_KRA_team1',
'ranksystem_max_WLK_team2',
'ranksystem_min_HAS_team1',
'ranksystem_max_KPI_team1',
'ranksystem_std_CNG_team1',
'ranksystem_max_SAP_team1',
'ranksystem_max_TRP_team2',
'ranksystem_min_JJK_team1',
'ranksystem_max_IMS_team1',
'ranksystem_max_OMY_team1',
'ranksystem_std_PIR_team1',
'ranksystem_mean_RT_team2',
'ranksystem_std_SEL_team1',
'ranksystem_min_SEL_team1',
'ranksystem_min_7OT_team2',
'ranksystem_min_DOL_team1',
'ranksystem_std_DES_team1',
'ranksystem_min_WLK_team1',
'ranksystem_max_SRS_team2',
'ranksystem_std_DC_team1',
'ranksystem_std_SEL_team2',
'ranksystem_std_RTP_team1',
'ranksystem_min_WIL_team1',
'ranksystem_min_SRS_team1',
'ranksystem_min_SAG_team1',
'ranksystem_min_RWP_team1',
'ranksystem_mean_TPR_team1',
'ranksystem_max_YAG_team1',
'ranksystem_max_MCL_team1',
'ranksystem_mean_7OT_team2',
'ranksystem_mean_ARG_team2',
'ranksystem_mean_NET_team1',
'ranksystem_mean_DAV_team1',
'ranksystem_min_NET_team1',
'ranksystem_max_HOL_team1',
'ranksystem_min_CWL_team2',
'ranksystem_min_LOG_team2',
'ranksystem_std_LOG_team2',
'ranksystem_min_FDM_team2',
'ranksystem_max_TW_team1',
'ranksystem_max_KBM_team1',
'ranksystem_max_HRN_team1',
'ranksystem_min_BWE_team2',
'ranksystem_std_JJK_team1',
'ranksystem_std_POM_team2',
'ranksystem_min_LEF_team2',
'ranksystem_max_TRK_team2',
'ranksystem_std_MB_team1',
'ranksystem_min_USA_team1',
'ranksystem_min_KRA_team1',
'ranksystem_min_FSH_team1',
'ranksystem_min_BIH_team2',
'ranksystem_mean_SE_team1',
'ranksystem_mean_LEF_team1',
'ranksystem_max_RIS_team1',
'ranksystem_min_JNG_team1',
'ranksystem_std_LOG_team1',
'ranksystem_min_KPK_team2',
'ranksystem_std_POM_team1',
'ranksystem_std_ARG_team2',
'ranksystem_min_MOR_team1',
'ranksystem_std_MOR_team2',
'ranksystem_mean_CBR_team1',
'ranksystem_max_TMR_team1',
'ranksystem_std_EBP_team1',
'ranksystem_min_DOK_team2',
'ranksystem_max_TSR_team1',
'ranksystem_min_DAV_team2',
'ranksystem_min_BOB_team1',
'ranksystem_min_DOL_team2',
'ranksystem_max_WMV_team2',
'ranksystem_min_POM_team2',
'ranksystem_max_GC_team1',
'ranksystem_std_FAS_team1',
'ranksystem_min_NET_team2',
'ranksystem_mean_SFX_team2',
'ranksystem_mean_POM_team1',
'ranksystem_mean_BBT_team1',
'ranksystem_max_SGR_team1',
'ranksystem_max_HKS_team1',
'ranksystem_std_WMV_team2',
'ranksystem_max_MKV_team1',
'ranksystem_std_KRA_team2',
'ranksystem_std_BWE_team1',
'ranksystem_mean_RPI_team2',
'ranksystem_max_KMV_team1',
'ranksystem_min_KCX_team1',
'ranksystem_max_RTH_team2',
'ranksystem_max_HAT_team1',
'ranksystem_min_DES_team1',
'ranksystem_max_LYD_team1',
'ranksystem_max_JON_team1',
'ranksystem_max_STR_team1',
'ranksystem_min_NOL_team1',
'ranksystem_std_HAS_team1',
'ranksystem_min_STH_team1',
'ranksystem_max_PTS_team1',
'ranksystem_std_NOL_team1',
'ranksystem_max_INP_team1',
'ranksystem_min_AP_team2',
'ranksystem_min_MB_team1',
'ranksystem_min_DC_team1',
'ranksystem_min_SFX_team1',
'ranksystem_max_MMG_team1',
'ranksystem_min_CPA_team1',
'ranksystem_std_SAG_team1',
'ranksystem_min_COL_team2',
'ranksystem_std_LEF_team1',
'ranksystem_min_SMS_team1',
'ranksystem_min_BUR_team1',
'ranksystem_min_BLS_team1',
'ranksystem_std_RT_team1',
'ranksystem_min_PIR_team1',
'ranksystem_min_SPR_team1',
'ranksystem_min_BPI_team1',
'ranksystem_std_SPR_team1',
'ranksystem_max_DC_team2',
'ranksystem_min_ADE_team1',
'ranksystem_max_USA_team2',
'ranksystem_min_DII_team2',
'ranksystem_max_PRR_team1',
'ranksystem_min_DCI_team2',
'ranksystem_min_SPW_team1',
'ranksystem_std_ARG_team1',
'ranksystem_std_REW_team1',
'ranksystem_mean_DOK_team1',
'ranksystem_std_CWL_team2',
'ranksystem_min_TPR_team1',
'ranksystem_std_SE_team1',
'ranksystem_std_FDM_team2',
'ranksystem_max_JRT_team1',
'ranksystem_min_TRP_team1',
'ranksystem_std_7OT_team1',
'ranksystem_min_HKB_team1',
'ranksystem_min_HAS_team2',
'ranksystem_mean_SMS_team1',
'ranksystem_std_LMC_team2',
'ranksystem_std_MKV_team1',
'ranksystem_min_WMV_team1',
'ranksystem_std_SPR_team2',
'ranksystem_mean_BNZ_team1',
'ranksystem_std_INC_team2',
'ranksystem_min_SRS_team2',
'ranksystem_min_GRN_team1',
'ranksystem_mean_PGH_team2',
'ranksystem_mean_DUN_team1',
'ranksystem_max_RM_team1',
'ranksystem_max_REN_team1',
'ranksystem_max_MvG_team1',
'ranksystem_std_HAS_team2',
'ranksystem_mean_GRN_team1',
'ranksystem_std_RTR_team1',
'ranksystem_std_EMK_team2',
'ranksystem_std_BIH_team2',
'ranksystem_min_SFX_team2',
'ranksystem_mean_PGH_team1',
'ranksystem_mean_TPR_team2',
'ranksystem_min_DUN_team2',
'ranksystem_std_NOL_team2',
'ranksystem_min_LMC_team2',
'ranksystem_min_WIL_team2',
'ranksystem_mean_DCI_team2',
'ranksystem_mean_KPK_team1',
'ranksystem_std_WIL_team2',
'ranksystem_std_WOB_team1',
'ranksystem_std_MAS_team1',
'ranksystem_min_MAS_team2',
'ranksystem_mean_WLK_team1',
'ranksystem_mean_SAG_team1',
'ranksystem_max_RAG_team1',
'ranksystem_max_LAB_team1',
'ranksystem_max_CNG_team2',
'ranksystem_min_SMS_team2',
'ranksystem_min_PIR_team2',
'ranksystem_std_KPK_team2',
'ranksystem_min_SP_team1',
'ranksystem_std_WMV_team1',
'ranksystem_mean_SAG_team2',
'ranksystem_mean_LAW_team2',
'ranksystem_max_SE_team2',
'ranksystem_mean_RWP_team1',
'ranksystem_min_FAS_team2',
'ranksystem_std_TRP_team2',
'ranksystem_std_SIM_team1',
'ranksystem_mean_DOK_team2',
'ranksystem_mean_CBR_team2',
'ranksystem_max_STM_team1',
'ranksystem_min_REW_team2',
'ranksystem_min_INC_team2',
'ranksystem_max_MPI_team1',
'ranksystem_min_ENT_team1',
'ranksystem_mean_CWL_team1',
'ranksystem_max_WOL_team2',
'ranksystem_max_KEL_team1',
'ranksystem_min_TRK_team1',
'ranksystem_max_MGS_team1',
'ranksystem_std_NET_team1',
'ranksystem_max_PKL_team1',
'ranksystem_std_TRK_team1',
'ranksystem_std_REI_team1',
'ranksystem_min_JEN_team1',
'ranksystem_max_ROH_team1',
'ranksystem_max_ROG_team1',
'ranksystem_std_SPW_team1',
'ranksystem_std_USA_team1',
'ranksystem_std_NET_team2',
'ranksystem_std_ADE_team1',
'ranksystem_std_WIL_team1',
'ranksystem_min_EMK_team2',
'ranksystem_std_STH_team1',
'ranksystem_mean_CNG_team1',
'ranksystem_min_LYN_team1',
'ranksystem_min_RTB_team1',
'ranksystem_mean_ARG_team1',
'ranksystem_min_REI_team1',
'ranksystem_mean_INC_team1',
'ranksystem_max_WMR_team1',
'ranksystem_min_DC2_team1',
'ranksystem_std_SMS_team1',
'ranksystem_min_DWH_team1',
'ranksystem_min_ACU_team1',
'ranksystem_min_TMR_team1',
'ranksystem_max_PH_team1',
'ranksystem_min_KCX_team2',
'ranksystem_min_MOR_team2',
'ranksystem_std_DII_team2',
'ranksystem_min_PGH_team2',
'ranksystem_min_ISR_team1',
'ranksystem_max_MGY_team1',
'ranksystem_mean_NOL_team2',
'ranksystem_std_DCI_team2',
'ranksystem_max_MIC_team1',
'ranksystem_min_USA_team2',
'ranksystem_std_CPR_team1',
'ranksystem_min_TW_team1',
'ranksystem_mean_USA_team2',
'ranksystem_mean_KCX_team1',
'ranksystem_max_STF_team1',
'ranksystem_min_SAG_team2',
'ranksystem_mean_AP_team2',
'ranksystem_std_BUR_team1',
'ranksystem_mean_AP_team1',
'ranksystem_mean_SPR_team2',
'ranksystem_mean_WOB_team1',
'ranksystem_min_NOL_team2',
'ranksystem_mean_BWE_team1',
'ranksystem_min_KBM_team1',
'ranksystem_std_BLS_team1',
'ranksystem_max_MUZ_team1',
'ranksystem_std_DAV_team2',
'ranksystem_min_WOB_team2',
'ranksystem_min_D1A_team1',
'ranksystem_mean_SRS_team1',
'ranksystem_mean_SEL_team1',
'ranksystem_mean_DES_team1',
'ranksystem_max_PMC_team1',
'ranksystem_mean_HAS_team2',
'ranksystem_mean_KPK_team2',
'ranksystem_std_TRP_team1',
'ranksystem_min_TPR_team2',
'ranksystem_min_REN_team1',
'ranksystem_mean_PIR_team2',
'ranksystem_mean_DWH_team1',
'ranksystem_mean_RPI_team1',
'ranksystem_mean_FAS_team2',
'ranksystem_mean_TRK_team2',
'ranksystem_std_KCX_team2',
'ranksystem_std_EBP_team2',
'ranksystem_std_DC2_team1',
'ranksystem_std_RPI_team2',
'ranksystem_mean_DUN_team2',
'ranksystem_mean_USA_team1',
'ranksystem_mean_BIH_team2',
'ranksystem_std_RT_team2',
'ranksystem_mean_RWP_team2',
'ranksystem_max_SAU_team1',
'ranksystem_mean_REW_team1',
'ranksystem_mean_BBT_team2',
'ranksystem_min_RT_team2',
'ranksystem_mean_LEF_team2',
'ranksystem_min_TRK_team2',
'ranksystem_min_MSX_team1',
'ranksystem_min_MPI_team1',
'ranksystem_max_STS_team1',
'ranksystem_max_RPI_team2',
'ranksystem_mean_DAV_team2',
'ranksystem_std_D1A_team1',
'ranksystem_min_BCM_team1',
'ranksystem_mean_INC_team2',
'ranksystem_std_AP_team2',
'ranksystem_mean_KCX_team2',
'ranksystem_mean_7OT_team1',
'ranksystem_min_ESR_team1',
'ranksystem_mean_EMK_team1',
'ranksystem_min_RTH_team2',
'ranksystem_max_NOR_team1',
'ranksystem_mean_DCI_team1',
'ranksystem_max_OCT_team1',
'ranksystem_max_PEQ_team1',
'ranksystem_max_TRX_team1',
'ranksystem_mean_DII_team1',
'ranksystem_min_RTR_team1',
'ranksystem_max_UCS_team1',
'ranksystem_mean_JJK_team1',
'ranksystem_std_RTB_team1',
'ranksystem_min_PTS_team1',
'ranksystem_mean_HAS_team1',
'ranksystem_mean_EBP_team1',
'ranksystem_mean_KRA_team1',
'ranksystem_std_BPI_team1',
'ranksystem_min_TRP_team2',
'ranksystem_std_CRO_team1',
'ranksystem_mean_MAS_team1',
'ranksystem_min_SE_team1',
'ranksystem_min_BNT_team1',
'ranksystem_min_KLK_team1',
'ranksystem_std_REW_team2',
'ranksystem_mean_CPA_team1',
'ranksystem_std_SAG_team2',
'ranksystem_min_CJB_team1',
'ranksystem_max_ZAM_team1',
'ranksystem_max_TOL_team1',
'ranksystem_std_PIR_team2',
'ranksystem_std_ACU_team1',
'ranksystem_min_KOS_team1',
'ranksystem_mean_REW_team2',
'ranksystem_mean_CPR_team1',
'ranksystem_mean_COL_team2',
'ranksystem_mean_CWL_team2',
'ranksystem_min_ECK_team1',
'ranksystem_std_RTH_team2',
'ranksystem_max_RSE_team1',
'ranksystem_mean_LMC_team1',
'ranksystem_std_GRN_team1',
'ranksystem_mean_SPR_team1',
'ranksystem_max_RSL_team1',
'ranksystem_min_EBB_team1',
'ranksystem_mean_BWE_team2',
'ranksystem_mean_BUR_team1',
'ranksystem_min_TSR_team1',
'ranksystem_min_SAP_team1',
'ranksystem_mean_MAS_team2',
'ranksystem_min_WMV_team2',
'ranksystem_min_FMG_team1',
'ranksystem_mean_EMK_team2',
'ranksystem_min_GRS_team1',
'ranksystem_mean_TRK_team1',
'ranksystem_mean_LAW_team1',
'ranksystem_mean_FDM_team2',
'ranksystem_mean_JNG_team1',
'ranksystem_mean_SEL_team2',
'ranksystem_std_SMS_team2',
'ranksystem_min_KPI_team1',
'ranksystem_max_RTR_team2',
'ranksystem_max_PPR_team1',
'ranksystem_std_CPA_team1',
'ranksystem_mean_MOR_team2',
'ranksystem_mean_KRA_team2',
'ranksystem_max_CPR_team2',
'ranksystem_mean_POM_team2',
'ranksystem_min_TRX_team1',
'ranksystem_std_BOB_team2',
'ranksystem_std_FSH_team1',
'ranksystem_min_SCR_team1',
'ranksystem_std_HER_team1',
'ranksystem_mean_PIG_team2',
'ranksystem_mean_DII_team2',
'ranksystem_std_KPI_team1',
'ranksystem_min_WOL_team2',
'ranksystem_mean_SRS_team2',
'ranksystem_max_TBD_team1',
'ranksystem_mean_LOG_team1',
'ranksystem_max_UPS_team1',
'ranksystem_max_WLS_team1',
'ranksystem_min_BD_team1',
'ranksystem_std_SP_team1',
'ranksystem_mean_NOL_team1',
'ranksystem_min_OMY_team1',
'ranksystem_mean_PIR_team1',
'ranksystem_min_SIM_team1',
'ranksystem_std_TW_team1',
'ranksystem_min_LYD_team1',
'ranksystem_min_JCI_team1',
'ranksystem_mean_SMS_team2',
'ranksystem_std_DWH_team1',
'ranksystem_min_DDB_team1',
'ranksystem_mean_NET_team2',
'ranksystem_std_JEN_team1',
'ranksystem_mean_LOG_team2',
'ranksystem_mean_EBP_team2',
'ranksystem_mean_SFX_team1',
'ranksystem_std_ECK_team1',
'ranksystem_min_HER_team1',
'ranksystem_mean_DES_team2',
'ranksystem_mean_DC_team1',
'ranksystem_mean_LYN_team1',
'ranksystem_min_BNM_team1',
'ranksystem_mean_MB_team1',
'ranksystem_min_HOL_team1',
'ranksystem_mean_LMC_team2',
'ranksystem_mean_WIL_team1',
'ranksystem_mean_WMV_team2',
'ranksystem_mean_RT_team1',
'ranksystem_mean_RTP_team1',
'ranksystem_mean_CNG_team2',
'ranksystem_min_STM_team1',
'ranksystem_mean_STH_team1',
'ranksystem_mean_WOL_team2',
'ranksystem_min_BP5_team1',
'ranksystem_min_AUS_team1',
'ranksystem_std_TSR_team1',
'ranksystem_std_KOS_team1',
'ranksystem_std_KLK_team1',
'ranksystem_min_UCS_team1',
'ranksystem_mean_DC_team2',
'ranksystem_max_WTE_team1',
'ranksystem_mean_WIL_team2',
'ranksystem_mean_PIG_team1',
'ranksystem_std_HKB_team1',
'ranksystem_mean_BOB_team2',
'ranksystem_min_BKM_team1',
'ranksystem_mean_TRP_team1',
'ranksystem_max_STH_team2',
'ranksystem_std_PIG_team2',
'ranksystem_mean_WMV_team1',
'ranksystem_std_SE_team2',
'ranksystem_mean_ECK_team1',
'ranksystem_mean_TRP_team2',
'ranksystem_min_IMS_team1',
'ranksystem_min_MvG_team1',
'ranksystem_min_MMG_team1',
'ranksystem_min_KMV_team1',
'ranksystem_mean_SAU_team1',
'ranksystem_mean_RTH_team2',
'ranksystem_min_AWS_team1',
'ranksystem_min_MCL_team1',
'ranksystem_std_MSX_team1',
'ranksystem_min_MKV_team1',
'ranksystem_mean_CRO_team1',
'ranksystem_std_CJB_team1',
'ranksystem_std_TMR_team1',
'ranksystem_min_SGR_team1',
'ranksystem_min_ROH_team1',
'ranksystem_std_ISR_team1',
'ranksystem_std_ENT_team1',
'ranksystem_mean_SPW_team1',
'ranksystem_mean_RTR_team1',
'ranksystem_max_PIG_team2',
'ranksystem_min_CMV_team1',
'ranksystem_mean_DC2_team1',
'ranksystem_mean_HKB_team1',
'ranksystem_std_GRN_team2',
'ranksystem_min_SAU_team1',
'ranksystem_mean_RTR_team2',
'ranksystem_mean_REI_team1',
'ranksystem_std_KBM_team1',
'ranksystem_min_HAT_team1',
'ranksystem_min_CTL_team1',
'ranksystem_min_BRZ_team1',
'ranksystem_mean_SP_team1',
'ranksystem_std_OMY_team1',
'ranksystem_std_BCM_team1',
'ranksystem_std_STR_team1',
'ranksystem_min_HRN_team1',
'ranksystem_min_PKL_team1',
'ranksystem_max_CPA_team2',
'ranksystem_min_YAG_team1',
'ranksystem_std_RTR_team2',
'ranksystem_std_FMG_team1',
'ranksystem_min_WMR_team1',
'ranksystem_min_INP_team1',
'ranksystem_min_CRW_team1',
'ranksystem_min_BOW_team1',
'ranksystem_max_MCL_team2',
'ranksystem_std_PTS_team1',
'ranksystem_min_ZAM_team1',
'ranksystem_min_RPI_team2',
'ranksystem_min_JON_team1',
'ranksystem_std_RIS_team1',
'ranksystem_min_CNG_team2',
'ranksystem_mean_SCR_team1',
'ranksystem_min_PIG_team2',
'ranksystem_mean_ENT_team1',
'ranksystem_min_COX_team1',
'ranksystem_min_SMN_team1',
'ranksystem_min_ERD_team1',
'ranksystem_min_HKS_team1',
'ranksystem_mean_ADE_team1',
'ranksystem_min_STF_team1',
'ranksystem_std_SGR_team1',
'ranksystem_min_RAG_team1',
'ranksystem_min_STR_team1',
'ranksystem_std_ESR_team1',
'ranksystem_min_MGY_team1',
'ranksystem_min_PRR_team1',
'ranksystem_min_GC_team1',
'ranksystem_min_ROG_team1',
'ranksystem_mean_ACU_team1',
'ranksystem_min_WTE_team1',
'ranksystem_std_SCR_team1',
'ranksystem_mean_TW_team1',
'ranksystem_std_JON_team1',
'ranksystem_std_BD_team1',
'ranksystem_min_MIC_team1',
'ranksystem_max_BOB_team2',
'ranksystem_min_MUZ_team1',
'ranksystem_min_RIS_team1',
'ranksystem_mean_SE_team2',
'ranksystem_min_KEL_team1',
'ranksystem_min_BOB_team2',
'ranksystem_min_PPR_team1',
'ranksystem_mean_RTB_team1',
'ranksystem_mean_ESR_team1',
'ranksystem_max_MSX_team2',
'ranksystem_max_GRN_team2',
'ranksystem_min_JRT_team1',
'ranksystem_mean_BLS_team1',
'ranksystem_mean_TSR_team1',
'ranksystem_min_PH_team1',
'ranksystem_min_NOR_team1',
'ranksystem_std_MB_team2',
'ranksystem_std_ERD_team1',
'ranksystem_min_PEQ_team1',
'ranksystem_mean_SPW_team2',
'ranksystem_mean_RAG_team1',
'ranksystem_mean_BPI_team1',
'ranksystem_max_FSH_team2',
'ranksystem_std_UCS_team1',
'ranksystem_mean_HER_team1',
'ranksystem_std_LYD_team1',
'ranksystem_min_SE_team2',
'ranksystem_min_HER_team2',
'ranksystem_mean_RTB_team2',
'ranksystem_max_TW_team2',
'ranksystem_max_BUR_team2',
'ranksystem_std_SPW_team2',
'ranksystem_std_GRS_team1',
'ranksystem_std_SAU_team1',
'ranksystem_min_RM_team1',
'ranksystem_mean_SIM_team1',
'ranksystem_max_SIM_team2',
'ranksystem_max_ACU_team2',
'ranksystem_mean_JEN_team2',
'ranksystem_std_BPI_team2',
'ranksystem_min_MGS_team1',
'ranksystem_min_DC_team2',
'ranksystem_std_ZAM_team1',
'ranksystem_std_STH_team2',
'ranksystem_std_ROG_team1',
'ranksystem_std_BRZ_team1',
'ranksystem_min_RSE_team1',
'ranksystem_min_LAB_team1',
'ranksystem_mean_MSX_team1',
'ranksystem_std_JCI_team1',
'ranksystem_std_CMV_team1',
'ranksystem_mean_BD_team1',
'ranksystem_std_BNT_team1',
'ranksystem_min_SPW_team2',
'ranksystem_max_DWH_team2',
'ranksystem_max_SPW_team2',
'ranksystem_max_TSR_team2',
'ranksystem_min_OCT_team1',
'ranksystem_std_HOL_team1',
'ranksystem_mean_JEN_team1',
'ranksystem_min_PMC_team1',
'ranksystem_std_GC_team1',
'ranksystem_min_STS_team1',
'ranksystem_std_MGY_team1',
'ranksystem_mean_D1A_team1',
'ranksystem_std_KMV_team1',
'ranksystem_std_REN_team1',
'ranksystem_mean_CJB_team1',
'ranksystem_max_MB_team2',
'ranksystem_mean_GRS_team1',
'ranksystem_std_PKL_team1',
'ranksystem_std_IMS_team1',
'ranksystem_std_MCL_team1',
'ranksystem_std_SMN_team1',
'ranksystem_std_SAP_team1',
'ranksystem_min_RSL_team1',
'ranksystem_std_DWH_team2',
'ranksystem_std_TW_team2',
'ranksystem_min_TBD_team1',
'ranksystem_std_PRR_team1',
'ranksystem_max_BLS_team2',
'ranksystem_std_ROH_team1',
'ranksystem_mean_STH_team2',
'ranksystem_mean_ADE_team2',
'ranksystem_min_RTR_team2',
'ranksystem_min_CPA_team2',
'ranksystem_max_CRO_team2',
'ranksystem_std_WMR_team1',
'ranksystem_mean_CPR_team2',
'ranksystem_mean_KBM_team1',
'ranksystem_std_RAG_team1',
'ranksystem_max_LYN_team2',
'ranksystem_mean_TMR_team1',
'ranksystem_mean_HOL_team1',
'ranksystem_min_CPR_team2',
'ranksystem_std_JRT_team1',
'ranksystem_std_BLS_team2',
'ranksystem_min_KLK_team2',
'ranksystem_mean_OMY_team1',
'ranksystem_mean_REN_team1',
'ranksystem_std_TSR_team2',
'ranksystem_std_LYN_team2',
'ranksystem_std_BKM_team1',
'ranksystem_std_AUS_team1',
'ranksystem_min_WLS_team1',
'ranksystem_mean_TSR_team2',
'ranksystem_mean_SAP_team1',
'ranksystem_mean_BNT_team1',
'ranksystem_mean_BCM_team1',
'ranksystem_max_GRS_team2',
'ranksystem_mean_FSH_team1',
'ranksystem_min_TOL_team1',
'ranksystem_max_ECK_team2',
'ranksystem_mean_MKV_team1',
'ranksystem_std_SP_team2',
'ranksystem_mean_CPA_team2',
'ranksystem_max_BD_team2',
'ranksystem_max_BCM_team2',
'ranksystem_std_MPI_team1',
'ranksystem_std_INP_team1',
'ranksystem_max_ENT_team2',
'ranksystem_std_WTE_team1',
'ranksystem_min_SIM_team2',
'ranksystem_mean_SMN_team1',
'ranksystem_mean_PTS_team1',
'ranksystem_min_STH_team2',
'ranksystem_min_DWH_team2',
'ranksystem_min_UPS_team1',
'ranksystem_std_NOR_team1',
'ranksystem_mean_KPI_team1',
'ranksystem_std_AWS_team1',
'ranksystem_std_RM_team1',
'ranksystem_std_BNM_team1',
'ranksystem_mean_BRZ_team1',
'ranksystem_mean_JCI_team1',
'ranksystem_std_MMG_team1',
'ranksystem_mean_ROH_team1',
'ranksystem_mean_FMG_team1',
'ranksystem_max_REI_team2',
'ranksystem_mean_ISR_team1',
'ranksystem_max_ADE_team2',
'ranksystem_std_TRX_team1',
'ranksystem_mean_KLK_team1',
'ranksystem_max_D1A_team2',
'ranksystem_std_BUR_team2',
'ranksystem_max_HKB_team2',
'ranksystem_mean_DDB_team1',
'ranksystem_mean_BLS_team2',
'ranksystem_std_BOW_team1',
'ranksystem_mean_RIS_team1',
'ranksystem_max_HER_team2',
'ranksystem_std_STS_team1',
'ranksystem_mean_DWH_team2',
'ranksystem_mean_ERD_team1',
'ranksystem_max_JEN_team2',
'ranksystem_max_BPI_team2',
'ranksystem_min_CRO_team2',
'ranksystem_min_ACU_team2',
'ranksystem_max_SP_team2',
'ranksystem_max_RTB_team2',
'ranksystem_mean_TW_team2',
'ranksystem_mean_GRN_team2',
'ranksystem_max_MKV_team2',
'ranksystem_std_DC2_team2',
'ranksystem_min_GRN_team2',
'ranksystem_mean_SP_team2',
'ranksystem_max_REN_team2',
'ranksystem_max_KPI_team2',
'ranksystem_std_PPR_team1',
'ranksystem_std_COX_team1',
'ranksystem_mean_CMV_team1',
'ranksystem_max_SCR_team2',
'ranksystem_min_ECK_team2',
'ranksystem_mean_JON_team1',
'ranksystem_min_TSR_team2',
'ranksystem_mean_KOS_team1',
'ranksystem_max_DC2_team2',
'ranksystem_min_LYN_team2',
'ranksystem_mean_PKL_team1',
'ranksystem_std_STF_team1',
'ranksystem_std_STM_team1',
'ranksystem_std_MSX_team2',
'ranksystem_std_KEL_team1',
'ranksystem_mean_SGR_team1',
'ranksystem_mean_ECK_team2',
'ranksystem_std_SAP_team2',
'ranksystem_mean_REI_team2',
'ranksystem_max_JCI_team2',
'ranksystem_mean_IMS_team1',
'ranksystem_min_HKB_team2',
'ranksystem_max_ZAM_team2',
'ranksystem_min_BUR_team2',
'ranksystem_mean_LYN_team2',
'ranksystem_std_ECK_team2',
'ranksystem_min_MB_team2',
'ranksystem_min_ENT_team2',
'ranksystem_max_ROH_team2',
'ranksystem_min_JCI_team2',
'ranksystem_min_BCM_team2',
'ranksystem_std_JCI_team2',
'ranksystem_mean_PKL_team2',
'ranksystem_min_REN_team2',
'ranksystem_std_HAT_team1',
'ranksystem_std_CTL_team1',
'ranksystem_std_DDB_team1',
'ranksystem_mean_REN_team2',
'ranksystem_mean_GC_team1',
'ranksystem_std_YAG_team1',
'ranksystem_max_ISR_team2',
'ranksystem_mean_EBB_team1',
'ranksystem_min_PKL_team2',
'ranksystem_mean_MMG_team1',
'ranksystem_std_LAB_team1',
'ranksystem_mean_MIC_team1',
'ranksystem_mean_STR_team1',
'ranksystem_mean_KMV_team1',
'ranksystem_std_MUZ_team1',
'ranksystem_max_KLK_team2',
'ranksystem_mean_WMR_team1',
'ranksystem_std_RSE_team1',
'ranksystem_std_KPI_team2',
'ranksystem_std_PEQ_team1',
'ranksystem_mean_TRX_team1',
'ranksystem_std_RSL_team1',
'ranksystem_mean_STS_team1',
'ranksystem_min_SGR_team2',
'ranksystem_mean_YAG_team1',
'ranksystem_std_TRX_team2',
'ranksystem_mean_MGY_team1',
'ranksystem_mean_BOW_team1',
'ranksystem_mean_MCL_team1',
'ranksystem_mean_ISR_team2',
'ranksystem_std_REI_team2',
'ranksystem_std_MGS_team1',
'ranksystem_max_SAP_team2',
'ranksystem_mean_LYD_team1',
'ranksystem_max_CMV_team2',
'ranksystem_mean_BUR_team2',
'ranksystem_min_SP_team2',
'ranksystem_std_CRO_team2',
'ranksystem_max_HOL_team2',
'ranksystem_mean_UCS_team1',
'ranksystem_mean_AUS_team1',
'ranksystem_std_KLK_team2',
'ranksystem_min_OMY_team2',
'ranksystem_mean_ROG_team1',
'ranksystem_mean_ZAM_team1',
'ranksystem_max_SMN_team2',
'ranksystem_min_FSH_team2',
'ranksystem_std_BCM_team2',
'ranksystem_std_UPS_team1',
'ranksystem_mean_STM_team1',
'ranksystem_min_ADE_team2',
'ranksystem_mean_MPI_team1',
'ranksystem_std_TOL_team1',
'ranksystem_std_OCT_team1',
'ranksystem_std_HKS_team1',
'ranksystem_min_LYD_team2',
'ranksystem_max_ROG_team2',
'ranksystem_max_RAG_team2',
'ranksystem_mean_WTE_team1',
'ranksystem_mean_SAP_team2',
'ranksystem_std_BD_team2',
'ranksystem_mean_NOR_team1',
'ranksystem_max_SAU_team2',
'ranksystem_max_RIS_team2',
'ranksystem_max_TRX_team2',
'ranksystem_min_GRS_team2',
'ranksystem_mean_TRX_team2',
'ranksystem_max_SGR_team2',
'ranksystem_std_UCS_team2',
'ranksystem_std_SIM_team2',
'ranksystem_std_ENT_team2',
'ranksystem_min_BD_team2',
'ranksystem_mean_MB_team2',
'ranksystem_mean_JCI_team2',
'ranksystem_mean_CTL_team1',
'ranksystem_max_WMR_team2',
'ranksystem_max_LYD_team2',
'ranksystem_max_STR_team2',
'ranksystem_min_STR_team2',
'ranksystem_std_STR_team2',
'ranksystem_std_GRS_team2',
'ranksystem_std_D1A_team2',
'ranksystem_mean_PEQ_team1',
'ranksystem_mean_KPI_team2',
'ranksystem_max_MGY_team2',
'ranksystem_mean_RM_team1',
'ranksystem_std_ACU_team2',
'ranksystem_std_JEN_team2',
'ranksystem_max_GC_team2',
'ranksystem_std_FSH_team2',
'ranksystem_std_SMN_team2',
'ranksystem_min_TW_team2',
'ranksystem_mean_SIM_team2',
'ranksystem_max_ESR_team2',
'ranksystem_mean_MvG_team1',
'ranksystem_max_BRZ_team2',
'ranksystem_std_TBD_team1',
'ranksystem_std_WLS_team1',
'ranksystem_mean_BKM_team1',
'ranksystem_max_CJB_team2',
'ranksystem_mean_BNM_team1',
'ranksystem_max_PKL_team2',
'ranksystem_mean_KEL_team1',
'ranksystem_std_FMG_team2',
'ranksystem_mean_PRR_team1',
'ranksystem_max_EBB_team2',
'ranksystem_max_UCS_team2',
'ranksystem_mean_BP5_team1',
'ranksystem_std_REN_team2',
'ranksystem_max_KOS_team2',
'ranksystem_std_CPA_team2',
'ranksystem_max_FMG_team2',
'ranksystem_min_ISR_team2',
'ranksystem_mean_COX_team1',
'ranksystem_max_ERD_team2',
'ranksystem_mean_PH_team1',
'ranksystem_mean_STF_team1',
'ranksystem_mean_AWS_team1',
'ranksystem_mean_RSE_team1',
'ranksystem_max_KBM_team2',
'ranksystem_std_RTB_team2',
'ranksystem_max_MMG_team2',
'ranksystem_min_SCR_team2',
'ranksystem_max_MPI_team2',
'ranksystem_max_KMV_team2',
'ranksystem_max_IMS_team2',
'ranksystem_std_PTS_team2',
'ranksystem_max_PTS_team2',
'ranksystem_mean_INP_team1',
'ranksystem_mean_SCR_team2',
'ranksystem_max_YAG_team2',
'ranksystem_mean_TBD_team1',
'ranksystem_min_REI_team2',
'ranksystem_mean_CRW_team1',
'ranksystem_mean_HAT_team1',
'ranksystem_min_D1A_team2',
'ranksystem_mean_HKS_team1',
'ranksystem_max_BNT_team2',
'ranksystem_min_ROH_team2',
'ranksystem_mean_MUZ_team1',
'ranksystem_mean_BD_team2',
'ranksystem_min_RIS_team2',
'ranksystem_min_IMS_team2',
'ranksystem_min_RTB_team2',
'ranksystem_min_RAG_team2',
'ranksystem_max_OMY_team2',
'ranksystem_mean_HRN_team1',
'ranksystem_min_SAP_team2',
'ranksystem_max_WTE_team2',
'ranksystem_std_SCR_team2',
'ranksystem_max_BNM_team2',
'ranksystem_std_HER_team2',
'ranksystem_max_JON_team2',
'ranksystem_min_BPI_team2',
'ranksystem_max_DDB_team2',
'ranksystem_min_FMG_team2',
'ranksystem_mean_WLS_team1',
'ranksystem_max_BKM_team2',
'ranksystem_min_TMR_team2',
'ranksystem_max_MvG_team2',
'ranksystem_max_NOR_team2',
'ranksystem_max_TMR_team2',
'ranksystem_min_HOL_team2',
'ranksystem_mean_OCT_team1',
'ranksystem_max_MUZ_team2',
'ranksystem_min_DC2_team2',
'ranksystem_max_AUS_team2',
'ranksystem_std_CPR_team2',
'ranksystem_min_ZAM_team2',
'ranksystem_max_STF_team2',
'ranksystem_max_RM_team2',
'ranksystem_max_PRR_team2',
'ranksystem_max_MIC_team2',
'ranksystem_max_AWS_team2',
'ranksystem_min_SAU_team2',
'ranksystem_max_HRN_team2',
'ranksystem_min_JEN_team2',
'ranksystem_std_HKB_team2',
'ranksystem_min_ESR_team2',
'ranksystem_min_BRZ_team2',
'ranksystem_min_BNT_team2',
'ranksystem_max_RSE_team2',
'ranksystem_mean_D1A_team2',
'ranksystem_mean_CRO_team2',
'ranksystem_std_CJB_team2',
'ranksystem_mean_MSX_team2',
'ranksystem_min_GC_team2',
'ranksystem_mean_KLK_team2',
'ranksystem_mean_MGS_team1',
'ranksystem_mean_HKB_team2',
'ranksystem_mean_ENT_team2',
'ranksystem_min_BLS_team2',
'ranksystem_min_UCS_team2',
'ranksystem_max_BP5_team2',
'ranksystem_max_MGS_team2',
'ranksystem_max_PH_team2',
'ranksystem_mean_JRT_team1',
'ranksystem_mean_LAB_team1',
'ranksystem_min_CMV_team2',
'ranksystem_min_BKM_team2',
'ranksystem_min_EBB_team2',
'ranksystem_min_MSX_team2',
'ranksystem_min_CJB_team2',
'ranksystem_min_ERD_team2',
'ranksystem_min_RSE_team2',
'ranksystem_min_KPI_team2',
'ranksystem_mean_PMC_team1',
'ranksystem_min_KMV_team2',
'ranksystem_min_MKV_team2',
'ranksystem_std_LYD_team2',
'ranksystem_std_HOL_team2',
'ranksystem_max_STM_team2',
'ranksystem_std_ISR_team2',
'ranksystem_min_TRX_team2',
'ranksystem_std_AUS_team2',
'ranksystem_max_PEQ_team2',
'ranksystem_mean_ACU_team2',
'ranksystem_max_BOW_team2',
'ranksystem_max_COX_team2',
'ranksystem_mean_PPR_team1',
'ranksystem_min_KBM_team2',
'ranksystem_mean_UPS_team1',
'ranksystem_std_PKL_team2',
'ranksystem_max_CRW_team2',
'ranksystem_mean_TOL_team1',
'ranksystem_mean_BPI_team2',
'ranksystem_std_MKV_team2',
'ranksystem_max_STS_team2',
'ranksystem_std_ADE_team2',
'ranksystem_max_CTL_team2',
'ranksystem_mean_DC2_team2',
'ranksystem_std_KBM_team2',
'ranksystem_std_CMV_team2',
'ranksystem_std_BKM_team2',
'ranksystem_min_SMN_team2',
'ranksystem_max_UPS_team2',
'ranksystem_std_RIS_team2',
'ranksystem_std_SAU_team2',
'ranksystem_min_AUS_team2',
'ranksystem_min_ROG_team2',
'ranksystem_min_WTE_team2',
'ranksystem_std_KMV_team2',
'ranksystem_min_WLS_team2',
'ranksystem_std_RSE_team2',
'ranksystem_min_STM_team2',
'ranksystem_min_NOR_team2',
'ranksystem_max_TBD_team2',
'ranksystem_min_PTS_team2',
'ranksystem_mean_LYD_team2',
'ranksystem_max_WLS_team2',
'ranksystem_min_KOS_team2',
'ranksystem_min_MGY_team2',
'ranksystem_min_AWS_team2',
'ranksystem_max_PPR_team2',
'ranksystem_std_MGY_team2',
'ranksystem_mean_UCS_team2',
'ranksystem_mean_IMS_team2',
'ranksystem_std_IMS_team2',
'ranksystem_mean_FSH_team2',
'ranksystem_mean_GRS_team2',
'ranksystem_max_HKS_team2',
'ranksystem_std_TMR_team2',
'ranksystem_std_PEQ_team2',
'ranksystem_min_PEQ_team2',
'ranksystem_min_JON_team2',
'ranksystem_min_MIC_team2',
'ranksystem_max_PMC_team2',
'ranksystem_mean_HER_team2',
'ranksystem_std_OMY_team2',
'ranksystem_min_OCT_team2',
'ranksystem_mean_RSL_team1',
'ranksystem_std_RAG_team2',
'ranksystem_mean_TMR_team2',
'ranksystem_mean_CJB_team2',
'ranksystem_std_BRZ_team2',
'ranksystem_std_ROH_team2',
'ranksystem_mean_STR_team2',
'ranksystem_min_YAG_team2',
'ranksystem_mean_OMY_team2',
'ranksystem_min_WMR_team2',
'ranksystem_min_RM_team2',
'ranksystem_min_PH_team2',
'ranksystem_max_HAT_team2',
'ranksystem_max_INP_team2',
'ranksystem_max_KEL_team2',
'ranksystem_min_BOW_team2',
'ranksystem_max_TOL_team2',
'ranksystem_max_JRT_team2',
'ranksystem_min_CTL_team2',
'ranksystem_std_KOS_team2',
'ranksystem_min_HRN_team2',
'ranksystem_mean_ESR_team2',
'ranksystem_std_WTE_team2',
'ranksystem_min_KEL_team2',
'ranksystem_min_DDB_team2',
'ranksystem_min_MPI_team2',
'ranksystem_std_KEL_team2',
'ranksystem_mean_MKV_team2',
'ranksystem_mean_BCM_team2',
'ranksystem_mean_NOR_team2',
'ranksystem_min_MMG_team2',
'ranksystem_std_ERD_team2',
'ranksystem_max_LAB_team2',
'ranksystem_min_MvG_team2',
'ranksystem_std_JON_team2',
'ranksystem_mean_KOS_team2',
'ranksystem_min_BNM_team2',
'ranksystem_min_BP5_team2',
'ranksystem_min_MCL_team2',
'ranksystem_mean_RIS_team2',
'ranksystem_std_WMR_team2',
'ranksystem_min_STF_team2',
'ranksystem_mean_SMN_team2',
'ranksystem_max_OCT_team2',
'ranksystem_mean_SGR_team2',
'ranksystem_mean_ROH_team2',
'ranksystem_mean_PTS_team2',
'ranksystem_mean_RAG_team2',
'ranksystem_min_HAT_team2',
'ranksystem_mean_BKM_team2',
'ranksystem_std_ZAM_team2',
'ranksystem_min_STS_team2',
'ranksystem_std_TOL_team2',
'ranksystem_min_CRW_team2',
'ranksystem_std_MPI_team2',
'ranksystem_std_ESR_team2',
'ranksystem_std_BNM_team2',
'ranksystem_std_BNT_team2',
'ranksystem_std_MGS_team2',
'ranksystem_std_COX_team2',
'ranksystem_mean_GC_team2',
'ranksystem_std_STM_team2',
'ranksystem_min_PRR_team2',
'ranksystem_std_MMG_team2',
'ranksystem_mean_HOL_team2',
'ranksystem_min_COX_team2',
'ranksystem_std_PRR_team2',
'ranksystem_min_TBD_team2',
'ranksystem_min_INP_team2',
'ranksystem_mean_WMR_team2',
'ranksystem_std_RM_team2',
'ranksystem_mean_RSE_team2',
'ranksystem_mean_EBB_team2',
'ranksystem_std_NOR_team2',
'ranksystem_min_JRT_team2',
'ranksystem_mean_SAU_team2',
'ranksystem_std_SGR_team2',
'ranksystem_std_ROG_team2',
'ranksystem_std_YAG_team2',
'ranksystem_std_STF_team2',
'ranksystem_std_GC_team2',
'ranksystem_mean_FMG_team2',
'ranksystem_std_STS_team2',
'ranksystem_mean_KBM_team2',
'ranksystem_mean_ZAM_team2',
'ranksystem_mean_MGY_team2',
'ranksystem_mean_BRZ_team2',


'ranksystem_mean_YAG_team2',
'ranksystem_mean_MPI_team2',
'ranksystem_mean_KMV_team2',
'ranksystem_mean_PH_team2',
'ranksystem_mean_JON_team2',
'ranksystem_mean_ERD_team2',
'ranksystem_mean_CMV_team2',
'ranksystem_std_BOW_team2',
'ranksystem_min_TOL_team2',
'ranksystem_std_TBD_team2',
'ranksystem_max_RSL_team2',
'ranksystem_mean_AUS_team2',
'ranksystem_min_PPR_team2',
'ranksystem_std_HAT_team2',
'ranksystem_min_UPS_team2',
'ranksystem_mean_KEL_team2',
'ranksystem_min_LAB_team2',
'ranksystem_mean_BNT_team2',
'ranksystem_std_MCL_team2',
'ranksystem_mean_TOL_team2',
'ranksystem_min_HKS_team2',
'ranksystem_std_PPR_team2',
'ranksystem_mean_DDB_team2',
'ranksystem_mean_STF_team2',
'ranksystem_min_MUZ_team2',
'ranksystem_mean_MIC_team2',
'ranksystem_mean_MvG_team2',
'ranksystem_mean_STS_team2',
'ranksystem_mean_RM_team2',
'ranksystem_mean_BP5_team2',
'ranksystem_mean_HRN_team2',
'ranksystem_mean_WTE_team2',
'ranksystem_min_MGS_team2',
'ranksystem_mean_PRR_team2',
'ranksystem_mean_BOW_team2',
'ranksystem_mean_HAT_team2',
'ranksystem_min_PMC_team2',
'ranksystem_mean_ROG_team2',
'ranksystem_std_UPS_team2',
'ranksystem_std_AWS_team2',
'ranksystem_std_CTL_team2',
'ranksystem_mean_PEQ_team2',
'ranksystem_mean_BNM_team2',
'ranksystem_mean_CTL_team2',
'ranksystem_std_WLS_team2',
'ranksystem_std_MUZ_team2',
'ranksystem_mean_MGS_team2',
'ranksystem_mean_STM_team2',
'ranksystem_mean_INP_team2',
'ranksystem_mean_MCL_team2',
'ranksystem_std_RSL_team2',
'ranksystem_std_JRT_team2',
'ranksystem_mean_MMG_team2',
'ranksystem_mean_CRW_team2',
'ranksystem_std_DDB_team2',
'ranksystem_mean_AWS_team2',
'ranksystem_mean_TBD_team2',

'ranksystem_mean_PPR_team2',
'ranksystem_std_LAB_team2',
'ranksystem_min_RSL_team2',
'ranksystem_mean_WLS_team2',

'ranksystem_mean_UPS_team2',
'ranksystem_mean_COX_team2',
'ranksystem_std_HKS_team2',

]
    rank_cols = getColumnsFromParts(["ranksystem_"], df_train.columns)
    df_train=df_train[rank_cols]
    df_test = df_test[rank_cols]
    adversarialValidation(df_train, df_test, drop_cols=drop_cols)
    

def procEDA(_df_train, _df_test, eda_cols=[], convert_obj=False):
    
 
    drop_cols = get_useless_columnsTrainTest(_df_train, _df_test, null_rate=0.95, repeat_rate=0.95)
    sys.exit()
    
    
    target_col = "y"
    
    
    
    if len(eda_cols) == 0:
        eda_cols = list(_df_train.columns)
        df_train = _df_train
        df_test = _df_test
        
        eda_cols.remove(target_col)
    else:
    
        tmp_cols = eda_cols + [target_col]
        df_train = _df_train[tmp_cols]
        df_test = _df_test[eda_cols]
    
    
    
    if convert_obj:
      for col in eda_cols:
          for tmp_df in [df_train, df_test]:
              tmp_df[col] = tmp_df[col].astype("object")  
    
    
    showDetails(df_train, df_test, eda_cols, target_col, corr_flag=False)
    


if __name__ == '__main__':
    
    #df_train, df_test = loadRaw()
    df_train, df_test = loadProc(stage=2, decode_flag=False)
    procEDA2(df_train, df_test)
    #procAdv(df_train, df_test)
    #tmp1(df_train, df_test)
    #procSave()
    