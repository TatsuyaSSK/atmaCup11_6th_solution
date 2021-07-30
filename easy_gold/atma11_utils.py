# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 02:20:17 2021

@author: r00526841
"""

import numpy as np
import pandas as pd

from utils import *
from image_utils import *
from MyFoldSplit import StratifiedKFoldWithGroupID



def train_val_split_for_ttl():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')

    ppath_to_dir = INPUT_DIR/"photos"
    ppath_to_imagenet_dir = INPUT_DIR/"imagenet"
    os.makedirs(ppath_to_imagenet_dir, exist_ok=True)


    se_y = df_train["target"]
    df_X = df_train #.drop(columns=["target"])
    folds = StratifiedKFoldWithGroupID(n_splits=2, group_id_col="art_series_id", stratified_target_id="target")
    for train_index, test_index in folds.split(df_X, se_y, _group=None):
        print(f"train : {train_index}")
        print(f"val : {test_index}")

        for idx in train_index:
            object_id = df_train.index[idx]
            label = int(df_train.iloc[idx]["target"])

            pp_dir = ppath_to_imagenet_dir / f"train/{label}"
            os.makedirs(pp_dir, exist_ok=True)
            new_pp = pp_dir / f"{object_id}.jpg"
            ppath_to_image = ppath_to_dir / f"{object_id}.jpg"
            shutil.copy(ppath_to_image, new_pp)

        for idx in test_index:
            object_id = df_train.index[idx]
            label = int(df_train.iloc[idx]["target"])

            pp_dir = ppath_to_imagenet_dir / f"val/{label}"
            os.makedirs(pp_dir, exist_ok=True)
            new_pp = pp_dir / f"{object_id}.jpg"
            ppath_to_image = ppath_to_dir / f"{object_id}.jpg"
            shutil.copy(ppath_to_image, new_pp)


           
        break


    pdb.set_trace()


def copyImg():
    
    ppath_to_dir = INPUT_DIR/"photos"
    ppath_to_new_photo_dir = INPUT_DIR/"new_photos"
    os.makedirs(ppath_to_new_photo_dir, exist_ok=True)

    
    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    #print(f"load df_train : {df_train.shape}")
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_nn.pkl')
    
    for index, row in df_train.iterrows():
        target_num = row["target"]
        ppath_to_image = ppath_to_dir / row["image_name"]
        img = Image.open(ppath_to_image)
        
        # pp_dir = ppath_to_dir / f"train/{target_num}"
        # os.makedirs(pp_dir, exist_ok=True)
        
        
        # new_pp = pp_dir / row["image_name"]
        # shutil.copy(ppath_to_image, new_pp)

        pp_dir = ppath_to_dir / f"train_salient/{target_num}"
        os.makedirs(pp_dir, exist_ok=True)
        new_pp = pp_dir / row["image_name"]

        salient_img = getSaliencyImg(str(ppath_to_image), salient_type="SR")
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_salient.jpg"), (salient_img * 255).astype("uint8"))


        img = cv2.imread(str(ppath_to_image))
        dst, dst_salient = getCenteringImgFromSaliencyImg(img, salient_img)

        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_center_img.jpg"), dst)
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}.jpg"), img)
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_center_salient_img.jpg"), (dst_salient * 255).astype("uint8"))

        
        
    for index, row in df_test.iterrows():
        
        ppath_to_image = ppath_to_dir / row["image_name"]
        pp_dir = ppath_to_dir / f"test"
        os.makedirs(pp_dir, exist_ok=True)
        
        
        new_pp = pp_dir / row["image_name"]
        shutil.copy(ppath_to_image, new_pp)

        salient_img = getSaliencyImg(str(ppath_to_image), salient_type="SR")
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_salient.jpg"), (salient_img * 255).astype("uint8"))


        img = cv2.imread(str(ppath_to_image))
        dst, dst_salient = getCenteringImgFromSaliencyImg(img, salient_img)

        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_center_img.jpg"), dst)
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}.jpg"), img)
        cv2.imwrite(str(ppath_to_new_photo_dir / f"{index}_center_salient_img.jpg"), (dst_salient * 255).astype("uint8"))

        
def drop_art_series(df_train):

    gp = df_train.groupby("art_series_id")["target"].nunique()
    drop_series = gp[gp>1].index
    df_train = df_train.loc[~df_train["art_series_id"].isin(drop_series)]   

    return df_train


def proc_old_oof():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    df_train = drop_art_series(df_train)

    file_prefix = "20210712_132557_ResNet_Wrapper--0.808721--"
    df_oof = pd.read_csv(OUTPUT_DIR/f"ave/{file_prefix}_oof.csv", index_col="object_id") 


    df_oof = df_oof.loc[df_oof.index.isin(df_train.index)]
    df_oof = df_oof.reset_index()
    df_oof.to_csv(OUTPUT_DIR/f"ave/{file_prefix}_oof.csv", index=False)

        
def sub_round():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    df_train = drop_art_series(df_train)
    file_prefix = "20210715-202758_20210716_161903_multiLabelNet--0.523242--"
    df_oof = pd.read_csv(OUTPUT_DIR/f"{file_prefix}_oof.csv", index_col="object_id") 
    df_sub = pd.read_csv(OUTPUT_DIR/f"{file_prefix}_submission.csv")
    print(df_sub.describe())

    pdb.set_trace()
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
    df_sub.to_csv(OUTPUT_DIR/f"{file_prefix}_submission_round.csv", index=False)
    

def lgb_prepro():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    df_train = drop_art_series(df_train)
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_nn.pkl')

    sub_name = "20210715_181909_multiLabelNet--0.850927--"
    df_oof = pd.read_csv(OUTPUT_DIR/f"{sub_name}_oof.csv", index_col="object_id") 
    df_sub = pd.read_csv(OUTPUT_DIR/f"{sub_name}_submission.csv")
    

    df_sub.index = df_test.index


    df_oof = df_oof.rename(columns={"target":"pred_target"})
    df_sub = df_sub.rename(columns={"target":"pred_target"})

    df_oof["target"] = df_train["target"]
    df_oof["art_series_id"] = df_train["art_series_id"]
    df_sub["art_series_id"] = df_test["art_series_id"]
    #df_sub["target"] = df_test["target"]

    df_oof.to_pickle(PROC_DIR/"df_proc_train_lgb.pkl")
    df_sub.to_pickle(PROC_DIR/"df_proc_test_lgb.pkl")

def force_continuous(prob_val, alpha=0.8):

    
    #prob_val = numpy_normalize(prob_val**alpha)
    return np.sum(prob_val*np.arange(4), axis=1)
        
def round_by_class_pro():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    df_train = drop_art_series(df_train)
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_nn.pkl')

    sub_name = "20210721-104210_20210721_104340_SimpleStackingWrapper--0.656662--"
    df_oof = pd.read_csv(OUTPUT_DIR/f"{sub_name}_oof.csv")
    if "object_id" in df_oof.columns:
        df_oof = df_oof.set_index("object_id")
    elif "index" in df_oof.columns:
        df_oof = df_oof.set_index("index")
        df_oof.index.name = "object_id"
    
    df_sub = pd.read_csv(OUTPUT_DIR/f"{sub_name}_submission.csv")
    df_sub.index = df_test.index

    sub_cl_name = "20210721-110813_20210721_144921_multiLabelNet--0.824911--"
    df_oof_cl = pd.read_csv(OUTPUT_DIR/f"{sub_cl_name}_oof.csv")
    if "object_id" in df_oof_cl.columns:
        df_oof_cl = df_oof_cl.set_index("object_id")
    elif "index" in df_oof_cl.columns:
        df_oof_cl = df_oof_cl.set_index("index")
        df_oof_cl.index.name = "object_id"

    df_sub_cl = pd.read_csv(OUTPUT_DIR/f"{sub_cl_name}_submission.csv")
    df_sub_cl.index = df_test.index

    df_oof["target_prob"] = df_oof_cl["target_prob"]
    df_oof["target_int"] = df_oof_cl["target_int"]

    df_oof["target"] = [c if p>=0.9 else r for r, c, p in zip(df_oof["target"], df_oof["target_int"], df_oof["target_prob"])]
    df_sub["target"] = [c if p>=0.9 else r for r, c, p in zip(df_sub["target"], df_sub_cl["target_int"], df_sub_cl["target_prob"])]

    df_train["oof"] =  df_oof["target"]
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_true=df_train["target"].values, y_pred=df_train["oof"].values))

    print(f"new rmse : {rmse}")
    df_oof.to_csv(OUTPUT_DIR/f"round_{rmse}_{sub_name}_oof.csv")
    df_sub.to_csv(OUTPUT_DIR/f"round_{rmse}_{sub_name}_submission.csv", index=False)


def checkImaghash():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_nn.pkl')
    df_train = drop_art_series(df_train)
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_nn.pkl')

    df = pd.concat([df_train, df_test])

    print(df.shape)

    hashes = []
    object_ids = []
    for object_id, rows in df.iterrows():
        ppath_to_img = INPUT_DIR/f"photos/{object_id}.jpg"
        img = Image.open(ppath_to_img)
        
        hash = getImageHash(img)
        hashes.append(hash)
        object_ids.append(object_id)

        #df.loc[object_id, "image_hash"] = hash
    
    hashes_all = np.array(hashes).astype(int)
    #hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()
    sims = np.array([(hashes_all[i] == hashes_all).sum(axis=1)/256 for i in range(hashes_all.shape[0])])

    threhold = 0.89

    indices1 = np.where(sims > threhold)
    indices2 = np.where(indices1[0] != indices1[1])
    object_ids1 = [object_ids[i] for i in indices1[0][indices2]]
    object_ids2 = [object_ids[i] for i in indices1[1][indices2]]
    dups = {tuple(sorted([object_id1, object_id2])):True for object_id1, object_id2 in zip(object_ids1, object_ids2)}
    print(f'found {len(dups)} duplicates')
    pdb.set_trace()

    f = open(PROC_DIR/'dups.npy')
    pickle.dump(dups,f)

    np.save(PROC_DIR/'dups.npy', dups)

    dups = np.load(PROC_DIR/'dups.npy', allow_pickle='TRUE')
    
if __name__ == '__main__':
    
    checkImaghash()