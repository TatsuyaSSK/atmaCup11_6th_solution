# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 02:41:05 2021

@author: r00526841
"""

from utils import *
from PIL import Image

def getImageStatistics(df, ppath_to_dir, ppath_to_label_dir=None):

    for index, row in df.iterrows():
        ppath_to_image = ppath_to_dir / row["image_name"]
        img = Image.open(ppath_to_image)
        np_im = np.array(img)
        
        df.loc[index, "img_H"] = np_im.shape[0]
        df.loc[index, "img_W"] = np_im.shape[1]
        df.loc[index, "img_C"] = np_im.shape[2]
        
        df.loc[index, "img_R_mean"] = np_im[...,0].mean()
        df.loc[index, "img_G_mean"] = np_im[...,1].mean()
        df.loc[index, "img_B_mean"] = np_im[...,2].mean()
        
        df.loc[index, "img_R_std"] = np_im[...,0].std()
        df.loc[index, "img_G_std"] = np_im[...,1].std()
        df.loc[index, "img_B_std"] = np_im[...,2].std()
        
        df.loc[index, "img_R_min"] = np_im[...,0].min()
        df.loc[index, "img_G_min"] = np_im[...,1].min()
        df.loc[index, "img_B_min"] = np_im[...,2].min()
        
        df.loc[index, "img_R_max"] = np_im[...,0].max()
        df.loc[index, "img_G_max"] = np_im[...,1].max()
        df.loc[index, "img_B_max"] = np_im[...,2].max()
        
        #pdb.set_trace()
        
        
    # for p in ppath_to_dir.iterdir():

    #     di = {}
        
    #     sar_img_list = ["0_VH", "1_VH", "0_VV", "1_VV"]
    #     for sar_name in sar_img_list:
    #         ppath_to_tif = p/f"{sar_name}.tif"
    #         img = Image.open(ppath_to_tif)
    #         np_im = np.array(img)
    #         di[f"{sar_name}_path_to_tif"] = ppath_to_tif
    #         di[f"{sar_name}_H"] = np_im.shape[0]
    #         di[f"{sar_name}_W"] = np_im.shape[1]
    #         di[f"{sar_name}_mean"] = np_im.mean()
    #         di[f"{sar_name}_std"] = np_im.std()
    #         di[f"{sar_name}_max"] = np_im.max()
    #         di[f"{sar_name}_min"] = np_im.min()
        

    #     if ppath_to_label_dir is not None:
    #         ppath_to_label = ppath_to_label_dir/f"{p.name}.png"
    #         label_img = Image.open(ppath_to_label)
    #         np_label_img = np.array(label_img)

    #         di["label_path"] = ppath_to_label
    #         di["label_H"] = np_label_img.shape[0]
    #         di["label_W"] = np_label_img.shape[1]
    #         di["label_mean"] = np_label_img.mean()
    #         di["label_std"] = np_label_img.std()
    #         di["label_max"] = np_label_img.max()
    #         di["label_min"] = np_label_img.min()
    #         di["num_1"] = np.count_nonzero(np_label_img)
    #         di["num_0"] = np_label_img.size - di["num_1"] 
    #         di["rate_new_building"] = float(di["num_1"]) / float(np_label_img.size)

    #     df_each = pd.DataFrame(di, index=[p.name])

    #     df = df.append(df_each)
        

    return df