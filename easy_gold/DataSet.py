

from utils import *
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_utils import *

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class MyDatasetEmbedding(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, transform=None):

        
        self.transform_ = transform
        self.np_train_X_cat_ = df_train_X[dataset_params["category_features_list"]].values.astype(np.int64)
        self.np_train_X_cont_ = df_train_X[dataset_params["continuous_features_list"]].values.astype(np.float32)
        self.np_train_y_ = df_train_y.values.astype(np.float32).reshape(-1, 1)
        #print((self.np_train_X_cat_ ))
        #print(type(self.np_train_y_ ))

    def __len__(self):
        return self.np_train_X_cont_.shape[0]

    def __getitem__(self, idx):
        #print("idx:{}".format(idx))
        if len(self.np_train_X_cat_) > 0:
            out_data_cat = self.np_train_X_cat_[idx, :]
        else:
            out_data_cat=[]
        out_data_cont = self.np_train_X_cont_[idx, :]
        #print("out_data:{}".format(out_data))

        out_label =  self.np_train_y_[idx]
        #print("out_label:{}".format(out_label))

        if self.transform_ != None:
            pass

        return [out_data_cat, out_data_cont], out_label   

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, transform=None):
        self.transform_ = transform
        self.np_train_X_ = df_train_X.values.astype(np.float32)
        self.np_train_y_ = df_train_y.values.astype(np.float32).reshape(-1, 1)
        #print(type(self.np_train_X_ ))
        #print(type(self.np_train_y_ ))

    def __len__(self):
        return self.np_train_X_.shape[0]

    def __getitem__(self, idx):
        #print("idx:{}".format(idx))

        out_data = self.np_train_X_[idx, :]
        #print("out_data:{}".format(out_data))

        out_label =  self.np_train_y_[idx]
        #print("out_label:{}".format(out_label))

        if self.transform_ != None:
            pass

        return [out_data], out_label 


class SupConDataset(torch.utils.data.Dataset):
    def __init__(
        self, df_train_X, df_train_y, dataset_params, train_flag
    ):
        self.df_train_X = df_train_X
        self.df_train_y = df_train_y
        #self.transform = dataset_params["transform"]
        self.img_size = dataset_params["img_size"]
        self.salient_flag = dataset_params["salient_flag"]
        self.train_flag = train_flag
        

        if self.salient_flag:
            IMG_MEAN = [0.485, 0.456, 0.406, 0.5]
            IMG_STD = [0.229, 0.224, 0.225, 0.5]
        else:
            IMG_MEAN = [0.485, 0.456, 0.406]
            IMG_STD = [0.229, 0.224, 0.225]
        if train_flag:
            comp_list = [
                            A.Resize(p=1.0, height=self.img_size, width=self.img_size),
                            A.RandomResizedCrop(p=1.0, height=self.img_size, width=self.img_size, scale=(0.5, 1.0)),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.HueSaturationValue(p=0.8),
                            A.ToGray(p=0.2),
                            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
                            ToTensorV2(always_apply=True),
                            ]
        else:
            comp_list = [A.Resize(p=1.0, height=self.img_size, width=self.img_size),
                         A.Normalize(mean=IMG_MEAN, std=IMG_STD),
                        ToTensorV2(always_apply=True),
            
            ]
        self.transform = A.Compose(comp_list)

    def __len__(self):
        return self.df_train_X.shape[0]

    def __getitem__(self, idx):
        
        image_name = self.df_train_X.iloc[idx]['image_name']
        ppath_to_img = INPUT_DIR/f"photos/{image_name}"
        img = Image.open(ppath_to_img)
        img = np.array(img)

        if self.salient_flag:
            
            salient_img = getSaliencyImg(path_to_image=str(ppath_to_img), salient_type="SR")
            
            #print(f"salient_img {idx}* {salient_img.shape}")
            #print(f"img {idx}* {np.array(img).shape}")
            img = np.dstack([img, salient_img])
            #print(f"after img {idx} * {img.shape}")
        img_1 = self.transform(image=img)["image"]
        img_2 = self.transform(image=img)["image"]
        #img = [img_1, img_2]
        
        w = self.df_train_X.iloc[idx]['loss_weight']
        
        return [img_1, img_2, w], self.df_train_y.iloc[idx].values#[0]



class MyDatasetResNet(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag):
        
        self.df_train_X = df_train_X
        self.df_train_y = df_train_y
        self.img_size = dataset_params["img_size"]
        self.salient_flag = dataset_params["salient_flag"]
        self.regression_flag = dataset_params["regression_flag"]

        #if self.salient_flag:
        #    IMG_MEAN = [0.485, 0.456, 0.406, 0.5] #, 0.485, 0.456, 0.406, 0.5]
        #    IMG_STD = [0.229, 0.224, 0.225, 0.5] #, 0.229, 0.224, 0.225, 0.5]
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]
        if train_flag:
            comp_list = [
                            A.Resize(p=1.0, height=self.img_size, width=self.img_size),
                            #A.RandomResizedCrop(p=1.0, height=self.img_size, width=self.img_size, scale=(0.5, 1.0)),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.CoarseDropout(),
                            A.ImageCompression(),
                            #A.ISONoise(),
                            A.MultiplicativeNoise(),
                            #A.HueSaturationValue(p=0.8),
                            #A.ToGray(p=0.2),
                            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
                            ToTensorV2(always_apply=True),
                            ]
        else:
            comp_list = [A.Resize(p=1.0, height=self.img_size, width=self.img_size),
                         A.Normalize(mean=IMG_MEAN, std=IMG_STD),
                        ToTensorV2(always_apply=True),
            
            ]
        self.transformer = A.Compose(comp_list)

        self.distance_matrix = np.array([[0,1,4,9],[1,0,1,4],[4,1,0,1],[9,4,1,0]])
        self.label_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        #size = (224, 224)
        #size = (300, 300)
        # size = (self.img_size, self.img_size)
        # additional_items = (
        #     [T.Resize(size)]
        #     if not train_flag
        #     else [
        #         #T.RandomGrayscale(p=0.2),
        #         T.RandomVerticalFlip(),
        #         T.RandomHorizontalFlip(),
        #         # T.ColorJitter(
        #         #     brightness=0.3,
        #         #     contrast=0.5,
        #         #     saturation=[0.8, 1.3],
        #         #     hue=[-0.05, 0.05],
        #         # ),
        #         T.RandomResizedCrop(size),
        #         T.Resize(size),
        #     ]
        # )

        # self.transformer = T.Compose(
        #     [*additional_items, T.ToTensor(), T.Normalize(mean=IMG_MEAN, std=IMG_STD)]
        # )
        
    def __len__(self):
        return self.df_train_X.shape[0]

    def __getitem__(self, idx):
        
        image_name = self.df_train_X.iloc[idx]['image_name']
        ppath_to_img = INPUT_DIR/f"photos/{image_name}"
        img = Image.open(ppath_to_img)
        img = np.array(img)
        
        if self.salient_flag:
            #img = np.array(img)
            #salient_img = getSaliencyImg(path_to_image=str(ppath_to_img), salient_type="SR")
            #img = np.dstack([img, salient_img])

            
            index = self.df_train_X.index[idx]
            ppath_to_new_dir = INPUT_DIR/f"new_photos"

            #img = np.array(Image.open(ppath_to_new_dir/f"{index}.jpg"))
            #saliency_img = np.array(Image.open(ppath_to_new_dir/f"{index}_salient.jpg"))
            img_center = np.array(Image.open(ppath_to_new_dir/f"{index}_center_img.jpg"))
            saliency_img_center = np.array(Image.open(ppath_to_new_dir/f"{index}_center_salient_img.jpg"))
            #img = np.dstack([img, saliency_img, img_center, saliency_img_center])
            img = np.dstack([img_center, saliency_img_center])
            
            
            
            

            #.set_trace()

        #img = self.transformer(img)
        img = self.transformer(image=img)["image"]

        w = self.df_train_X.iloc[idx]['loss_weight']

        label = self.df_train_y.iloc[idx].values#[0]
        if not self.regression_flag:
            #print(f"label : {label}")
            w = self.distance_matrix[int(label), :]
            label = self.label_matrix[int(label), :]
        
        return [img, w],label


class SequenceTransformer(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        
        np_sample = np.array(sample.values.tolist()).transpose().astype(np.int64)
        #print(f"trans : {np_sample}")
        #print(f"trans : {np_sample.shape}")
        #sys.exit()
        return np_sample



class MyDatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag):

        self.last_query_flag = dataset_params["last_query_flag"]
        
        self.max_seq = dataset_params["max_seq"]
        self.pad_num = dataset_params["pad_num"]
        self.sequence_features_list = dataset_params["sequence_features_list"]
        self.continuous_features_list = dataset_params["continuous_features_list"]
        self.bssid_features_list = dataset_params["bssid_features_list"]
        self.fq_features_list = dataset_params["fq_features_list"]
        self.label_col = dataset_params["label_col"]
        self.row_index_name = df_train_X.index.name

        print(df_train_X.columns)
        print(df_train_y.columns)
        print(self.sequence_features_list)
        print(self.continuous_features_list)
        
        #self.df_site = df_train_X[["site_id"]]
        
        # for i in range(self.max_seq):
        #     self.df_site[f"site_id_{i}"] = self.df_site["site_id"]
        # self.df_site.drop(columns=["site_id"], inplace=True)
        
        #self.df_bssid = df_train_X[self.sequence_features_list]
        #self.df_rssi = df_train_X[self.continuous_features_list].astype("float32")

        # print(self.df_rssi.iloc[0].values.dtype)
        
        # sys.exit()
        
        #self.df_y = df_train_y

        for col in self.label_col:
            if train_flag:
                
                df_train_X[col]=df_train_y[col]
            else:
                df_train_X[col]=0

        gp_key_id = "path"
        time_id = "t1_wifi"

        prev_renew_cols = ["prev_x", "prev_y"]

        num_label_col = len(self.label_col)
        use_cols = self.sequence_features_list + self.continuous_features_list + self.label_col
        self.prev_renew_idx = [use_cols.index(c) for c in prev_renew_cols]
        print(use_cols)

        df_grp = df_train_X.reset_index().groupby(gp_key_id, sort=False)


        np_x_list = []
        np_y_list = []
        np_mask_list = []
        group_id_list = []
        raw_id_list = []
        for i, (id, gp) in enumerate(df_grp):
            gp = gp.sort_values(time_id)
            
            
            raw_id = gp.index.values

            np_gp = gp[use_cols].values
            #assert np_gp.shape[0] >=  self.max_seq
            

            np_gp = make_time_series(np_gp, windows_size=self.max_seq, pad_num=self.pad_num)
            #np_gp = np_gp[self.max_seq-1:]
        
            np_x = np_gp[..., :-num_label_col]
            np_y = np_gp[..., -num_label_col:]

            #print(np_gp.shape)
            #pdb.set_trace()

            #pad mask
            np_mask = np.all(np.equal(np_x, self.pad_num),axis=-1)
            
            np_x_list.append(np_x)
            np_mask_list.append(np_mask)
            group_id_list+= [i] * np_x.shape[0]

            if self.last_query_flag:
                np_y_list.append(np_y[:, -1, :]) #last
            else:
                #pdb.set_trace()
                np_y_list.append(np_y) #last
                raw_id = make_time_series(raw_id.reshape(-1, 1), windows_size=self.max_seq, pad_num=-1)
                #pdb.set_trace()
                # = raw_id[self.max_seq-1:]
               

            raw_id_list.append(raw_id)



            #pdb.set_trace()

        self.np_x = np.concatenate(np_x_list)
        self.np_y = np.concatenate(np_y_list)
        self.np_mask = np.concatenate(np_mask_list)
        self.group_id_list = group_id_list
        self.np_raw_id = np.concatenate(raw_id_list)

        # print(self.np_x.shape)
        # print(self.np_y.shape)
        # print(self.np_mask.shape)
        # print(f"len :{len(self.group_id_list)}")
        # pdb.set_trace()
        
        
        



    def __len__(self):
        return self.np_x.shape[0]

    def __getitem__(self, idx):
        
        #print(f"len :{len(self.group_id_list)}, {idx}")
   
        return [self.np_x[idx], self.np_mask[idx], self.group_id_list[idx], self.prev_renew_idx, self.np_raw_id[idx]], self.np_y[idx]


class MyDatasetLSTM2(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag):

        self.last_query_flag = dataset_params["last_query_flag"]
        
        self.max_seq = dataset_params["max_seq"]
        self.pad_num = dataset_params["pad_num"]
        self.sequence_features_list = dataset_params["sequence_features_list"]
        self.continuous_features_list = dataset_params["continuous_features_list"]
        self.weight_list = dataset_params["weight_list"]
        self.use_feature_cols = dataset_params["use_feature_cols"]
        self.label_col = dataset_params["label_col"]
        self.row_index_name = df_train_X.index.name

        print(df_train_X.columns)
        print(df_train_y.columns)
        print(self.sequence_features_list)
        print(self.continuous_features_list)
        
        #self.df_site = df_train_X[["site_id"]]
        
        # for i in range(self.max_seq):
        #     self.df_site[f"site_id_{i}"] = self.df_site["site_id"]
        # self.df_site.drop(columns=["site_id"], inplace=True)
        
        #self.df_bssid = df_train_X[self.sequence_features_list]
        #self.df_rssi = df_train_X[self.continuous_features_list].astype("float32")

        # print(self.df_rssi.iloc[0].values.dtype)
        
        # sys.exit()
        
        #self.df_y = df_train_y

        for col in self.label_col:
            if train_flag:
                
                df_train_X[col]=df_train_y[col]
            else:
                df_train_X[col]=0

        gp_key_id = "path"
        time_id = "timestamp"

        

        num_label_col = len(self.label_col)
        num_weight_col = len(self.weight_list)
        use_cols = self.use_feature_cols + self.weight_list+ self.label_col
        print(use_cols)

        df_grp = df_train_X.reset_index().groupby(gp_key_id, sort=False)


        np_x_list = []
        np_w_list = []
        np_y_list = []
        np_mask_list = []
        group_id_list = []
        raw_id_list = []
        for i, (id, gp) in enumerate(df_grp):
            gp = gp.sort_values(time_id)
            
            
            raw_id = gp.index.values

            np_gp = gp[use_cols].values
            #assert np_gp.shape[0] >=  self.max_seq
            

            np_gp = make_time_series(np_gp, windows_size=self.max_seq, pad_num=self.pad_num)
            #np_gp = np_gp[self.max_seq-1:]
        
            np_x = np_gp[..., :-(num_weight_col+num_label_col)]
            np_w = np_gp[..., -(num_weight_col+num_label_col):-num_label_col]
            np_y = np_gp[..., -num_label_col:]

            #print(np_gp.shape)
            

            #pad mask
            np_mask = np.all(np.equal(np_x, self.pad_num),axis=-1)
            
            np_x_list.append(np_x)
            np_mask_list.append(np_mask)
            
            group_id_list+= [i] * np_x.shape[0]

            if self.last_query_flag:
                np_y_list.append(np_y[:, -1, :]) #last
                np_w_list.append(np_w[:, -1, :]) #last
            else:
                #pdb.set_trace()
                np_y_list.append(np_y) 
                np_w_list.append(np_w)
                raw_id = make_time_series(raw_id.reshape(-1, 1), windows_size=self.max_seq, pad_num=-1)
                #pdb.set_trace()
                # = raw_id[self.max_seq-1:]
               

            raw_id_list.append(raw_id)



            #pdb.set_trace()

        self.np_x = np.concatenate(np_x_list)
        self.np_w = np.concatenate(np_w_list)
        self.np_y = np.concatenate(np_y_list)
        self.np_mask = np.concatenate(np_mask_list)
        self.group_id_list = group_id_list
        self.np_raw_id = np.concatenate(raw_id_list)

        # print(self.np_x.shape)
        # print(self.np_y.shape)
        # print(self.np_mask.shape)
        # print(f"len :{len(self.group_id_list)}")
        # pdb.set_trace()
        
        
        



    def __len__(self):
        return self.np_x.shape[0]

    def __getitem__(self, idx):
        
        #print(f"len :{len(self.group_id_list)}, {idx}")
   
        return [self.np_x[idx], self.np_mask[idx], self.group_id_list[idx], self.np_w[idx],self.np_raw_id[idx]], self.np_y[idx]



def collate_fn_Transformer(batch):
  

    batch_x_list, batch_x_conti_list, batch_row_id_list, batch_padding_list, batch_y = zip(*batch)

    new_x_list = []
    for x in zip(*batch_x_list):
        l_x = np.array(x)
        x_tensor = torch.tensor(l_x, dtype=torch.int64)
        new_x_list.append(x_tensor)

    for x in zip(*batch_x_conti_list):
        l_x = np.array(x)
        x_tensor = torch.tensor(l_x, dtype=torch.float32)
        new_x_list.append(x_tensor)

    #for x in zip(*batch_row_id_list):
    l_x = np.array(batch_row_id_list)
    x_tensor = torch.tensor(l_x, dtype=torch.int64)
    new_x_list.append(x_tensor)

    #for x in zip(*batch_padding_list):
    l_x = np.array(batch_padding_list)
    x_tensor = torch.tensor(l_x, dtype=torch.bool)
    new_x_list.append(x_tensor)


   
        
    new_batch_y = torch.tensor(np.concatenate(batch_y, 0).reshape(-1, 1))
    
    #import pdb; pdb.set_trace()



    return new_x_list, new_batch_y
