

from utils import *
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_utils import *




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
                            #A.HueSaturationValue(p=0.8),
                            #.ToGray(p=0.2),
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
        # if train_flag:
        #     comp_list = [
        #                     A.Resize(p=1.0, height=self.img_size, width=self.img_size),
        #                     #A.RandomResizedCrop(p=1.0, height=self.img_size, width=self.img_size, scale=(0.5, 1.0)),
        #                     A.HorizontalFlip(p=0.5),
        #                     A.VerticalFlip(p=0.5),
        #                     #A.CoarseDropout(),
        #                     #A.ImageCompression(),
        #                     #A.ISONoise(),
        #                     #A.MultiplicativeNoise(),
        #                     #A.HueSaturationValue(p=0.8),
        #                     #A.ToGray(p=0.2),
        #                     A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        #                     ToTensorV2(always_apply=True),
        #                     ]
        # else:
        #     comp_list = [A.Resize(p=1.0, height=self.img_size, width=self.img_size),
        #                  A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        #                 ToTensorV2(always_apply=True),
            
        #     ]
        # self.transformer = A.Compose(comp_list)

        self.distance_matrix = np.array([[0,1,4,9],[1,0,1,4],[4,1,0,1],[9,4,1,0]])
        self.label_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        #size = (224, 224)
        #size = (300, 300)
        size = (self.img_size, self.img_size)
        additional_items = (
            [T.Resize(size)]
            if not train_flag
            else [
                #T.RandomGrayscale(p=0.2),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                # T.ColorJitter(
                #     brightness=0.3,
                #     contrast=0.5,
                #     saturation=[0.8, 1.3],
                #     hue=[-0.05, 0.05],
                # ),
                T.RandomResizedCrop(size),
                T.Resize(size),
            ]
        )

        self.transformer = T.Compose(
            [*additional_items, T.ToTensor(), T.Normalize(mean=IMG_MEAN, std=IMG_STD)]
        )
        
    def __len__(self):
        return self.df_train_X.shape[0]

    def __getitem__(self, idx):
        
        image_name = self.df_train_X.iloc[idx]['image_name']
        ppath_to_img = INPUT_DIR/f"photos/{image_name}"
        img = Image.open(ppath_to_img)
        #img = np.array(img)
        
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

        img = self.transformer(img)
        #img = self.transformer(image=img)["image"]

        w = self.df_train_X.iloc[idx]['loss_weight']

        label = self.df_train_y.iloc[idx].values#[0]
        if not self.regression_flag:
            #print(f"label : {label}")
            w = self.distance_matrix[int(label), :]
            label = self.label_matrix[int(label), :]
        
        return [img, w],label

