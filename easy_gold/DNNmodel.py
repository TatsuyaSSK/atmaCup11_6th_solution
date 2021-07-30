
from utils import *
from image_utils import *

from typing import Optional, Dict, List, Callable, Union, Collection
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from abc import ABCMeta, abstractmethod
import pytorch_lightning as pl

from torchvision.models import resnet34, resnet18, vgg16
import torchvision.models as torch_models

import timm


from sklearn import random_projection
from DataSet import *
from Loss import *


#from torch_geometric.nn import GCNConv, AGNNConv, ChebConv, NNConv, DeepGCNLayer
#from torch_geometric.data import Data, Batch
#from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
#from torch_geometric.utils import to_networkx

#import networkx as nx

print(torch.__version__)

def set_seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



set_seed_torch(SEED_NUMBER)




class PytorchLightningModelBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.score_dict = {}
        self.best_score_dict = {}
        self.best_epoch = 1


    # @abstractmethod
    # def forward():

    #     pass

    def checkBest(self):

        score = self.score_dict["valid"][self.monitor]
        if ((self.mode=="max") and (self.last_score < score)) or ((self.mode=="min") and (self.last_score > score)):
            self.last_score = score
            self.best_score_dict = self.score_dict
            self.best_epoch = self.current_epoch
            print(f"renew best!! : {self.current_epoch}")


    def getScoreInfo(self):


        return self.best_score_dict, self.best_epoch, self.score_dict


    

    def criterion(self, y_true, y_pred, weight=None):

        #print(f"y_true : {y_true.shape}")
        #print(f"y_pred : {y_pred.shape}")
        #print(f"weight : {weight.shape}")
        #pdb.set_trace()

        if weight is None:
            return nn.MSELoss()(y_pred[:, :2].float(), y_true[:, :2].float())
        else:
            return weighted_mse_loss(input=y_pred[:, :2].float(), target=y_true[:, :2].float(), weight=weight)


        

    def setParams(self, _params):

        self.learning_rate = _params["learning_rate"]
        self.eval_metric_func_dict = _params["eval_metric_func_dict__"]
        self.monitor=_params["eval_metric"]
        self.mode=_params['eval_max_or_min']

        self.last_score = -10000000000 if self.mode == "max" else 10000000000

        print("show eval metrics : ")
        print(self.eval_metric_func_dict)

    def training_step(self, batch, batch_idx):
        
        out = self._forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])
        train_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self._forward(batch)

        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])
        val_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def training_epoch_end(self, outputs):
        
        
        score_dict = calcBatchMeanEvalScoreDictFromEvalScoreDictList(outputs, not_proc_cols=["loss"])
        score_dict["loss"] = torch.stack([o['loss'] for o in outputs]).mean().item()

        print_text = f"Epoch {self.current_epoch}, "
        for name, score in score_dict.items():
            self.log(f"train_{name}", score)
            print_text += f"{name} {score}, "
        print(print_text)

        self.score_dict = {}
        self.score_dict["train"] = score_dict

        #

        # if self._on_training_epoch_end is not None:
        #     for f in self._on_training_epoch_end:
        #         f(self.current_epoch, loss)

    def validation_epoch_end(self, outputs):

        score_dict = calcBatchMeanEvalScoreDictFromEvalScoreDictList(outputs, not_proc_cols=["val_loss"])
        score_dict["val_loss"] = torch.stack([o['val_loss'] for o in outputs]).mean().item()

        print_text = f"Epoch {self.current_epoch}, "
        for name, score in score_dict.items():
            self.log(f"val_{name}", score)
            print_text += f"{name} {score}, "
        print(print_text)

        self.score_dict["valid"] = score_dict
        self.checkBest()
        #loss = torch.stack([o['val_loss'] for o in outputs]).mean()

        # if self._on_validation_epoch_end is not None:
        #     for f in self._on_validation_epoch_end:
        #         f(self.current_epoch, loss)

    

    def test_step(self, batch, batch_idx):
        out = self._forward(batch)
        #pdb.set_trace()
        return {'out': out}

    def test_epoch_end(self, outputs):
        #pdb.set_trace()  
        self.final_preds = torch.cat([o['out'] for o in outputs]).data.cpu().detach().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, pretrained=False)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        print(self.encoder)
        #pdb.set_trace()

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class SSLsimSiam(PytorchLightningModelBase):

    def __init__(
        self, base_name: str, pretrained=False,
        in_channels: int=3, feat_dim: int=128
    ):

        """Initialize"""
        super(SSLsimSiam, self).__init__()


        model_names = sorted(name for name in torch_models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(torch_models.__dict__[name]))

        self.base_name = base_name
        self.model = SimSiam(torch_models.__dict__[base_name])

                                
                                


        # # # prepare backbone
        # if hasattr(timm.models, base_name):
        #     base_model = timm.create_model(
        #         base_name, num_classes=0, pretrained=pretrained, in_chans=in_channels)
        #     in_features = base_model.num_features
        #     print("load imagenet pretrained:", pretrained)
        # else:
        #     raise NotImplementedError

        # self.backbone = base_model
        # print(self.backbone)
        # print(f"{base_name}: {in_features}")

        

    def _forward(self, batch):
        
     
        img1 =  batch[0][0]
        img2 =  batch[0][1]
        p1, p2, z1, z2 = self.model(x1=img1, x2=img2)
        

        return p1, p2, z1, z2

    def criterion(self, p1, p2, z1, z2):

        criterion_func = nn.CosineSimilarity(dim=1)
        loss = -(criterion_func(p1, z2).mean() + criterion_func(p2, z1).mean()) * 0.5

        return loss

   


    def training_step(self, batch, batch_idx):
        
        p1, p2, z1, z2 = self._forward(batch)
        

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(p1, p2, z1, z2)


        train_batch_eval_score_dict= calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=loss.item(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        p1, p2, z1, z2 = self._forward(batch)
        
        loss = self.criterion(p1, p2, z1, z2)



        val_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=loss.item(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def _forward_test(self, batch):
        img1 =  batch[0][0]
        img2 =  batch[0][1]
        p1, p2, z1, z2 = self.model(x1=img1, x2=img2)

        #print(f"p1 : {p1.shape}")
        
        feat = p1.squeeze()
        #
        projection = random_projection.GaussianRandomProjection(n_components=1)
        out = projection.fit_transform(feat.cpu())
        out = torch.tensor(out, device=img1.device)
        #pdb.set_trace()

        return out

    def test_step(self, batch, batch_idx):
        out = self._forward_test(batch)
        #pdb.set_trace()
        return {'out': out}

class SupConModel(PytorchLightningModelBase):

    def __init__(
        self, base_name: str, pretrained=False,
        in_channels: int=3, feat_dim: int=128
    ):

        self.dummy_eval_num = 0

        """Initialize"""
        self.base_name = base_name
        super(SupConModel, self).__init__()

        # # prepare backbone
        if hasattr(timm.models, base_name):
            base_model = timm.create_model(
                base_name, num_classes=0, pretrained=pretrained, in_chans=in_channels)
            in_features = base_model.num_features
            print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError

        self.backbone = base_model
        print(self.backbone)
        print(f"{base_name}: {in_features}")
        #pdb.set_trace()

        if "vit" in base_name:
            self.backbone.patch_embed = ConvEmbed(in_chans=in_channels, embed_dim=384)
            self.backbone.blocks = self.backbone.blocks[:-1] 

        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/networks/resnet_big.py#L174
        self.head = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, feat_dim)
            )

    def _forward(self, batch):
        
     
        img1 =  batch[0][0]
        img2 =  batch[0][1]
        images = torch.cat([img1, img2], dim=0)
        
        feat = self.backbone(images)
        feat = F.normalize(self.head(feat), dim=1)

        return feat

    def _forward_test(self, batch):
        img1 =  batch[0][0]
        feat = self.backbone(img1)
        feat = feat.squeeze()
        #
        projection = random_projection.GaussianRandomProjection(n_components=1)
        out = projection.fit_transform(feat.cpu())
        out = torch.tensor(out, device=img1.device)
        #pdb.set_trace()

        return out

    def criterion(self, y_true, y_pred, weight=None):

        bsz = y_true.shape[0]

        

        f1, f2 = torch.split(y_pred, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        #pdb.set_trace()

        num_y = y_true.shape[1]

        total_loss = 0.0
        for i in range(num_y):
            loss = SupConLoss()(features, y_true[:, i])
            total_loss += loss

        return total_loss

    def training_step(self, batch, batch_idx):
        
        out = self._forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])

        def my_eval_dummy(y_pred, y_true):
            self.dummy_eval_num+=1
            return self.dummy_eval_num
        eval_metric_func_dict= {"eval_dummy":my_eval_dummy}

        train_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=eval_metric_func_dict)
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self._forward(batch)

        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])

        def my_eval_dummy(y_pred, y_true):
            self.dummy_eval_num+=1
            return self.dummy_eval_num
        eval_metric_func_dict= {"eval_dummy":my_eval_dummy}

        val_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=eval_metric_func_dict)
        


        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def test_step(self, batch, batch_idx):
        out = self._forward_test(batch)
        #pdb.set_trace()
        return {'out': out}

class ConvEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):#, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        # self.flatten = flatten

        self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//8, 3, stride=2, padding=1),
                nn.Conv2d(embed_dim//8, embed_dim//4, 3, stride=2, padding=1),
                nn.Conv2d(embed_dim//4, embed_dim//2, 3, stride=2, padding=1),
                nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1),
                nn.Conv2d(embed_dim, embed_dim, 1, stride=1, padding=0),
                
            )

        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        #B, C, H, W = x.shape
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        #if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #x = self.norm(x)
        return x
class myMultilabelNet(PytorchLightningModelBase):
    def __init__(self, base_name: str, pretrained=False,
                in_channels: int=3, num_out=1, 
                regression_flag=True, 
                tech_weight=None, material_weight=None) -> None:

        super().__init__()
        
        self.regression_flag=regression_flag
        self.num_out = num_out

        self.tech_weight = tech_weight
        self.material_weight = material_weight
        self.base_name = base_name

        # num_classes=128 if "vit" in base_name else 0
        # # # prepare backbone
        # if hasattr(timm.models, base_name):
        #     base_model = timm.create_model(
        #         base_name, num_classes=num_classes, pretrained=False, in_chans=in_channels)
        #     in_features = base_model.num_features
        #     print("load imagenet pretrained:", pretrained)
        # else:
        #     raise NotImplementedError

        # self.backbone = base_model
        

        # print(self.backbone)
        # if "vit" in base_name:
        #     self.backbone.patch_embed = ConvEmbed(in_chans=in_channels, embed_dim=768)
        #     self.backbone.blocks = self.backbone.blocks[:-1]

        #     dim_mlp = self.backbone.head.weight.shape[1]
        #     self.backbone.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.head)
        # #pdb.set_trace()

        # print(self.backbone)
        # print(f"{base_name}: {in_features}")

        # in_features=128 if "vit" in base_name else in_features
        # self.classifier = nn.Linear(in_features=in_features, out_features=num_out, bias=True)


        # self.classifier = nn.Sequential(
        #         nn.Linear(in_features, in_features),
        #         nn.Linear(in_features, in_features),
        #         #nn.ReLU(inplace=True),
        #         nn.Linear(in_features, num_out)
        #     )
        self.backbone = timm.create_model(base_name, pretrained=False)
        self.backbone.classifier = nn.Linear(in_features=1280, out_features=num_out, bias=True)

     
        #print(self.model)
        #pdb.set_trace()

    def loadBackbone(self, ppath_to_backbone_dir, fold_num):
        
        if not "vit" in self.base_name:
            prefix = f"fold_{fold_num}__iter_"
            name_list = list(ppath_to_backbone_dir.glob(f'model__{prefix}*.pkl'))
            if len(name_list)==0:
                print(f'[ERROR] Pretrained nn model was NOT EXITS! : {prefix}')
                return -1
            ppath_to_model = name_list[0]

            prefix=f"fold_{fold_num}"
            ppath_to_ckpt_model = searchCheckptFile(ppath_to_backbone_dir, ppath_to_model, prefix)
            tmp_dict = torch.load(str(ppath_to_ckpt_model))["state_dict"]
            print(tmp_dict.keys())
            backbone_dict = {k.replace("backbone.", ""):v for k, v in tmp_dict.items() if "backbone" in k }

        else:

            ppath_to_ckpt_model=PATH_TO_MODEL_DIR/"checkpoint_0499.pth.tar"
            #ppath_to_model= ppath_to_backbone_dir /PATH_TO_MODEL_DIR/"20210717-143003/model__fold_0__iter_98__20210717-143003__SSL_Wrapper.pkl"
            #tmp_dict = torch.load(str(ppath_to_model))
            tmp_dict = torch.load(str(ppath_to_ckpt_model))["state_dict"]
        
            backbone_dict = {k.replace("module.encoder_q.", ""):v for k, v in tmp_dict.items() if "module.encoder_q." in k }
        
        self.backbone.load_state_dict(backbone_dict)
        print(f"load backbone : {ppath_to_ckpt_model}")
        #pdb.set_trace()


    def _forward(self, batch):
        
     
        img =  batch[0][0]
        #print(f"img : {img.shape}")
        

        out = self.backbone(img)
        #(f"out : {out.shape}")
        #out = self.classifier(out)
        #print(f"out2 : {out.shape}")


        #out_target = self.fc_reg(out)
        #out_tech = self.fc_tech(out)
        #out_material = self.fc_material(out)

        #pdb.set_trace()
        #out = torch.cat([out_target.unsqueeze(1), out_tech, out_material], axis=-1)
        #out = torch.cat([out_target, out_tech, out_material], axis=-1)
        # out_target = out[:, 0]
        # out_tech = out[:, 1:]

        #

        return out

    def forward(self, batch):
        out = self._forward(batch)
        return out



    def criterion(self, y_true, y_pred, weight=None):

        #print(f"y_true : {y_true.shape}")
        #print(f"y_pred : {y_pred.shape}")
        #print(f"weight : {weight.shape}")

        if self.regression_flag:

            if weight is None:
                loss_target =  nn.MSELoss()(y_pred[:, 0].float(), y_true[:, 0].float())
            else:
                loss_target =  weighted_mse_loss(input=y_pred[:, 0].float(), target=y_true[:, 0].float(), weight=weight.float())
        else:
            #pdb.set_trace()
            #y_true = y_true.squeeze()
            #loss_target =  nn.CrossEntropyLoss()(y_pred, y_true.long())
            loss = - (1.0-y_true)* nn.LogSoftmax(dim=1)(1-y_pred)
            loss_target = (loss* weight).sum() #(dim=1).mean()
            
        
        tech_weight = None#torch.tensor(self.tech_weight,device=y_pred.device) if self.tech_weight is not None else None
        material_weight = None#torch.tensor(self.material_weight,device=y_pred.device)if self.material_weight is not None else None

        loss_tech=0
        if (len(y_true.shape) > 1) and (y_true.shape[1] > 1):
            num_tech = 3
            loss_tech = nn.BCEWithLogitsLoss(pos_weight =tech_weight)(y_pred[:, 1:num_tech+1].float(),  y_true[:, 1:num_tech+1].float())
            #loss_material = nn.BCEWithLogitsLoss(pos_weight =material_weight)(y_pred[:, num_tech+1:].float(),  y_true[:, num_tech+1:].float())

        return loss_target +  loss_tech #+ loss_material /3.0

    def training_step(self, batch, batch_idx):
        
        out = self.forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])


        if self.regression_flag:
            out_target = out[:, 0]
            out_tech = torch.sigmoid(out[:, 1:])
            #pdb.set_trace()
            out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)
        else:
            out = F.softmax(out, dim=1)

        

        train_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self.forward(batch)

        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])

        if self.regression_flag:
            out_target = out[:, 0]
            out_tech = torch.sigmoid(out[:, 1:])
            #pdb.set_trace()
            out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)
        else:
            out = F.softmax(out, dim=1)

        val_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)

        if self.regression_flag:
            out_target = out[:, 0]
            out_tech = torch.sigmoid(out[:, 1:])
            #pdb.set_trace()
            out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)

        else:
            out = F.softmax(out, dim=1)

        #pdb.set_trace()
        return {'out': out}


    # def test_epoch_end(self, outputs):
        
    #     self.final_preds = torch.cat([o['out'] for o in outputs]).data.cpu().detach().numpy().reshape(-1, 1)

    #     #pdb.set_trace()  


    def setParams(self, _params):

        self.learning_rate = _params["learning_rate"]
        self.eval_metric_func_dict = _params["eval_metric_func_dict__"]
        self.monitor=_params["eval_metric"]
        self.mode=_params['eval_max_or_min']

        self.last_score = -10000000000 if self.mode == "max" else 10000000000

        print("show eval metrics : ")
        print(self.eval_metric_func_dict)

        if _params["pretrain_model_dir_name"] is not None:
            ppath_to_backbone_dir = PATH_TO_MODEL_DIR/_params["pretrain_model_dir_name"]
            self.loadBackbone(ppath_to_backbone_dir, fold_num=_params["fold_n"])

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class myResNet(PytorchLightningModelBase):
    def __init__(self, base_name, num_out, regression_flag=True) -> None:
        super().__init__()
        
        self.regression_flag=regression_flag
        self.num_out = num_out

        model_names = sorted(name for name in torch_models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(torch_models.__dict__[name]))

        self.base_name = base_name
        self.backbone = torch_models.__dict__[base_name](pretrained=False)
        self.backbone.fc = nn.Linear(in_features=512, out_features=num_out, bias=True)
        for name, param in self.backbone.named_parameters():
           if name not in ['fc.weight', 'fc.bias']:
               param.requires_grad = False


        print(self.backbone)

        self.backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.backbone.fc.bias.data.zero_()
        
        #self.model = timm.create_model('efficientnet_b1_pruned', pretrained=False)
        #self.model.classifier = nn.Linear(in_features=1280, out_features=num_out, bias=True)
        #print(self.model)
        #pdb.set_trace()
       
        #self.model = timm.create_model('mobilenetv2_100', pretrained=False)
        #self.model.classifier = nn.Linear(in_features=1280, out_features=num_out, bias=True)

        
        #self.model = timm.create_model('seresnet152d', pretrained=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=num_out, bias=True)
        #print(self.model)
        #pdb.set_trace()

        #self.model = timm.create_model('gluon_senet154', pretrained=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=num_out, bias=True)
        #print(self.model)
        #pdb.set_trace()

        #self.model = timm.create_model('gluon_resnext101_32x4d', pretrained=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=num_out, bias=True)
        



        #self.model = vgg16(pretrained=False)
        #self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_out, bias=True)
        
        #self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
        #self.model.head = nn.Linear(in_features=1024, out_features=num_out, bias=True)
        #
        #

        #from pprint import pprint
        #model_names = timm.list_models(pretrained=True)
        #pprint(model_names)
        #sys.exit()
        

        #self.model = resnet18(pretrained=False)
        #self.model.fc = nn.Linear(in_features=512, out_features=num_out, bias=True)
        
    def _forward(self, batch):
        
     
        img =  batch[0][0]
        #print(f"img : {img.shape}")
        out = self.backbone(img)
        #out = torch.sigmoid(out)
        #out = torch.clamp(out, 0, 1)
        
        
        return out

    def forward(self, batch):
        out = self._forward(batch)
        return out

    
    def criterion(self, y_true, y_pred, weight=None):

        #print(f"y_true : {y_true.shape}")
        #print(f"y_pred : {y_pred.shape}")
        #print(f"weight : {weight.shape}")

        if self.regression_flag:

            if weight is None:
                loss_target =  nn.MSELoss()(y_pred[:, 0].float(), y_true[:, 0].float())
            else:
                loss_target =  weighted_mse_loss(input=y_pred[:, 0].float(), target=y_true[:, 0].float(), weight=weight.float())
        else:
            #pdb.set_trace()
            #y_true = y_true.squeeze()
            #loss_target =  nn.CrossEntropyLoss()(y_pred, y_true.long())
            loss = - (1.0-y_true)* nn.LogSoftmax(dim=1)(1-y_pred)
            loss_target = (loss* weight).sum() #(dim=1).mean()
            
        
        tech_weight = None#torch.tensor(self.tech_weight,device=y_pred.device) if self.tech_weight is not None else None
        material_weight = None#torch.tensor(self.material_weight,device=y_pred.device)if self.material_weight is not None else None

        loss_tech=0
        if (len(y_true.shape) > 1) and (y_true.shape[1] > 1):
            num_tech = 3
            loss_tech = nn.BCEWithLogitsLoss(pos_weight =tech_weight)(y_pred[:, 1:num_tech+1].float(),  y_true[:, 1:num_tech+1].float())
            loss_material = nn.BCEWithLogitsLoss(pos_weight =material_weight)(y_pred[:, num_tech+1:].float(),  y_true[:, num_tech+1:].float())

        return loss_target +  loss_tech + 0*loss_material ##/3.0

    # def criterion(self, y_true, y_pred, weight=None):

    #     # print(f"y_true : {y_true.shape}")
    #     # print(f"y_pred : {y_pred.shape}")
    #     # print(f"weight : {weight.shape}")
    #     #pdb.set_trace()

    #     if self.regression_flag:

    #         if weight is None:
    #             return nn.MSELoss()(y_pred.squeeze().float(), y_true.float())
    #         else:
    #             return weighted_mse_loss(input=y_pred.squeeze().float(), target=y_true.float(), weight=weight.float())
    #     else:
    #         return nn.CrossEntropyLoss()(y_pred, y_true.long())

    def training_step(self, batch, batch_idx):
        
        out = self.forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])


        if self.regression_flag:
            out_target = out[:, 0]
            out_tech = torch.sigmoid(out[:, 1:])
            #pdb.set_trace()
            out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)
        else:
            out = F.softmax(out, dim=1)

        

        train_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self.forward(batch)

        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])

        if self.regression_flag:
            out_target = out[:, 0]
            out_tech = torch.sigmoid(out[:, 1:])
            #pdb.set_trace()
            out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)
        else:
            out = F.softmax(out, dim=1)

        val_batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1].data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        


        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def test_step(self, batch, batch_idx):
        out = self._forward(batch)
        
        if self.regression_flag==False:
            #pdb.set_trace()
            out = torch.argmax(out, dim=1)
        return {'out': out}

    def test_epoch_end(self, outputs):
        
        self.final_preds = torch.cat([o['out'] for o in outputs]).data.cpu().detach().numpy()

        #pdb.set_trace()  

    def setParams(self, _params):

        self.learning_rate = _params["learning_rate"]
        self.eval_metric_func_dict = _params["eval_metric_func_dict__"]
        self.monitor=_params["eval_metric"]
        self.mode=_params['eval_max_or_min']

        self.last_score = -10000000000 if self.mode == "max" else 10000000000

        print("show eval metrics : ")
        print(self.eval_metric_func_dict)

        if _params["pretrain_model_dir_name"] is not None:
            ppath_to_backbone_dir = PATH_TO_MODEL_DIR/_params["pretrain_model_dir_name"]
            self.loadBackbone(ppath_to_backbone_dir, fold_num=_params["fold_n"])

    def loadBackbone(self, ppath_to_backbone_dir, fold_num):
        
        prefix = f"fold_{fold_num}__iter_"
        name_list = list(ppath_to_backbone_dir.glob(f'model__{prefix}*.pkl'))
        if len(name_list)==0:
            print(f'[ERROR] Pretrained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]

        prefix=f"fold_{fold_num}"
        ppath_to_ckpt_model = searchCheckptFile(ppath_to_backbone_dir, ppath_to_model, prefix)


        #ppath_to_ckpt_model=PATH_TO_MODEL_DIR/"checkpoint_0091.pth.tar"
        #ppath_to_model= ppath_to_backbone_dir /PATH_TO_MODEL_DIR/"20210717-143003/model__fold_0__iter_98__20210717-143003__SSL_Wrapper.pkl"
        #tmp_dict = torch.load(str(ppath_to_model))
        tmp_dict = torch.load(str(ppath_to_ckpt_model))["state_dict"]

        backbone_dict = {k.replace("model.encoder.", ""):v for k, v in tmp_dict.items() if "model.encoder." in k }
        #pdb.set_trace()
        
        self.backbone.load_state_dict(backbone_dict, strict=False)

        print(f"load backbone : {ppath_to_ckpt_model}")
        



