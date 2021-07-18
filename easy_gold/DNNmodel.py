
from utils import *

from typing import Optional, Dict, List, Callable, Union, Collection
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from abc import ABCMeta, abstractmethod
import pytorch_lightning as pl

from torchvision.models import resnet34, resnet18, vgg16

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




class PositionEncode(nn.Module):
    def __init__(self, dim):
        super(PositionEncode, self).__init__()
        # position = torch.zeros(length,dim)
        # print(f"position : {position}")
        
        # p = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        # print(f"p : {p}")
        # print(f"p : {p.shape}")
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        print(f"div : {div}")
        print(f"div : {div.shape}")

        # position[:,0::2] = torch.sin(p * div)
        # position[:,1::2] = torch.cos(p * div)
        # print(f"position : {position}")
        # print(f"position : {position.shape}")
        # #sys.exit()

        #position = position.reshape(length, 1, dim) 
        #self.register_buffer('position', position)
        self.register_buffer('div', div)


    def forward(self, x):
        length, batch_size, hidden_dim = x.shape
        position = torch.zeros(length,hidden_dim, device=x.device)
        p = torch.arange(0, length, dtype=torch.float, device=x.device).unsqueeze(1)
        position[:,0::2] = torch.sin(p * self.div)
        position[:,1::2] = torch.cos(p * self.div)
        position = position.reshape(length, 1, -1) 
        #print(f"position.shape : {position.shape}")

        position = position.repeat(1, batch_size, 1)
        #print(f"position.shape : {position.shape}")
        position = position[:length, :, :, ].contiguous()
        #print(f"position.shape : {position.shape}")
        return position



  

class EmbeddingDNN(nn.Module):
    def __init__(self, emb_dim_pairs_list, num_cont_features, emb_dropout):

        self.save_param_epoch_=0
        self.save_param_val_score_=0

        self.num_embs_D_features_ = sum([D for m, D in emb_dim_pairs_list])
        self.num_cont_features_ = num_cont_features

        super(EmbeddingDNN, self).__init__()
        
        self.emb_layers = nn.ModuleList([nn.Embedding(m, D) for m, D in emb_dim_pairs_list])
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

        #
        self.fc1 = nn.Linear(self.num_embs_D_features_ + self.num_cont_features_, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 1)
        self.dout1 = nn.Dropout(0.5)
        self.dout2 = nn.Dropout(0.5)
        self.dout3 = nn.Dropout(0.5)
        self.prelu = nn.PReLU(1)

    def forward(self, input_X_list):

        cat_data = input_X_list[0]
        cont_data = input_X_list[1]
        
        if self.num_embs_D_features_ != 0:

            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
            #x = self.bn1(x)
            x = torch.cat([x, cont_data], 1)
        else:
            x = cont_data
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dout1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dout2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dout3(x)
        x = self.bn3(x)
        # x = self.fc4(x)
        # x = self.prelu(x)
        #y = self.out(x)
        y = torch.sigmoid(self.out(x))

        return y

    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score
   
    
class MyDNNmodel(nn.Module):
    def __init__(self, init_x_num):

        self.save_param_epoch_=0
        self.save_param_val_score_=0

        super(MyDNNmodel, self).__init__()
        
        self.fc1 = nn.Linear(init_x_num, 256)

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        #self.fc4 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        self.dout1 = nn.Dropout(0.25)
        self.dout2 = nn.Dropout(0.35)
        self.dout3 = nn.Dropout(0.2)
        self.prelu = nn.PReLU(1)

    def forward(self, input_X_list):

        input_X = input_X_list[0]
        x = self.fc1(input_X)
        x = F.relu(x)
        x = self.dout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dout3(x)
        #x = self.fc4(x)
        #x = self.prelu(x)
        #y = self.out(x)
        y = torch.sigmoid(self.out(x))

        return y

    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score
        
 
class LastQueryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(query=src[-1,:,:].unsqueeze(0),  # last query
                              key=src,
                              value=src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # broadcast and add
        src = src + self.dropout1(src2)

        # remaining part is same as the normal transformer
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

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


        # # prepare backbone
        if hasattr(timm.models, base_name):
            base_model = timm.create_model(
                base_name, num_classes=0, pretrained=False, in_chans=in_channels)
            in_features = base_model.num_features
            print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError

        self.backbone = base_model
        print(self.backbone)
        print(f"{base_name}: {in_features}")

        
        

        self.classifier = nn.Linear(in_features=in_features, out_features=num_out, bias=True)

        #self.model = timm.create_model('efficientnet_b1', pretrained=False)
        #self.model.classifier = nn.Linear(in_features=in_features, out_features=num_out, bias=True)

     
        #print(self.model)
        #pdb.set_trace()

    def loadBackbone(self, ppath_to_backbone_dir, fold_num):
        
        prefix = f"fold_{fold_num}__iter_"
        name_list = list(ppath_to_backbone_dir.glob(f'model__{prefix}*.pkl'))
        if len(name_list)==0:
            print(f'[ERROR] Pretrained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]

        prefix=f"fold_{fold_num}"
        ppath_to_ckpt_model = searchCheckptFile(ppath_to_backbone_dir, ppath_to_model, prefix)

        #ppath_to_model=PATH_TO_MODEL_DIR/"20210717-103724/model__fold_0__iter_91__20210717-103724__SSL_Wrapper.pkl"
        #ppath_to_model= ppath_to_backbone_dir /PATH_TO_MODEL_DIR/"20210717-143003/model__fold_0__iter_98__20210717-143003__SSL_Wrapper.pkl"
        #tmp_dict = torch.load(str(ppath_to_model))
        tmp_dict = torch.load(str(ppath_to_ckpt_model))["state_dict"]
        backbone_dict = {k.replace("backbone.", ""):v for k, v in tmp_dict.items() if "backbone" in k }
        self.backbone.load_state_dict(backbone_dict)

        print(f"load backbone : {ppath_to_ckpt_model}")
        #pdb.set_trace()


    def _forward(self, batch):
        
     
        img =  batch[0][0]
        #print(f"img : {img.shape}")
        out = self.backbone(img)
        out = self.classifier(out)

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
        # print(f"weight : {weight.shape}")
        #pdb.set_trace()

        if self.regression_flag:

            if weight is None:
                loss_target =  nn.MSELoss()(y_pred[:, 0].float(), y_true[:, 0].float())
            else:
                loss_target =  weighted_mse_loss(input=y_pred[:, 0].float(), target=y_true[:, 0].float(), weight=weight.float())
        else:
            loss_target =  nn.CrossEntropyLoss()(y_pred[:, 0], y_true.long())
        
        tech_weight = None#torch.tensor(self.tech_weight,device=y_pred.device) if self.tech_weight is not None else None
        material_weight = None#torch.tensor(self.material_weight,device=y_pred.device)if self.material_weight is not None else None

        num_tech = 3
        loss_tech = nn.BCEWithLogitsLoss(pos_weight =tech_weight)(y_pred[:, 1:num_tech+1].float(),  y_true[:, 1:num_tech+1].float())
        #loss_material = nn.BCEWithLogitsLoss(pos_weight =material_weight)(y_pred[:, num_tech+1:].float(),  y_true[:, num_tech+1:].float())

        return loss_target +  loss_tech #+ loss_material /3.0

    def training_step(self, batch, batch_idx):
        
        out = self.forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])

        out_target = out[:, 0]
        out_tech = torch.sigmoid(out[:, 1:])
        #pdb.set_trace()
        out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)

        

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

        out_target = out[:, 0]
        out_tech = torch.sigmoid(out[:, 1:])
        #pdb.set_trace()
        out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)

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
        out_target = out[:, 0]
        out_tech = torch.sigmoid(out[:, 1:])
        #pdb.set_trace()
        out = torch.cat([out_target.unsqueeze(1), out_tech], axis=-1)

        #pdb.set_trace()
        return {'out': out}


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

class myResNet(PytorchLightningModelBase):
    def __init__(self, num_out, regression_flag=True) -> None:
        super().__init__()
        
        self.regression_flag=regression_flag
        self.num_out = num_out

        #self.model = timm.create_model('xception', pretrained=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=num_out, bias=True)
        #self.model.classifier = nn.Linear(in_features=1408, out_features=num_out, bias=True)
        #
        #
        #

        self.model = timm.create_model('efficientnet_b1_pruned', pretrained=False)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_out, bias=True)
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
        out = self.model(img)
        #out = torch.sigmoid(out)
        #out = torch.clamp(out, 0, 1)
        
        
        return out

    def forward(self, batch):
        out = self._forward(batch)
        return out

    def criterion(self, y_true, y_pred, weight=None):

        # print(f"y_true : {y_true.shape}")
        # print(f"y_pred : {y_pred.shape}")
        # print(f"weight : {weight.shape}")
        #pdb.set_trace()

        if self.regression_flag:

            if weight is None:
                return nn.MSELoss()(y_pred.squeeze().float(), y_true.float())
            else:
                return weighted_mse_loss(input=y_pred.squeeze().float(), target=y_true.float(), weight=weight.float())
        else:
            return nn.CrossEntropyLoss()(y_pred, y_true.long())

    def test_step(self, batch, batch_idx):
        out = self._forward(batch)
        
        if self.regression_flag==False:
            #pdb.set_trace()
            out = torch.argmax(out, dim=1)
        return {'out': out}

    def test_epoch_end(self, outputs):
        
        self.final_preds = torch.cat([o['out'] for o in outputs]).data.cpu().detach().numpy().reshape(-1, 1)

        #pdb.set_trace()  


class LastLSTM(PytorchLightningModelBase):
    def __init__(self,
                 n_numerical_features: int,
                 emb_dim_pairs_list,
                 prev_renew_col_idx_dict,
                 ##num_bssid_features:int,
                 ##num_fq_features:int,
                 last_query_flag:bool,
                 d_model: int = 512,#256,

                 #dropout: float = 0.1,
                 dropout_emb: float = 0.1,

                 lstm_hidden_dim:int = 128*2,
                 num_lstm_layers:int = 2,
                 dropout_lstm:float = 0.1,
                 
                ) -> None:
        super().__init__()

        self.last_query_flag = last_query_flag
        self.prev_renew_col_idx_dict = prev_renew_col_idx_dict

        self.prev_idx_for_renew_batch = -1

        self.n_numerical_features = n_numerical_features
        ##self.num_bssid_features=num_bssid_features
        ##self.num_fq_features=num_fq_features
        
        self.d_model = d_model


        self.emb_layers = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_dim_pairs_list])
        self.total_cat_emb_dim = sum([d for m, d in emb_dim_pairs_list])
        self.categorical_proj = nn.Sequential(
            nn.Linear(self.total_cat_emb_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

 
        #self.bssid_emb_layer = nn.Embedding(54454+1, self.d_model)
        ##self.bssid_emb_layer = nn.Embedding(54468+1, self.d_model)
        
        

        # self.total_bssid_emb_dim = self.num_bssid_features * self.d_model
        # self.bssid_proj = nn.Sequential(
        #     nn.Linear(self.total_bssid_emb_dim, self.d_model),
        #     nn.LayerNorm(self.d_model),
        # )

        # self.fq_emb_layer = nn.Embedding(27+1, self.d_model)
        # self.total_fq_emb_dim = self.num_fq_features * self.d_model
        # self.fq_proj = nn.Sequential(
        #     nn.Linear(self.total_fq_emb_dim, self.d_model),
        #     nn.LayerNorm(self.d_model),
        # )
        

        self.conti_embed_layers = nn.ModuleList([nn.Linear(1,self.d_model,bias=False) for n in range(n_numerical_features)])
        self.continuous_proj = nn.Sequential(
            nn.Linear(self.d_model*n_numerical_features, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.emb_src_dim = self.d_model * 2##* 4

        self.dropout_emb = nn.Dropout(dropout_emb)
        self.layer_normal = nn.LayerNorm(self.emb_src_dim)

        # self.total_proj = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.emb_src_dim, self.emb_src_dim),
        # )


        self.lstm = nn.LSTM(
            input_size=self.emb_src_dim,
            bidirectional=True,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout_lstm)

        last_dim = 32
        self.decoder_xy = nn.Sequential(
            nn.Linear(lstm_hidden_dim*2, last_dim), #for bidirectional
            #nn.Linear(lstm_hidden_dim, last_dim),
            #nn.ReLU(True),
            nn.Linear(last_dim, 2),
        )
    
        

    def init_weights(self):
        initrange = 0.1

        for i, l in enumerate(self.emb_layers):
            l.weight.data.uniform_(-initrange, initrange)

        for i, l in enumerate(self.conti_embed_layers):
            l.weight.data.uniform_(-initrange, initrange)

        ##self.bssid_emb_layer.weight.data.uniform_(-initrange, initrange)
        ##self.fq_emb_layer.weight.data.uniform_(-initrange, initrange)

    

        # def weights_init(m):
        #     classname = m.__class__.__name__
        #     if classname.find('Conv') != -1:
        #         torch.nn.init.normal_(m.weight, 0.0, 0.02)
        #     elif classname.find('BatchNorm') != -1:
        #         torch.nn.init.normal_(m.weight, 1.0, 0.02)
        #         torch.nn.init.zeros_(m.bias)


    def _forward(self, batch):

        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        '''
        #
        src = batch[0][0]


        add_flag=False

        if add_flag:
            emb_sum = None
            for i, emb_layer in enumerate(self.emb_layers):
                emb = emb_layer(src[...,i])
                if i == 0:
                    emb_sum = emb
                else:
                    emb_sum += emb

            num_s = len(self.emb_layers) 
            for i, emb_layer in enumerate(self.conti_embed_layers):
                #import pdb; pdb.set_trace()
                emb_sum += emb_layer(src[...,i+num_s].unsqueeze(-1))
            
            embedded_src = emb_sum.transpose(0, 1) # (S, N, E)
        else:

            

            emb_list = [emb_layer(src[...,i].long()) for i, emb_layer in enumerate(self.emb_layers)]
            emb_concat = torch.cat(emb_list, axis=-1)
            #print(f"emb_concat : {emb_concat.shape}")
            emb_concat = self.categorical_proj(emb_concat)
            #print(f"emb_concat proj: {emb_concat.shape}")

            num_s = len(self.emb_layers)
            #print(f"num_s: {num_s}")
           
            # bssid_emb = [self.bssid_emb_layer(src[..., i+num_s].long()) for i in range(self.num_bssid_features)]
            # bssid_emb_concat = torch.cat(bssid_emb, axis=-1)
            # bssid_emb_concat = self.bssid_proj(bssid_emb_concat) 

            # num_s += self.num_bssid_features
            # #print(f"num_s: {num_s}")
            
            # fq_emb = [self.fq_emb_layer(src[..., i+num_s].long()) for i in range(self.num_fq_features)]
            # fq_emb_concat = torch.cat(fq_emb, axis=-1)
            # fq_emb_concat = self.fq_proj(fq_emb_concat) 

            

            ##num_s += self.num_fq_features            
            conti_emb_list = [emb_layer(src[...,i+num_s].unsqueeze(-1).float()) for i, emb_layer in enumerate(self.conti_embed_layers)]
            #print(f"conti_emb_list : {conti_emb_list}")
            #print(f"conti_emb_list : {conti_emb_list[0].shape}")
            conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
            #print(f"conti_emb_concat : {conti_emb_concat.shape}")

            conti_emb_concat = self.continuous_proj(conti_emb_concat)
            #print(f"conti_emb_concat proj : {conti_emb_concat.shape}")


            ##embedded_src = torch.cat([emb_concat, bssid_emb_concat, fq_emb_concat, conti_emb_concat], axis=-1)
            embedded_src = torch.cat([emb_concat,  conti_emb_concat], axis=-1)
            #embedded_src = emb_concat

            #print(f"embedded_src : {embedded_src.shape}")
            

            embedded_src=embedded_src.transpose(0, 1)

            #import pdb; pdb.set_trace()



        #input_emb = embedded_src * np.sqrt(self.ninp)

        input_emb = self.dropout_emb(embedded_src)
        input_emb = self.layer_normal(input_emb)



        #input_emb = self.total_proj(embedded_src)
        #input_emb = embedded_src

        #

        input_emb,_ = self.lstm(input_emb)

        #embedded_src += self.postition(embedded_src)

        #
        if self.last_query_flag:
            output_last = input_emb[-1, ...]
        else:
            output_last = input_emb
            output_last=output_last.transpose(0, 1)
        #pdb.set_trace()

      
        output_xy = self.decoder_xy(output_last)

        # if self.last_query_flag:
        #     output_all = torch.cat([output_xy, batch[0][4].view(-1, 1)], axis=-1)
        # else:
        #     output_all = torch.cat([output_xy, batch[0][4]], axis=-1)
       

        #pdb.set_trace()
        #output_all = torch.cat([batch[0][4].view(-1, 1), batch[0][4].view(-1, 1), batch[0][4].view(-1, 1)], axis=-1)

        
        
        return output_xy

    def forward(self, batch):
        out = self._forward(batch)
        return out

    def criterion(self, y_true, y_pred, weight=None):

        #print(f"y_true : {y_true.shape}")
        #print(f"y_pred : {y_pred.shape}")
        #print(f"weight : {weight.shape}")
        #pdb.set_trace()

        if weight is None:
            return nn.MSELoss()(y_pred[:, :2].float(), y_true[:, :2].float())
        else:
            return weighted_mse_loss(input=y_pred[:, :2].float(), target=y_true[:, :2].float(), weight=weight)

    def training_step(self, batch, batch_idx):
        
        out = self._forward(batch)

        
        if self.last_query_flag==False:
            out = out[~batch[0][1]]
            y_true = batch[-1][~batch[0][1]]
            weight = batch[0][3][~batch[0][1]]
            #weight = y_true[..., -1].view(-1, 1)
            
        else:
            y_true = batch[-1]
            weight = batch[0][3]

        loss = self.criterion(y_pred=out, y_true=y_true, weight=weight)
        #pdb.set_trace()
        train_batch_eval_score_dict=calcEvalScoreDict(y_true=y_true.data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        #pdb.set_trace()
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self._forward(batch)
        if self.last_query_flag==False:
            out = out[~batch[0][1]]
            y_true = batch[-1][~batch[0][1]]
            weight = batch[0][3][~batch[0][1]]
            #weight = y_true[..., -1].view(-1, 1)
        else:
            y_true = batch[-1]
            weight = batch[0][3]
            #weight = batch[-1][..., -1].view(-1, 1)

        loss = self.criterion(y_pred=out, y_true=y_true, weight=weight)
        #pdb.set_trace()
        val_batch_eval_score_dict=calcEvalScoreDict(y_true=y_true.data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)



        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret
        
    def register_batch(self, batch, prev_out, prev_mask):

        
        

        di = self.prev_renew_col_idx_dict
        

        if self.last_query_flag:
            batch[0][0][:, -1, di["prev_x"]] = prev_out[:, 0].item()
            batch[0][0][:, -1, di["prev_y"]] = prev_out[:, 1].item()

            batch[0][0][:, -1, di["next_x"]] = batch[0][0][:, -1, di["prev_x"]] + batch[0][0][:, -1, di["cum_rel_x"]]
            batch[0][0][:, -1, di["next_y"]] = batch[0][0][:, -1, di["prev_y"]] + batch[0][0][:, -1, di["cum_rel_y"]]
        
        else:
            masked_cur_batch = batch[0][0][:,(~prev_mask).view(-1),:]
            masked_prev_out = prev_out[:,(~prev_mask).view(-1),:]

            batch[0][0][:,(~prev_mask).view(-1),di["prev_x"]]  = masked_prev_out[:,:,0].double()
            batch[0][0][:,(~prev_mask).view(-1),di["prev_y"]]  = masked_prev_out[:,:,1].double()
            
            batch[0][0][:,(~prev_mask).view(-1),di["next_x"]] = batch[0][0][:,(~prev_mask).view(-1),di["prev_x"]] + batch[0][0][:,(~prev_mask).view(-1),di["cum_rel_x"]]
            batch[0][0][:,(~prev_mask).view(-1),di["next_y"]] = batch[0][0][:,(~prev_mask).view(-1),di["prev_y"]] + batch[0][0][:,(~prev_mask).view(-1),di["cum_rel_y"]]

        #pdb.set_trace()

        # for i, idx in enumerate(renew_cols_idx_list):
        #     if self.last_query_flag:
        #         #print(batch[0][0][:, -1, idx])
        #         #print(prev_out[:, i])
        #         batch[0][0][:, -1, idx] = prev_out[:, i].item()
        #         #print(batch[0][0][:, -1, idx])
        #         #pdb.set_trace()
        #     else:
                
        #         # print(batch[0][0][:,:,idx])
        #         # print(prev_out[:,:,i])
        #         # print((~prev_mask))

        #         masked_cur_batch = batch[0][0][:,(~prev_mask).view(-1),:]
        #         masked_prev_out = prev_out[:,(~prev_mask).view(-1),:]
                
        #         masked_cur_batch[:,-1,idx] = masked_prev_out[:,-1,i].double()
        #         masked_cur_batch[:,:-1,idx] +=masked_prev_out[:, :-1, i][:,:, None].double()
        #         masked_cur_batch[:,:-1,idx] = masked_cur_batch[:,:-1,idx]/2
        #         #pdb.set_trace()
        #         batch[0][0][:,(~prev_mask).view(-1),idx] = masked_cur_batch[:,:,idx].view(-1)
                
        #         #print(batch[0][0][:,:,idx])
        #         #pdb.set_trace()

                


        
        return batch

    def renew_batch(self, batch):

        current_id = batch[0][2].item()


        if self.prev_idx_for_renew_batch == current_id:
            #self.seq_num_in_batch += 1
            
            batch[0][0][:, :-1,:] = self.prev_batch[0][0][:,1:,:]
            batch = self.register_batch(batch, self.prev_out, prev_mask=self.prev_batch[0][1])
            #pdb.set_trace()

        self.prev_idx_for_renew_batch = current_id
        #self.seq_num_in_batch = 0

        return batch
            


    def test_step(self, batch, batch_idx):

        #print(batch)
        if self.oof_prediction==False:
            batch = self.renew_batch(batch)

        out = self._forward(batch)
        
        if self.oof_prediction==False:
            self.prev_batch = batch #self.register_batch(batch, out)
            self.prev_out = out #self.register_batch(batch, out)

        
        sort_idx = batch[0][4]
        if self.last_query_flag==False:
            out =  out[~batch[0][1]]
            sort_idx = sort_idx[~batch[0][1]]
        
        return {'out': out, "sort_idx":sort_idx}
        
    def test_epoch_end(self, outputs):
        #
        sort_idx = torch.cat([o['sort_idx'] for o in outputs])
        pred = torch.cat([o['out'] for o in outputs])
        if self.last_query_flag==False:
            sort_idx = sort_idx.data.cpu().detach().numpy()
            pred = pred.data.cpu().detach().numpy()
            df = pd.DataFrame(pred)
            df["idx"] = sort_idx
            gp = df.groupby("idx").mean()
            self.final_preds = gp.values

        else:
            self.final_preds = pred[sort_idx].data.cpu().detach().numpy()
       



class SimpleTransformer(PytorchLightningModelBase):
    def __init__(self,
                 n_numerical_features: int,
                 emb_dim_pairs_list,
                 num_bssid_features:int,
                 num_fq_features:int,
                 last_query_flag:bool,
                

                 num_dec_features:int = 2,
                 d_model: int = 32*4,
                 nhead: int = 1,
                 num_encoder_layers: int = 1,
                 num_decoder_layers:int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 dropout_emb: float = 0.0,
                 activation: str = "relu",

                 lstm_hidden_dim:int = 128,
                 num_lstm_layers:int = 1,
                 dropout_lstm:float = 0.1,
                 
                 norm_encoder_output: bool = False,
                 
                 on_training_epoch_end: Optional[List[Callable]] = None,
                 on_validation_epoch_end: Optional[List[Callable]] = None,
                 subtract: bool = False) -> None:
        super().__init__()

        self.last_query_flag = last_query_flag

        self.prev_idx_for_renew_batch = -1


        self.n_numerical_features = n_numerical_features
        self.num_bssid_features=num_bssid_features
        self.num_fq_features=num_fq_features
        self.num_dec_features = num_dec_features

        
        self.d_model = d_model


        self.emb_layers = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_dim_pairs_list])
        self.total_cat_emb_dim = sum([d for m, d in emb_dim_pairs_list])
        self.categorical_proj = nn.Sequential(
            #nn.Linear(self.total_cat_emb_dim, self.d_model),
            #nn.LayerNorm(self.d_model),
            nn.Conv1d(in_channels=self.total_cat_emb_dim, out_channels=self.d_model, kernel_size=1),

        )

        self.bssid_emb_layer = nn.Embedding(54468+1, self.d_model)
        
        self.total_bssid_emb_dim = self.num_bssid_features * self.d_model
        self.bssid_proj = nn.Sequential(
            nn.Conv1d(in_channels=self.total_bssid_emb_dim, out_channels=self.d_model, kernel_size=1),
            #nn.Linear(self.total_bssid_emb_dim, self.d_model),
            #nn.LayerNorm(self.d_model),
        )

        self.fq_emb_layer = nn.Embedding(27+1, self.d_model)
        self.total_fq_emb_dim = self.num_fq_features * self.d_model
        self.fq_proj = nn.Sequential(
            nn.Conv1d(in_channels=self.total_fq_emb_dim, out_channels=self.d_model, kernel_size=1),
            #nn.Linear(self.total_fq_emb_dim, self.d_model),
            #nn.LayerNorm(self.d_model),
        )

        self.conti_embed_layers = nn.ModuleList([nn.Linear(1,self.d_model,bias=False) for n in range(self.n_numerical_features)])
        self.continuous_proj = nn.Sequential(
            nn.Conv1d(in_channels=self.d_model*(self.n_numerical_features), out_channels=self.d_model, kernel_size=1),
            #nn.Linear(self.d_model*(self.n_numerical_features-self.num_dec_features), self.d_model),
            #nn.Linear(self.d_model*(self.n_numerical_features), self.d_model),
            #nn.LayerNorm(self.d_model),
        )

        self.final_d_model = self.d_model * 4

        self.dec_embed_layers = nn.ModuleList([nn.Linear(1,self.d_model,bias=False) for n in range(self.num_dec_features)])
        self.dec_proj = nn.Sequential(
            nn.Conv1d(in_channels=self.d_model*(self.num_dec_features), out_channels=self.final_d_model, kernel_size=1),
            #nn.Linear(self.d_model*(self.num_dec_features), self.final_d_model),
            #nn.LayerNorm(self.final_d_model),
        )


        self.postition = PositionEncode(self.final_d_model)

        self.dropout_emb = nn.Dropout(dropout_emb)
        self.layer_normal = nn.LayerNorm(self.final_d_model)
        self.layer_normal_tgt = nn.LayerNorm(self.final_d_model)

        #self.conv = nn.Conv1d(in_channels=self.d_emb, out_channels=self.d_model, kernel_size=1)

        # self.lstm = nn.LSTM(
        #     input_size=self.emb_src_dim,
        #     bidirectional=False,
        #     hidden_size=lstm_hidden_dim,
        #     num_layers=num_lstm_layers,
        #     dropout=dropout_lstm)

        # self.emb_src_dim = lstm_hidden_dim
        encoder_layer = LastQueryTransformerEncoderLayer(self.final_d_model, nhead, dim_feedforward, dropout)
        # norm = nn.LayerNorm(self.emb_src_dim) if norm_encoder_output else None
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm)


        #encoder_layer = nn.TransformerEncoderLayer(self.final_d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(self.final_d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        decoder_layer = nn.TransformerDecoderLayer(self.final_d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(self.final_d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        #self.out = nn.Linear(512, target_vocab_length)


        self.decoder_xy = nn.Sequential(
            nn.Linear(self.final_d_model, self.d_model),
            #nn.ReLU(True),
            nn.Linear(self.d_model, 2),
        )


        #self.decoder_floor = nn.Linear(self.emb_src_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        for i, l in enumerate(self.emb_layers):
            l.weight.data.uniform_(-initrange, initrange)

        for i, l in enumerate(self.conti_embed_layers):
            l.weight.data.uniform_(-initrange, initrange)

        for i, l in enumerate(self.dec_embed_layers):
            l.weight.data.uniform_(-initrange, initrange)

        self.bssid_emb_layer.weight.data.uniform_(-initrange, initrange)
        self.fq_emb_layer.weight.data.uniform_(-initrange, initrange)

        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #mask = mask.float().masked_fill(mask == 0, float('-1e8')).masked_fill(mask == 1, float(0.0))
        return mask

    def _forward(self, batch):

        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        src_key_padding_mask is (N, S) with boolean True/False. #padにmaskするやつ
        src_mask is (S, S) with float(’-inf’) and float(0.0). #未来の情報にmaskするやつ


        src: (S, N, E)

        tgt: (T, N, E)

        src_mask: (S, S)

        tgt_mask: (T, T)

        memory_mask: (T, S)

        src_key_padding_mask: (N, S)

        tgt_key_padding_mask: (N, T)

        memory_key_padding_mask: (N, S)
        '''
        #
        src = batch[0][0]
        # print( self.emb_layers[0].weight.data)
        # print( self.decoder_xy[0].weight)
        # print( self.decoder_xy[1].weight)
        # pdb.set_trace()

        add_flag=False

        if add_flag:
            emb_sum = None
            for i, emb_layer in enumerate(self.emb_layers):
                emb = emb_layer(src[...,i])
                if i == 0:
                    emb_sum = emb
                else:
                    emb_sum += emb

            num_s = len(self.emb_layers) 
            for i, emb_layer in enumerate(self.conti_embed_layers):
                #import pdb; pdb.set_trace()
                emb_sum += emb_layer(src[...,i+num_s].unsqueeze(-1))
            
            embedded_src = emb_sum.transpose(0, 1) # (S, N, E)
        else:

           

            emb_list = [emb_layer(src[...,i].long()) for i, emb_layer in enumerate(self.emb_layers)]
            emb_concat = torch.cat(emb_list, axis=-1)
            emb_concat = emb_concat.permute([0, 2, 1])
            
            #print(f"emb_concat : {emb_concat.shape}")
            emb_concat = self.categorical_proj(emb_concat)
            #print(f"emb_concat proj: {emb_concat.shape}")

            num_s = len(self.emb_layers)
            bssid_emb = [self.bssid_emb_layer(src[..., i+num_s].long()) for i in range(self.num_bssid_features)]
            bssid_emb_concat = torch.cat(bssid_emb, axis=-1)
            bssid_emb_concat = bssid_emb_concat.permute([0, 2, 1])

            bssid_emb_concat = self.bssid_proj(bssid_emb_concat) 

            num_s += self.num_bssid_features
            #print(f"num_s: {num_s}")
            
            fq_emb = [self.fq_emb_layer(src[..., i+num_s].long()) for i in range(self.num_fq_features)]
            fq_emb_concat = torch.cat(fq_emb, axis=-1)
            fq_emb_concat = fq_emb_concat.permute([0, 2, 1])

            fq_emb_concat = self.fq_proj(fq_emb_concat) 

            num_s += self.num_fq_features
            decode_cols_idx_list = [i[0].item() for i in batch[0][3]]

            # conti_emb_list=[]
            # for i, emb_layer in enumerate(self.conti_embed_layers):
            #     idx = i+num_s
            #     print(f"i:{i}, idx:{idx}, num_s:{num_s}")

            #     if idx in decode_cols_idx_list:
            #         print("skip")
            #     else:
            #         conti_emb_list.append(emb_layer(src[...,idx].unsqueeze(-1).float()))
                    


     
            conti_emb_list = [emb_layer(src[...,i+num_s].unsqueeze(-1).float()) for i, emb_layer in enumerate(self.conti_embed_layers) if  (i+num_s) not in decode_cols_idx_list ]
            #conti_emb_list = [emb_layer(src[...,i+num_s].unsqueeze(-1).float()) for i, emb_layer in enumerate(self.conti_embed_layers)]
            #dec_emb_list = [self.dec_embed_layers[i](src[...,idx].unsqueeze(-1).float()) for i, idx in enumerate(decode_cols_idx_list)]
            dec_emb_list=[]
            #for i, idx in enumerate(decode_cols_idx_list):
                



            #TODO: delete this!
            #dec_emb_list_tmp = [t/t  for t in dec_emb_list]
            #dec_emb_list = dec_emb_list_tmp

            dec_emb_concat = torch.cat(dec_emb_list, axis=-1)

            conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
            conti_emb_concat = conti_emb_concat.permute([0, 2, 1])

            # print(f"conti_emb_concat before: {conti_emb_concat.shape}")
            # conti_emb_concat = conti_emb_concat.permute([0, 2, 1])
            # print(f"conti_emb_concat after: {conti_emb_concat.shape}")

            #dec_emb_concat = self.dec_proj(dec_emb_concat)
            #pdb.set_trace()
            conti_emb_concat = self.continuous_proj(conti_emb_concat)
            #print(f"conti_emb_concat proj : {conti_emb_concat.shape}")


            #embedded_src = torch.cat([emb_concat, bssid_emb_concat, fq_emb_concat, conti_emb_concat], axis=-1)
            embedded_src = torch.cat([emb_concat, bssid_emb_concat, fq_emb_concat, conti_emb_concat], axis=1)
           

            #print(f"embedded_src : {embedded_src.shape}")
            embedded_src=embedded_src.permute([2, 0, 1])

            #embedded_src=embedded_src.transpose(0, 1)
            #embedded_tgt=dec_emb_concat.transpose(0, 1)

            #pdb.set_trace()

        #print(f"embedded_src : {embedded_src.shape}")
        #


        #input_emb = embedded_src * np.sqrt(self.ninp)

        
        input_emb = self.dropout_emb(embedded_src)
        input_emb = self.layer_normal(input_emb)

        #input_tgt_emb = self.dropout_emb(embedded_tgt)
        #input_tgt_emb = self.layer_normal_tgt(input_tgt_emb)

        #

        #input_emb,_ = self.lstm(input_emb)

        #embedded_src += self.postition(embedded_src)

        # row_id = src[-2]
        src_key_padding_mask = batch[0][1]

        

        src_mask = None#self.generate_square_subsequent_mask(input_emb.size(0)).cuda()
        #output = self.transformer_encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        #input_emb=input_emb.transpose(0, 1)
        memory = self.encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #print(memory)
        #pdb.set_trace()
        tgt_mask=None #self.generate_square_subsequent_mask(input_tgt_emb.size(0)).cuda()
        memory_mask=self.generate_square_subsequent_mask(memory.size(0)).cuda()
        #pdb.set_trace()

        #pdb.set_trace()
        #src_key_padding_mask=None
        #output = self.decoder(tgt=input_tgt_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                      tgt_key_padding_mask=src_key_padding_mask,
        #                      memory_key_padding_mask=src_key_padding_mask)
        
        #output = self.out(output)
        output = memory
        #print(f"out enc: {output}")
        #print(f"out enc: {output.shape}")

        #pdb.set_trace()
        
        #output_xy = self.reg_out_xy(src[2], output)
        #output_floor = self.reg_out_floor(src[2], output)
        
        if self.last_query_flag:
            output_last = output[-1, ...]
        else:
            output_last = output
            output_last=output_last.transpose(0, 1)
        
        
        output_xy = self.decoder_xy(output_last)
        #pdb.set_trace()
        #output_floor = self.decoder_floor(output_last)
        #print(f"output_xy dec: {output_xy.shape}")
        #print(f"output_floor dec: {output_floor.shape}")
        if self.last_query_flag:
            output_all = torch.cat([output_xy, batch[0][4].view(-1, 1)], axis=-1)
        else:
            output_all = torch.cat([output_xy, batch[0][4]], axis=-1)
       


        #
        
        return output_all

    def forward(self, batch):
        out = self._forward(batch)
        return out
        
    def training_step(self, batch, batch_idx):
        
        out = self._forward(batch)

        #loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)
        if self.last_query_flag==False:
            out = out[~batch[0][1]]
            y_true = batch[-1][~batch[0][1]]
            weight = y_true[..., -1].view(-1, 1)
        else:
            y_true = batch[-1]
            weight = batch[-1][..., -1].view(-1, 1)

        loss = self.criterion(y_pred=out, y_true=y_true, weight=weight)
        #pdb.set_trace()
        train_batch_eval_score_dict=calcEvalScoreDict(y_true=y_true.data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)
        #pdb.set_trace()
        


        
        self.log('loss', loss, logger=False)
        ret = {'loss': loss}

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):

        out = self._forward(batch)
        if self.last_query_flag==False:
            out = out[~batch[0][1]]
            y_true = batch[-1][~batch[0][1]]
            weight = y_true[..., -1].view(-1, 1)
        else:
            y_true = batch[-1]
            weight = batch[-1][..., -1].view(-1, 1)

        loss = self.criterion(y_pred=out, y_true=y_true, weight=weight)
        #pdb.set_trace()
        val_batch_eval_score_dict=calcEvalScoreDict(y_true=y_true.data.cpu().detach().numpy(), y_pred=out.data.cpu().detach().numpy(), eval_metric_func_dict=self.eval_metric_func_dict)



        self.log('val_loss', loss, logger=False)
        ret = {'val_loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret
        
    def register_batch(self, batch, prev_out, prev_mask):

        renew_cols_idx_list = batch[0][3]

        for i, idx in enumerate(renew_cols_idx_list):
            if self.last_query_flag:
                #print(batch[0][0][:, -1, idx])
                #print(prev_out[:, i])
                batch[0][0][:, -1, idx] = prev_out[:, i].item()
                #print(batch[0][0][:, -1, idx])
                #pdb.set_trace()
            else:

                #print(batch[0][0][:,(~batch[0][1]).view(-1),idx])
                #print(prev_out[:,(~prev_mask).view(-1),i])
                
                batch[0][0][:,(~prev_mask).view(-1),idx]= prev_out[:,(~prev_mask).view(-1),i].double()
                #print(batch[0][0][:,(~batch[0][1]).view(-1),idx])
                #pdb.set_trace()

                #batch[0][0][:, -1, idx] = out[:, i].item()


        
        return batch

    def renew_batch(self, batch):

        current_id = batch[0][2].item()


        if self.prev_idx_for_renew_batch == current_id:
            
            batch[0][0][:, :-1,:] = self.prev_batch[0][0][:,1:,:]
            batch = self.register_batch(batch, self.prev_out, prev_mask=self.prev_batch[0][1])
            #pdb.set_trace()

        self.prev_idx_for_renew_batch = current_id

        return batch
            


    def test_step(self, batch, batch_idx):

        #print(batch)
        if self.oof_prediction==False:
            batch = self.renew_batch(batch)

        out = self._forward(batch)
        
        if self.oof_prediction==False:
            self.prev_batch = batch #self.register_batch(batch, out)
            self.prev_out = out #self.register_batch(batch, out)

        
        sort_idx = batch[0][4]
        if self.last_query_flag==False:
            out =  out[~batch[0][1]]
            sort_idx = sort_idx[~batch[0][1]]
        
        return {'out': out, "sort_idx":sort_idx}
        
    def test_epoch_end(self, outputs):
        #
        sort_idx = torch.cat([o['sort_idx'] for o in outputs])
        pred = torch.cat([o['out'] for o in outputs])
        if self.last_query_flag==False:
            sort_idx = sort_idx.data.cpu().detach().numpy()
            pred = pred.data.cpu().detach().numpy()
            df = pd.DataFrame(pred)
            df["idx"] = sort_idx
            gp = df.groupby("idx").mean()
            self.final_preds = gp.values

        else:
            self.final_preds = pred[sort_idx].data.cpu().detach().numpy()
       
        
        


class LastQueryTransformer(PytorchLightningModelBase):
    def __init__(self,
                 n_numerical_features: int,
                 emb_dim_pairs_list,
                 num_bssid_features:int,
                 d_model: int = 32*4,
                 nhead: int = 1,
                 num_encoder_layers: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 dropout_emb: float = 0.0,

                 lstm_hidden_dim:int = 128,
                 num_lstm_layers:int = 1,
                 dropout_lstm:float = 0.1,
                 
                 norm_encoder_output: bool = False,
                 
                 on_training_epoch_end: Optional[List[Callable]] = None,
                 on_validation_epoch_end: Optional[List[Callable]] = None,
                 subtract: bool = False) -> None:
        super().__init__()


        self.prev_idx_for_renew_batch = -1


        self.n_numerical_features = n_numerical_features
        self.num_bssid_features=num_bssid_features
        
        self.d_model = d_model


        self.emb_layers = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_dim_pairs_list])
        self.total_cat_emb_dim = sum([d for m, d in emb_dim_pairs_list])
        self.categorical_proj = nn.Sequential(
            nn.Linear(self.total_cat_emb_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.bssid_emb_layer = nn.Embedding(29241, self.d_model)
        self.total_bssid_emb_dim = self.num_bssid_features * self.d_model
        self.bssid_proj = nn.Sequential(
            nn.Linear(self.total_bssid_emb_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.conti_embed_layers = nn.ModuleList([nn.Linear(1,self.d_model,bias=False) for n in range(n_numerical_features)])
        self.continuous_proj = nn.Sequential(
            nn.Linear(self.d_model*n_numerical_features, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.emb_src_dim = self.d_model * 3

        self.postition = PositionEncode(self.emb_src_dim)

        self.dropout_emb = nn.Dropout(dropout_emb)
        self.layer_normal = nn.LayerNorm(self.emb_src_dim)

        #self.conv = nn.Conv1d(in_channels=self.d_emb, out_channels=self.d_model, kernel_size=1)

        self.lstm = nn.LSTM(
            input_size=self.emb_src_dim,
            bidirectional=False,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout_lstm)

        self.emb_src_dim = lstm_hidden_dim
        encoder_layer = LastQueryTransformerEncoderLayer(self.emb_src_dim, nhead, dim_feedforward, dropout)
        norm = nn.LayerNorm(self.emb_src_dim) if norm_encoder_output else None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm)

        self.decoder_xy = nn.Sequential(
            nn.Linear(self.emb_src_dim, self.d_model),
            #nn.ReLU(True),
            nn.Linear(self.d_model, 2),
        )


        self.decoder_floor = nn.Linear(self.emb_src_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        for i, l in enumerate(self.emb_layers):
            l.weight.data.uniform_(-initrange, initrange)

        for i, l in enumerate(self.conti_embed_layers):
            l.weight.data.uniform_(-initrange, initrange)

        self.bssid_emb_layer.weight.data.uniform_(-initrange, initrange)

        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)



    def _forward(self, batch):

        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        src_key_padding_mask is (N, S) with boolean True/False. #padにmaskするやつ
        src_mask is (S, S) with float(’-inf’) and float(0.0). #未来の情報にmaskするやつ
        '''
        #
        src = batch[0][0]


        add_flag=False

        if add_flag:
            emb_sum = None
            for i, emb_layer in enumerate(self.emb_layers):
                emb = emb_layer(src[...,i])
                if i == 0:
                    emb_sum = emb
                else:
                    emb_sum += emb

            num_s = len(self.emb_layers) 
            for i, emb_layer in enumerate(self.conti_embed_layers):
                #import pdb; pdb.set_trace()
                emb_sum += emb_layer(src[...,i+num_s].unsqueeze(-1))
            
            embedded_src = emb_sum.transpose(0, 1) # (S, N, E)
        else:

           

            emb_list = [emb_layer(src[...,i].long()) for i, emb_layer in enumerate(self.emb_layers)]
            emb_concat = torch.cat(emb_list, axis=-1)
            #print(f"emb_concat : {emb_concat.shape}")
            emb_concat = self.categorical_proj(emb_concat)
            #print(f"emb_concat proj: {emb_concat.shape}")

            num_s = len(self.emb_layers)
            bssid_emb = [self.bssid_emb_layer(src[..., i+num_s].long()) for i in range(self.num_bssid_features)]
            bssid_emb_concat = torch.cat(bssid_emb, axis=-1)
            bssid_emb_concat = self.bssid_proj(bssid_emb_concat) 

            

            num_s += self.num_bssid_features
            conti_emb_list = [emb_layer(src[...,i+num_s].unsqueeze(-1).float()) for i, emb_layer in enumerate(self.conti_embed_layers)]
            #print(f"conti_emb_list : {conti_emb_list}")
            #print(f"conti_emb_list : {conti_emb_list[0].shape}")
            conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
            #print(f"conti_emb_concat : {conti_emb_concat.shape}")

            conti_emb_concat = self.continuous_proj(conti_emb_concat)
            #print(f"conti_emb_concat proj : {conti_emb_concat.shape}")


            embedded_src = torch.cat([emb_concat, bssid_emb_concat, conti_emb_concat], axis=-1)
            #embedded_src = emb_concat

            #print(f"embedded_src : {embedded_src.shape}")
            

            embedded_src=embedded_src.transpose(0, 1)

            #import pdb; pdb.set_trace()

        #print(f"embedded_src : {embedded_src.shape}")
        #


        #input_emb = embedded_src * np.sqrt(self.ninp)

        
        input_emb = self.dropout_emb(embedded_src)
        input_emb = self.layer_normal(input_emb)

        #pdb.set_trace()

        input_emb,_ = self.lstm(input_emb)

        #embedded_src += self.postition(embedded_src)

        # row_id = src[-2]
        src_key_padding_mask = batch[0][1]

        

        src_mask = None#self.generate_square_subsequent_mask(input_emb.size(0)).cuda()
        output = self.transformer_encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #output = input_emb
        #print(f"out enc: {output}")
        #print(f"out enc: {output.shape}")

        #pdb.set_trace()
        
        #output_xy = self.reg_out_xy(src[2], output)
        #output_floor = self.reg_out_floor(src[2], output)
        
        
        
        output_last = output[-1, ...]
        output_xy = self.decoder_xy(output_last)
        output_floor = self.decoder_floor(output_last)
        #print(f"output_xy dec: {output_xy.shape}")
        #print(f"output_floor dec: {output_floor.shape}")
        
        output_all = torch.cat([output_xy, output_floor], axis=-1)
        #print(f"output_all dec: {output_all.shape}")

        #output = self.out_act(output)
        #print(f"out act: {output}")

        # output_xy = output_xy.transpose(1, 0)
        # output_floor = output_floor.transpose(1, 0)
        # print(f"output_xy dec: {output_xy.shape}")
        # print(f"output_floor dec: {output_floor.shape}")
        #print(f"out transpose: {output.shape}")

        #output = output[~src_key_padding_mask]
        #print(f"out mask: {output.shape}")


        #self.current_row_id = row_id[~src_key_padding_mask]

        #
        
        return output_all

    def forward(self, batch):
        out = self._forward(batch)
        return out
        
    def register_batch(self, batch, out):

        renew_cols_idx_list = batch[0][3]

        for i, idx in enumerate(renew_cols_idx_list):

            batch[0][0][:, -1, idx] = out[:, i].item()

        
        return batch

    def renew_batch(self, batch):

        current_id = batch[0][2].item()


        if self.prev_idx_for_renew_batch == current_id:
            
            batch[0][0][:, :-1,:] = self.prev_batch[0][0][:,1:,:]
            #pdb.set_trace()

        self.prev_idx_for_renew_batch = current_id

        return batch
            


    def test_step(self, batch, batch_idx):

        #print(batch)
        if self.oof_prediction==False:
            batch = self.renew_batch(batch)

        out = self._forward(batch)
        
        if self.oof_prediction==False:
            self.prev_batch = self.register_batch(batch, out)

        

        return {'out': out}
        
        
        


class LastQueryDualTransformer(pl.LightningModule):
    def __init__(self,
                 n_numerical_features: int,
                 n_categorical_features: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 dropout_emb: float = 0.0,
                 dropout_lstm: float = 0.1,
                 num_lstm_layers: int = 2,
                 lstm_hidden_dim: int = 128,
                 norm_encoder_output: bool = False,
                 cat_input_dims: Optional[Collection[int]] = None,
                 cat_emb_dims: Optional[Union[int, Collection[int]]] = None,
                 cat_emb_assign_map: Optional[Dict[int, int]] = None,
                 learning_rate: float = 1e-3,
                 on_training_epoch_end: Optional[List[Callable]] = None,
                 on_validation_epoch_end: Optional[List[Callable]] = None,
                 subtract: bool = False) -> None:
        super().__init__()

        self.n_numerical_features = n_numerical_features
        self.n_categorical_features = n_categorical_features
        self.d_model = d_model

        if cat_input_dims:
            assert cat_emb_dims is not None
            if not isinstance(cat_emb_dims, Collection):
                cat_emb_dims = [cat_emb_dims] * len(cat_input_dims)
            assert len(cat_input_dims) == len(cat_emb_dims)

            self.emb_layers = nn.ModuleList([nn.Embedding(d_in, d_out)
                                             for d_in, d_out in zip(cat_input_dims, cat_emb_dims)])
            self.d_emb = sum(cat_emb_dims) + n_numerical_features
        else:
            self.d_emb = n_numerical_features

        self.dropout_emb = nn.Dropout(dropout_emb)

        self.conv = nn.Conv1d(in_channels=self.d_emb, out_channels=self.d_model, kernel_size=1)

        encoder_layer = LastQueryTransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout)
        norm = nn.LayerNorm(self.d_model) if norm_encoder_output else None
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm)

        self.seq = nn.LSTM(
            input_size=self.d_model,
            bidirectional=False,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout_lstm)

        self.subtract = subtract

        self.cat_emb_assign_map = cat_emb_assign_map
        self.linear = nn.Linear(3*lstm_hidden_dim if subtract else 2*lstm_hidden_dim, 1)

        self.learning_rate = learning_rate
        self.cat_emb_assign_map = cat_emb_assign_map

        self._on_training_epoch_end = on_training_epoch_end
        self._on_validation_epoch_end = on_validation_epoch_end

    def get_emb(self, i) -> nn.Embedding:
        if self.cat_emb_assign_map is not None and i in self.cat_emb_assign_map:
            idx_emb = self.cat_emb_assign_map[i]
        else:
            idx_emb = i
        return self.emb_layers[idx_emb]

    def _extract(self,
                 src_numerical: torch.Tensor,
                 src_categorical: Optional[torch.Tensor]
                 ) -> torch.Tensor:
        """
        :param src_numerical: [batch, seq_len, #features]
        :param src_categorical: [batch, seq_len, #features]
        """
        if self.n_categorical_features:
            assert src_categorical is not None
            embed = []

            for i in range(self.n_categorical_features):
                embed.append(self.get_emb(i)(src_categorical[:, :, i]))

            embed = torch.cat(embed, dim=2)
            embed = self.dropout_emb(embed)
            src = torch.cat([src_numerical, embed], dim=2)
        else:
            src = src_numerical

        # src: [batch, seq_len, d_emb] => [batch, d_emb, seq_len]
        src = src.permute([0, 2, 1])

        # src: [batch, d_emb, seq_len] => [batch, d_model, seq_len]
        src = self.conv(src)

        # src: [batch, d_model, seq_len] => [seq_len, batch, d_model]
        src = src.permute([2, 0, 1])

        # memory: [seq_len, batch, d_model]
        memory = self.encoder(src)

        # hidden: [seq_len, batch, 2*lstm_hidden_dim]
        hidden, _ = self.seq(memory)

        # last_state: [batch, 2*lstm_hidden_dim]
        last_state = hidden[-1, :, :]

        return last_state

    def _forward(self,
                src_numerical1: torch.Tensor,
                src_categorical1: Optional[torch.Tensor],
                src_numerical2: torch.Tensor,
                src_categorical2: Optional[torch.Tensor],

                ):
        # out1, out2 : [batch, lstm_hidden_dim]
        out1 = self._extract(src_numerical1, src_categorical1)
        out2 = self._extract(src_numerical2, src_categorical2)

        if self.subtract:
            out = torch.cat([out1, out2, out1 - out2], dim=1)
        else:
            out = torch.cat([out1, out2], dim=1)

        out = self.linear(out)

        return out[:, 0]

    def forward(self,
                src_numerical1: torch.Tensor,
                src_categorical1: Optional[torch.Tensor],
                src_numerical2: torch.Tensor,
                src_categorical2: Optional[torch.Tensor]
                ):
        out = self._forward(src_numerical1, src_categorical1, src_numerical2, src_categorical2)
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        src_numerical1, src_categorical1, src_numerical2, src_categorical2, y, y_w = batch

        out = self._forward(src_numerical1, src_categorical1, src_numerical2, src_categorical2)

        loss = nn.BCEWithLogitsLoss(weight=y_w)(out, y)

        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src_numerical1, src_categorical1, src_numerical2, src_categorical2, y, y_w = batch

        out = self._forward(src_numerical1, src_categorical1, src_numerical2, src_categorical2)

        loss = nn.BCEWithLogitsLoss()(out, y)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        src_numerical1, src_categorical1, src_numerical2, src_categorical2, _, _ = batch

        out = self._forward(src_numerical1, src_categorical1, src_numerical2, src_categorical2)

        return {'out': out}

    def training_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs]).mean()

        if self._on_training_epoch_end is not None:
            for f in self._on_training_epoch_end:
                f(self.current_epoch, loss)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()

        if self._on_validation_epoch_end is not None:
            for f in self._on_validation_epoch_end:
                f(self.current_epoch, loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class RegressionHead(nn.Module):
    
    def __init__(self, site_dim, d_model, nhead, output_dim):
        super().__init__()
        
        
        self.site_emb_layer = nn.Embedding(site_dim, d_model)
       
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        
        self.out_layer = nn.Linear(d_model, output_dim)
        
        
    def forward(self, site_x, memory):
        
        site_emb =self.site_emb_layer(site_x)
        site_emb=site_emb.transpose(0, 1)

        
        
        attn_output, attn_output_weights = self.multihead_attn(query=site_emb, key=memory, value=memory)
        
        
        
        out = self.out_layer(attn_output[0])
        
       
        return out
            


class Transformer_DNN(nn.Module):
    def __init__(self,
                ninp:int,
                nhead:int, 
                nhid:int, 
                nlayers:int,
                emb_dim_pairs_list,
                num_continuout_features,
                dropout:float =0.5,
                site_dim=24,
                ):
        
        super(Transformer_DNN, self).__init__()

        #self.current_row_id=0
        
        s=1
        self.ninp = ninp*s

        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model=self.ninp, 
                                                 nhead=nhead, 
                                                 dim_feedforward=nhid, 
                                                 dropout=dropout, 
                                                 activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.emb_layers = nn.ModuleList([nn.Embedding(m, self.ninp) for m, _ in emb_dim_pairs_list])
        self.categorical_proj = nn.Sequential(
            nn.Linear(self.ninp*len(emb_dim_pairs_list), self.ninp//2),
            nn.LayerNorm(self.ninp//2),
        )  


        self.conti_embed_layers = nn.ModuleList([nn.Linear(1,self.ninp,bias=False) for n in range(num_continuout_features)])
        self.continuous_proj = nn.Sequential(
            nn.Linear(self.ninp*num_continuout_features, self.ninp//2),
            nn.LayerNorm(self.ninp//2),
        )
        
        
        #self.postition = PositionEncode(self.ninp)

        # self.decoder_xy = nn.Linear(ninp, 2)
        # self.decoder_floor = nn.Linear(ninp, 1)
        # self.out_act = nn.Sigmoid()
        self.reg_out_xy = RegressionHead(site_dim=site_dim, d_model=self.ninp, nhead=nhead, output_dim=2)
        self.reg_out_floor = RegressionHead(site_dim=site_dim, d_model=self.ninp, nhead=nhead, output_dim=1)

        self.layer_normal = nn.LayerNorm(self.ninp)
        self.dropout = nn.Dropout(p=dropout)

        
        
        
        self.proj = nn.Linear(self.ninp*len(emb_dim_pairs_list)+self.ninp*num_continuout_features, self.ninp)
        
        
        self.site_emb_layer = nn.Embedding(site_dim, 2)
        self.dropout_wide = nn.Dropout(p=dropout)
        self.layer_normal_wide = nn.LayerNorm(ninp*100+2)
        self.proj_wide = nn.Linear(ninp*100+2, self.ninp)
        # self.proj_wide2 = nn.Linear(self.ninp, 128)
        # self.proj_wide3 = nn.Linear(128, 16)
        self.proj_out_xy = nn.Linear(self.ninp, 2)
        self.proj_out_floor = nn.Linear(self.ninp, 1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1

        for i, l in enumerate(self.emb_layers):
            l.weight.data.uniform_(-initrange, initrange)

        for i, l in enumerate(self.conti_embed_layers):
            l.weight.data.uniform_(-initrange, initrange)

        self.site_emb_layer.weight.data.uniform_(-initrange, initrange)

        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        
        emb_list = [emb_layer(src[i]) for i, emb_layer in enumerate(self.emb_layers)]
        emb_concat = torch.cat(emb_list, axis=-1)
        emb_concat = emb_concat.view(emb_concat.shape[0], -1)
        #print(emb_concat.shape)
        
        
        num_s = len(self.emb_layers) 
        conti_emb_list = [emb_layer(src[i+num_s].unsqueeze(-1)) for i, emb_layer in enumerate(self.conti_embed_layers)]
        conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
        conti_emb_concat = conti_emb_concat.view(conti_emb_concat.shape[0], -1)
        #print(conti_emb_concat.shape)
        
        site_emb =self.site_emb_layer(src[2])
        site_emb = site_emb.view(site_emb.shape[0], -1)
        #print(site_emb.shape)
        
        
        embedded_src = torch.cat([emb_concat, conti_emb_concat, site_emb], axis=-1)
        x = self.dropout_wide(embedded_src)
        x = self.layer_normal_wide(x)
        x = self.proj_wide(x)
        #x = self.proj_wide2(x)
        #x = self.proj_wide3(x)
        output_xy = self.proj_out_xy(x)
        output_floor = self.proj_out_floor(x)
        output_all = torch.cat([output_xy, output_floor], axis=-1)
        
        return output_all
        #import pdb; pdb.set_trace()
        
        
        
        

    def __forward(self, src):
        
        emb_list = [emb_layer(src[i]) for i, emb_layer in enumerate(self.emb_layers)]
        emb_concat = torch.cat(emb_list, axis=-1)
        
        
        num_s = len(self.emb_layers) 
        conti_emb_list = [emb_layer(src[i+num_s].unsqueeze(-1)) for i, emb_layer in enumerate(self.conti_embed_layers)]
        conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
        
        embedded_src = torch.cat([emb_concat, conti_emb_concat], axis=-1)
        
        embedded_src = self.proj(embedded_src)
        
        embedded_src=embedded_src.transpose(0, 1)
        
        
        input_emb = self.dropout(embedded_src)
        input_emb = self.layer_normal(input_emb)

        

        # row_id = src[-2]
        src_key_padding_mask = None

        src_mask = None#self.generate_square_subsequent_mask(input_emb.size(0)).cuda()
        #output = self.transformer_encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = input_emb
        #print(f"out enc: {output}")
        #print(f"out enc: {output.shape}")
        
        output_xy = self.reg_out_xy(src[2], output)
        output_floor = self.reg_out_floor(src[2], output)
        
        
        
        #output0 = output[0, ...]
        #output_xy = self.decoder_xy(output0)
        #output_floor = self.decoder_floor(output0)
        #print(f"output_xy dec: {output_xy.shape}")
        #print(f"output_floor dec: {output_floor.shape}")
        
        output_all = torch.cat([output_xy, output_floor], axis=-1)
        
        
        return output_all
        
        
        
        
        
        
        

    def _forward(self, src):

        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        src_key_padding_mask is (N, S) with boolean True/False. #padにmaskするやつ
        src_mask is (S, S) with float(’-inf’) and float(0.0). #未来の情報にmaskするやつ
        '''
        #

        add_flag=False

        if add_flag:
            emb_sum = None
            for i, emb_layer in enumerate(self.emb_layers):
                emb = emb_layer(src[i])
                if i == 0:
                    emb_sum = emb
                else:
                    emb_sum += emb

            num_s = len(self.emb_layers) 
            for i, emb_layer in enumerate(self.conti_embed_layers):
                #import pdb; pdb.set_trace()
                emb_sum += emb_layer(src[i+num_s].unsqueeze(-1))
            
            embedded_src = emb_sum.transpose(0, 1) # (S, N, E)
        else:
            

            emb_list = [emb_layer(src[i]) for i, emb_layer in enumerate(self.emb_layers)]
            emb_concat = torch.cat(emb_list, axis=-1)
            #print(f"emb_concat : {emb_concat.shape}")
            emb_concat = self.categorical_proj(emb_concat)
            #print(f"emb_concat proj: {emb_concat.shape}")

            num_s = len(self.emb_layers) 
            conti_emb_list = [emb_layer(src[i+num_s].unsqueeze(-1)) for i, emb_layer in enumerate(self.conti_embed_layers)]
            #print(f"conti_emb_list : {conti_emb_list}")
            #print(f"conti_emb_list : {conti_emb_list[0].shape}")
            conti_emb_concat = torch.cat(conti_emb_list, axis=-1)
            #print(f"conti_emb_concat : {conti_emb_concat.shape}")

            conti_emb_concat = self.continuous_proj(conti_emb_concat)
            #print(f"conti_emb_concat proj : {conti_emb_concat.shape}")

            
            embedded_src = torch.cat([emb_concat, conti_emb_concat], axis=-1)
            #embedded_src = emb_concat

            #print(f"embedded_src : {embedded_src.shape}")


            embedded_src=embedded_src.transpose(0, 1)

            #import pdb; pdb.set_trace()

        #print(f"embedded_src : {embedded_src.shape}")



        #input_emb = embedded_src * np.sqrt(self.ninp)

        #embedded_src += self.postition(embedded_src)
        input_emb = self.dropout(embedded_src)
        input_emb = self.layer_normal(input_emb)

        

        # row_id = src[-2]
        src_key_padding_mask = None

        src_mask = None#self.generate_square_subsequent_mask(input_emb.size(0)).cuda()
        output = self.transformer_encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #output = input_emb
        #print(f"out enc: {output}")
        #print(f"out enc: {output.shape}")
        
        output_xy = self.reg_out_xy(src[2], output)
        output_floor = self.reg_out_floor(src[2], output)
        
        
        
        #output0 = output[0, ...]
        #output_xy = self.decoder_xy(output0)
        #output_floor = self.decoder_floor(output0)
        #print(f"output_xy dec: {output_xy.shape}")
        #print(f"output_floor dec: {output_floor.shape}")
        
        output_all = torch.cat([output_xy, output_floor], axis=-1)
        #print(f"output_all dec: {output_all.shape}")

        #output = self.out_act(output)
        #print(f"out act: {output}")

        # output_xy = output_xy.transpose(1, 0)
        # output_floor = output_floor.transpose(1, 0)
        # print(f"output_xy dec: {output_xy.shape}")
        # print(f"output_floor dec: {output_floor.shape}")
        #print(f"out transpose: {output.shape}")

        #output = output[~src_key_padding_mask]
        #print(f"out mask: {output.shape}")


        #self.current_row_id = row_id[~src_key_padding_mask]

        #
        
        return output_all

    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score

    


class MyDatasetLSTM(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag):

        
        self.transform_ = None#dataset_params["transform_class"]()
        self.df_train_X_sequence_ = df_train_X[dataset_params["sequence_features_list"]] #np.array(df_train_X[dataset_params["sequence_features_list"]].values.tolist()).transpose(0, 2, 1).astype(np.int64)
        self.df_train_X_embedding_cat_ = df_train_X[dataset_params["embedding_category_features_list"]] 
        self.df_train_X_cont_ = df_train_X[dataset_params["continuous_features_list"]] 
        
        #self.np_train_X_cat_ = df_train_X[dataset_params["embedding_category_features_list"]].values.astype(np.int64)
        #self.np_train_X_cont_ = df_train_X[dataset_params["continuous_features_list"]].values.astype(np.float32)
        
        self.train_sequence_index_list = df_train_X[dataset_params["sequence_index_col"]].unique()

        if (train_flag==True) & (dataset_params["length_validation"]==True):
            self.se_train_input_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["input_sequence_dummy_len_col"]].agg(pd.Series.mode)
            self.se_train_output_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["output_sequence_dummy_len_col"]].agg(pd.Series.mode)
        else:
            self.se_train_input_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["input_sequence_len_col"]].agg(pd.Series.mode)
            self.se_train_output_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["output_sequence_len_col"]].agg(pd.Series.mode)
            
        self.se_train_weight = df_train_X[dataset_params["weight_col"]]

        #print(df_train_y.shape)
        self.df_train_y_ = df_train_y #np.array(df_train_y.values.tolist()).transpose(0, 2, 1).astype(np.float32)
        #print((self.np_train_X_cat_.shape))
        #sys.exit()
        #print(type(self.np_train_y_ ))

 

    def __len__(self):
        return len(self.train_sequence_index_list)

    def __getitem__(self, idx):
        #print("idx:{}".format(idx))
        
        mol_id = self.train_sequence_index_list[idx]
        #print(self.se_train_input_sequence_len)
        input_len = self.se_train_input_sequence_len.loc[mol_id]
        output_len = self.se_train_output_sequence_len.loc[mol_id]
        # print(f"mol_id : {mol_id}")
        # print(f"input_len : {input_len}")
        # print(f"output_len : {output_len}")
        
        id_seqpos_list = [f"{mol_id}_{i}" for i in range(input_len)]
        out_data_seq = self.df_train_X_sequence_.loc[id_seqpos_list, :].values.astype(np.int64)
        #df_sample = self.df_train_X_sequence_.loc[id_seqpos_list, :]
        #out_data_seq = df_sample.values.astype(np.int64)
        
        out_data_cat = self.df_train_X_embedding_cat_.loc[id_seqpos_list, :].values.astype(np.int64) #self.np_train_X_cat_[idx, ...] if len(self.np_train_X_cat_) > 0 else []
        out_data_cont = self.df_train_X_cont_.loc[id_seqpos_list, :].values.astype(np.float32) #self.np_train_X_cont_[idx, ...] if len(self.np_train_X_cont_) > 0 else []
        #print(out_data_seq.shape)
        #print(out_data_cat.shape)
        #print(out_data_cont.shape)
        #sys.exit()


        id_output_list = [f"{mol_id}_{i}" for i in range(output_len)]
        #print(id_output_list)
        #print(self.df_train_y_.loc[id_output_list, :])
        out_label = self.df_train_y_.loc[id_output_list, :].values.astype(np.float32)

        weight = self.se_train_weight.loc[id_output_list].values.reshape(-1, 1).astype(np.float32)

        # print(f"out_label : {out_label.shape}")
        # print(f"weight : {weight.shape}")
        # print(weight)

        np_label_and_weight = np.concatenate([out_label, weight], axis=1)

        # print(f"np_label_and_weight : {np_label_and_weight.shape}")
        # print(np_label_and_weight[:,-1])
        
        # sys.exit()

        if self.transform_ is not None:
            pass
            
            # out_data_cat = self.transform_(out_data_cat)
            # out_label = self.transform_(out_label)

        return [out_data_seq, output_len, input_len, out_data_cat, out_data_cont], np_label_and_weight   
    
    
class LSTM_DNN(nn.Module):
    def __init__(self,
                seq_dim_pairs_list,
                emb_dim_pairs_list, 
                num_cont_features,
                num_target,
                seq_dropout=0.2, 
                hidden_dim=128 * 2, 
                hidden_layers=3, 
                emb_dropout=0.2,
                ):
        
        super(LSTM_DNN, self).__init__()
        
        
        self.seq_dropout = seq_dropout
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers


        self.save_param_epoch_=0
        self.save_param_val_score_=0

        self.num_seq_D_features_ = sum([D for m, D in seq_dim_pairs_list])
        self.num_embs_D_features_ = sum([D for m, D in emb_dim_pairs_list])
        self.num_cont_features_ = num_cont_features

        
        self.seq_layers = nn.ModuleList([nn.Embedding(m, D) for m, D in seq_dim_pairs_list])
        self.emb_layers = nn.ModuleList([nn.Embedding(m, D) for m, D in emb_dim_pairs_list])
        self.emb_dropout = emb_dropout #nn.Dropout(emb_dropout)

        self.flag=0

        if self.flag==0:

            self.hidden_dim = 64
        
            self.lstm = nn.LSTM(
                input_size=self.num_seq_D_features_ + self.num_cont_features_,
                hidden_size=self.hidden_dim,
                num_layers=self.hidden_layers,
                dropout=self.seq_dropout,
                bidirectional=True,
                batch_first=True,
            )
        elif self.flag==2:
            
            self.hidden_dim = 64

            self.gru = nn.GRU(
                input_size=self.num_seq_D_features_+ self.num_cont_features_,
                hidden_size=self.hidden_dim,
                num_layers=self.hidden_layers,
                dropout=self.seq_dropout,
                bidirectional=True,
                batch_first=True,

            )
        elif self.flag==8:
            self.lstm_four = nn.LSTM(
                input_size=self.num_seq_D_features_ + self.num_cont_features_,
                hidden_size=self.hidden_dim,
                num_layers=4,
                dropout=self.seq_dropout,
                bidirectional=True,
                batch_first=True,
            )

            # self.lstm_double = nn.LSTM(
            #     input_size=self.num_seq_D_features_ + self.num_cont_features_,
            #     hidden_size=self.hidden_dim,
            #     num_layers=2,
            #     dropout=self.seq_dropout,
            #     bidirectional=True,
            #     batch_first=True,
            # )

        

        # self.lstm0 = nn.LSTM(
        #     input_size=self.num_seq_D_features_ + self.num_cont_features_,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,
        # )

        # self.lstm1 = nn.LSTM(
        #     input_size=hidden_dim *2,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,
        # )
        # self.lstm2 = nn.LSTM(
        #     input_size=hidden_dim *2,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,
        # )

        # self.gru = nn.GRU(
        #     input_size=self.num_seq_D_features_+ self.num_cont_features_,
        #     hidden_size=self.hidden_dim,
        #     num_layers=self.hidden_layers,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,

        # )

        # self.gru0 = nn.GRU(
        #     input_size=self.num_seq_D_features_+ self.num_cont_features_,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,

        # )

        # self.gru1 = nn.GRU(
        #     input_size=hidden_dim *2,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,#self.hidden_layers,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,

        # )

        # self.gru2 = nn.GRU(
        #     input_size=hidden_dim *2,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,#self.hidden_layers,
        #     dropout=self.seq_dropout,
        #     bidirectional=True,
        #     batch_first=True,

        # )

        seq_output_dim = self.hidden_dim *2

        self.linear = nn.Linear(seq_output_dim, num_target) #bidirectional

        self.linear0 = nn.Linear(seq_output_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_target)

        self.total_linear = nn.Linear(seq_output_dim + self.num_embs_D_features_ , hidden_dim)

        # self.postition = PositionEncode(hidden_dim)
        # self.transformer = torch.nn.TransformerEncoder(
        #     torch.nn.TransformerEncoderLayer(hidden_dim, 8, 256, dropout=0.2, activation='relu'),
        #     2
        #)

    def forward(self, input_X_list):
        seq_data = input_X_list[0]
        pred_len = input_X_list[1][0]
        cat_data = input_X_list[3]
        cont_data = input_X_list[4]

        batch_size = seq_data.shape[0]

        if self.num_seq_D_features_ != 0:

            x = [seq_layer(seq_data[..., i]) for i, seq_layer in enumerate(self.seq_layers)]
            
            x = torch.cat(x, 2)
            if self.flag == 2:
                x = x.permute(0, 2, 1) # reshape to [batch, channels, time]
                x = F.dropout2d(x, self.seq_dropout, training=self.training) # groups by [batch, channels], drops out a channel for all timestamps
                x = x.permute(0, 2, 1) # back to [batch, time, channels]
            
            if self.num_cont_features_ != 0: 
                x = torch.cat([x, cont_data], 2)

        else:
            print("error")
            #x = cont_data
            sys.exit()
        
        #print(f"x : {x.shape}")

        
        if self.flag==0:
            output, hidden = self.lstm(x)
        # elif flag==1:
        #     output, hidden = self.lstm0(x)
        #     output, hidden = self.gru1(output)
        elif self.flag == 2:
            output, hidden = self.gru(x)
        # elif flag==3:
        #     output, hidden = self.gru0(x)
        #     output, hidden = self.lstm1(output)
        
        # elif flag==4:
        #     output, hidden = self.gru0(x)
        #     output, hidden = self.lstm1(output)
        #     output, hidden = self.lstm2(output)

        # elif flag==5:
        #     output, hidden = self.lstm0(x)
        #     output, hidden = self.gru1(output)
        #     output, hidden = self.lstm2(output)

        # elif flag==6:
        #     output, hidden = self.lstm0(x)
        #     output, hidden = self.lstm1(output)
        #     output, hidden = self.gru2(output)

        # elif flag==7:
        #     output, hidden = self.lstm_double(x)

        elif self.flag==8:
            output, hidden = self.lstm_four(x)
            


        #truncated = output[:, : pred_len, :]
        #out_seq = truncated
        out_seq = output

        total_list = [out_seq]

        if self.num_embs_D_features_ != 0:

            cat_x = [emb_layer(cat_data[..., i]) for i, emb_layer in enumerate(self.emb_layers)]
            # for e in cat_x:
            #     print(f"emb : {e.shape}")
            cat_x = torch.cat(cat_x, 2)

            cat_x = cat_x.permute(0, 2, 1) # reshape to [batch, channels, time]
            cat_x = F.dropout2d(cat_x, self.emb_dropout, training=self.training) # groups by [batch, channels], drops out a channel for all timestamps
            cat_x = cat_x.permute(0, 2, 1) # back to [batch, time, channels]
            cat_x = cat_x[:, : pred_len, :]

            total_list.append(cat_x)
        

        
        total_x = torch.cat(total_list, 2)
        out = self.total_linear(total_x)
        out = self.linear2(out)
        y = out[:, : pred_len, :]
        #print(f"y : {y.shape}")


        return y

    def forward2(self, input_X_list):

        seq_data = input_X_list[0]
        pred_len = input_X_list[1][0]
        
        

        if self.num_seq_D_features_ != 0:

            x = [seq_layer(seq_data[..., i]) for i, seq_layer in enumerate(self.seq_layers)]
            #for e in x:
            #    print(f"emb : {e.shape}")
            x = torch.cat(x, 2)
            x = x.permute(0, 2, 1) # reshape to [batch, channels, time]
            x = F.dropout2d(x, self.seq_dropout, training=self.training) # groups by [batch, channels], drops out a channel for all timestamps
            x = x.permute(0, 2, 1) # back to [batch, time, channels]
            
        else:
            print("error")
            #x = cont_data
            sys.exit()
        
        #print(f"x : {x.shape}")
        output, hidden = self.lstm(x)
        #output, hidden = self.lstm1(output)
        #output, hidden = self.gru(x)
        #output, hidden = self.gru1(output)
        #output, hidden = self.gru2(output)
        #print(f"output : {output.shape}")
        #print(f"hidden : {hidden}")
        truncated = output[:, : pred_len, :]
        #print(f"truncated : {truncated.shape}")
        #out = self.linear(truncated)
        out = self.linear0(truncated)
        #out = self.linear1(out)
        out = self.linear2(out)
        y = out
        #print(f"y : {y.shape}")


        return y

    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score


def check_graph(graph_data):
    print("**** show graph proparty ****")
    print(f"structure of graph : {graph_data}")
    print(f"key of graph : {graph_data.keys}")
    print(f"number of node : {graph_data.num_nodes}")
    print(f"number of edges : {graph_data.num_edges}")
    print(f"number of node features : {graph_data.num_node_features}")
    print(f"contain isolation node : {graph_data.contains_isolated_nodes()}")
    print(f"contain self loops : {graph_data.contains_self_loops()}")
    print()
    print("===== fratures of node : x =====")
    print(graph_data["x"])
    print("===== labels of node : y =====")
    print(graph_data["y"])
    print("===== structure of edge =====")
    print(graph_data["edge_index"])


def getGraphDataBatch(node_features, edge_index_batchdata, edge_features_batchdata):
        
    batch_list = []
    for b in range(edge_index_batchdata.shape[0]):
        x = node_features[b]
        edge_features = edge_features_batchdata[b]
        # print(f"x : {x.shape}")
        # print(f"edge_features : {edge_features.shape}")
        # print(f"edge_features[edge_features != -1] ] {edge_features[edge_features != -1].shape}")

        new_edge_features = edge_features[edge_features != -1].reshape(-1, edge_features.shape[1])
        #print(new_edge_features)
        edge_index = edge_index_batchdata[b]
        new_edge_index = edge_index[edge_index != -1].reshape(2, -1)
        #print(edge_index)
        #print(new_edge_index.shape)


        data = Data(x=x.type(torch.float), edge_index=new_edge_index.type(torch.long), edge_attr=new_edge_features.type(torch.float))
        batch_list.append(data)
        #check_graph(data)
        #sys.exit()
    batch = Batch.from_data_list(batch_list, [])

    del batch_list
    #gc.collect()

    return batch

class MyDatasetGraph(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag, load_saved_graph_data=True):

        
        self.transform_ = None#dataset_params["transform_class"]()
        self.df_train_X_node_features_ = df_train_X[dataset_params["node_feature_list"]] #np.array(df_train_X[dataset_params["sequence_features_list"]].values.tolist()).transpose(0, 2, 1).astype(np.int64)
        self.se_train_X_node_index_ = df_train_X[dataset_params["node_index_col"]] 
        self.df_train_X_edge_connect_ = df_train_X[dataset_params["edge_connect_list"]] 
        self.df_train_X_edge_adjacent_features_ = df_train_X[dataset_params["edge_adjacent_feature_list"]]
        self.df_train_X_edge_connect_features_ = df_train_X[dataset_params["edge_connect_feature_list"]] 
         
        self.train_sequence_index_list = df_train_X[dataset_params["sequence_index_col"]].unique()

        if (train_flag==True) & (dataset_params["length_validation"]==True):
            self.se_train_input_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["input_sequence_dummy_len_col"]].agg(pd.Series.mode)
            self.se_train_output_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["output_sequence_dummy_len_col"]].agg(pd.Series.mode)
        else:
            
            self.se_train_input_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["input_sequence_len_col"]].agg(pd.Series.mode)
            self.se_train_output_sequence_len = df_train_X.groupby(dataset_params["sequence_index_col"])[dataset_params["output_sequence_len_col"]].agg(pd.Series.mode)
        


        self.se_train_weight = df_train_X[dataset_params["weight_col"]]
        #print(df_train_y.shape)
        self.df_train_y_ = df_train_y #np.array(df_train_y.values.tolist()).transpose(0, 2, 1).astype(np.float32)
        #print((self.np_train_X_cat_.shape))
        #sys.exit()
        #print(type(self.np_train_y_ ))

        self.load_saved_graph_data = load_saved_graph_data
        self.ppath_to_saved_data_dir = dataset_params["ppath_to_saved_data_dir"]
        if self.load_saved_graph_data:
            if self.ppath_to_saved_data_dir.exists():
                shutil.rmtree(self.ppath_to_saved_data_dir)
            os.makedirs(self.ppath_to_saved_data_dir)
        

            


 

    def __len__(self):
        return len(self.train_sequence_index_list)

    def __getitem__(self, idx):
        #print("idx:{}".format(idx))
        
        mol_id = self.train_sequence_index_list[idx]
        #print(f"mol id : {mol_id}")

        ppath_to_mol_id_data = self.ppath_to_saved_data_dir / f"local_data_{mol_id}.npz"
        
        if ppath_to_mol_id_data.exists() &  self.load_saved_graph_data:

            npz_file = np.load(ppath_to_mol_id_data)

            out_node_features=npz_file["out_node_features"]
            output_len=npz_file["output_len"]
            input_len=npz_file["input_len"]
            edge_index=npz_file["edge_index"]
            np_total_edge_feataures=npz_file["np_total_edge_feataures"]
            np_label_and_weight=npz_file["out_label"]
        
        else:

            

            #print(self.se_train_input_sequence_len)
            input_len = self.se_train_input_sequence_len.loc[mol_id]
            output_len = self.se_train_output_sequence_len.loc[mol_id]
            # print(f"mol_id : {mol_id}")
            # print(f"input_len : {input_len}")
            # print(f"output_len : {output_len}")


            
            id_seqpos_list = [f"{mol_id}_{i}" for i in range(input_len)]
            #out_node_features = self.df_train_X_node_features_.loc[id_seqpos_list, :].values 
            # print(out_node_features)
            # print(out_node_features.shape)
            
            
            np_idx = self.se_train_X_node_index_.loc[id_seqpos_list].values[:input_len-1]
            np_idx_next = np_idx +1 

            np_src = np.concatenate([np_idx, np_idx_next])
            np_dst = np.concatenate([np_idx_next, np_idx])

            df_edge_feature = self.df_train_X_edge_adjacent_features_.loc[id_seqpos_list, :].iloc[:input_len-1, :]
            df_edge_feature["adj_connect_forward"]=1

            df_edge_feature_inv = self.df_train_X_edge_adjacent_features_.loc[id_seqpos_list, :].iloc[:input_len-1, :]
            df_edge_feature_inv["adj_connect_backward"]=1

            df_adj_connect_edge_feature = pd.concat([df_edge_feature, df_edge_feature_inv]).fillna(0)
            #print(df_adj_connect_edge_feature)
            #sys.exit()

            total_edge_features_list = [df_adj_connect_edge_feature]

            df_mol_id = self.df_train_X_edge_connect_.loc[id_seqpos_list,:]

            for col in df_mol_id.columns:
                np_src_idx = np.array([src_i if dst_i != -1 else -1 for src_i, dst_i in zip(self.se_train_X_node_index_, df_mol_id[col])])
                np_dst_idx = df_mol_id[col].values

                df_edge_feature = self.df_train_X_edge_connect_features_.loc[id_seqpos_list, :]
                df_edge_feature["connect_forward"]=[ 1 if tmp_id != -1 else -1 for tmp_id in np_src_idx]

                df_edge_feature_inv = self.df_train_X_edge_connect_features_.loc[id_seqpos_list, :]
                df_edge_feature_inv["connect_backward"]=[ 1 if tmp_id != -1 else -1 for tmp_id in np_dst_idx]

                df_connect_edge_feature = pd.concat([df_edge_feature, df_edge_feature_inv]).fillna(0)
                total_edge_features_list.append(df_connect_edge_feature)

                np_src = np.concatenate([np_src, np_src_idx, np_dst_idx])
                np_dst = np.concatenate([np_dst, np_dst_idx, np_src_idx])
            # print(np_src)
            # print(np_dst)
            #print(f"np_src : {np_src.shape}")
            #print(f"np_dst : {np_dst.shape}")
            
            edge_index = np.array([np_src, np_dst])

            df_total_edge_features = pd.concat(total_edge_features_list).fillna(0)
            
            

            df_node_features, df_total_edge_features, edge_index = self.addCodon(df_node_features=self.df_train_X_node_features_.loc[id_seqpos_list, :], df_total_edge_features=df_total_edge_features, edge_index=edge_index)
            out_node_features = df_node_features.values

            m1_idx = (df_total_edge_features == -1).any(axis=1)
            df_total_edge_features.loc[m1_idx, :] = -1
            np_total_edge_feataures = df_total_edge_features.values


            id_output_list = [f"{mol_id}_{i}" for i in range(output_len)]
            #print(id_output_list)
            #print(self.df_train_y_.loc[id_output_list, :])
            out_label = self.df_train_y_.loc[id_output_list, :].values.astype(np.float32)
            weight = self.se_train_weight.loc[id_output_list].values.reshape(-1, 1).astype(np.float32)

            # print(f"out_label : {out_label.shape}")
            # print(f"weight : {weight.shape}")
            # print(weight)

            np_label_and_weight = np.concatenate([out_label, weight], axis=1)
            
            if self.load_saved_graph_data:
                np.savez_compressed(ppath_to_mol_id_data, 
                                    out_node_features=out_node_features, 
                                    output_len=output_len,
                                    input_len=input_len,
                                    edge_index=edge_index,
                                    np_total_edge_feataures=np_total_edge_feataures,
                                    out_label=np_label_and_weight)


        if self.transform_ is not None:
            pass
            
            # out_data_cat = self.transform_(out_data_cat)
            # out_label = self.transform_(out_label)

        return [out_node_features, output_len, input_len, edge_index, np_total_edge_feataures], np_label_and_weight

    def addCodon(self, df_node_features, df_total_edge_features, edge_index):
        # print(f"df_node_features : {df_node_features.shape}")
        # print(f"df_node_features : {df_node_features.columns}")
        # print(f"df_node_features : {df_node_features.index}")

        num_default_node = len(df_node_features)
        current_node_id = num_default_node

        codon_node_list = []
        df_node_features["is_codon_node"] = 0

        
        df_total_edge_features["base_to_codon_forward"] = 0
        df_total_edge_features["base_to_codon_backward"] = 0
        df_total_edge_features["prev_codon_to_codon_forward"] = 0
        df_total_edge_features["prev_codon_to_codon_backward"] = 0

        edge_index_src_base_to_codon = []
        edge_index_dst_base_to_codon = []

        edge_feataure_src_base_to_codon = []
        edge_feataure_dst_base_to_codon = []

        edge_index_src_prev_codon_to_codon = []
        edge_index_dst_prev_codon_to_codon = []

        edge_feataure_src_prev_codon_to_codon = []
        edge_feataure_dst_prev_codon_to_codon = []


        for i, (seq_id, row) in enumerate(df_node_features.iterrows()):
            if i % 6 == 0:
                
                row.name = f"node_{current_node_id}_codon"
                row.loc[:]=0
                row.loc["is_codon_node"]=1
                codon_node_list.append(row)

                row_edge_feature_src_base_to_codon = df_total_edge_features.iloc[i].copy()
                row_edge_feature_src_base_to_codon.loc[:] = 0
                row_edge_feature_src_base_to_codon.loc["base_to_codon_forward"] = 1
                edge_feataure_src_base_to_codon.append(row_edge_feature_src_base_to_codon)

                row_edge_feature_dst_base_to_codon = df_total_edge_features.iloc[i].copy()
                row_edge_feature_dst_base_to_codon.loc[:] = 0
                row_edge_feature_dst_base_to_codon.loc["base_to_codon_backward"] = 1
                edge_feataure_dst_base_to_codon.append(row_edge_feature_dst_base_to_codon)

                edge_index_src_base_to_codon.append(i)
                edge_index_dst_base_to_codon.append(current_node_id)

                if current_node_id > num_default_node:
                    row_edge_feataure_src_prev_codon_to_codon = df_total_edge_features.iloc[i].copy()
                    row_edge_feataure_src_prev_codon_to_codon.loc[:] = 0
                    row_edge_feataure_src_prev_codon_to_codon.loc["prev_codon_to_codon_forward"] = 1
                    edge_feataure_src_prev_codon_to_codon.append(row_edge_feataure_src_prev_codon_to_codon)

                    row_edge_feataure_dst_prev_codon_to_codon = df_total_edge_features.iloc[i].copy()
                    row_edge_feataure_dst_prev_codon_to_codon.loc[:] = 0
                    row_edge_feataure_dst_prev_codon_to_codon.loc["prev_codon_to_codon_backward"] = 1
                    edge_feataure_dst_prev_codon_to_codon.append(row_edge_feataure_dst_prev_codon_to_codon)

                    edge_index_src_prev_codon_to_codon.append(current_node_id - 1)
                    edge_index_dst_prev_codon_to_codon.append(current_node_id)


                    # print(f"edge_feataure_src_base_to_codon : {edge_feataure_src_base_to_codon}")
                    # print(f"edge_feataure_dst_base_to_codon : {edge_feataure_dst_base_to_codon}")

                    # print(f"edge_index_src_base_to_codon : {edge_index_src_base_to_codon}")
                    # print(f"edge_index_dst_base_to_codon : {edge_index_dst_base_to_codon}")

                    # print(f"edge_feataure_src_prev_codon_to_codon : {edge_feataure_src_prev_codon_to_codon}")
                    # print(f"edge_feataure_dst_prev_codon_to_codon : {edge_feataure_dst_prev_codon_to_codon}")

                    # print(f"edge_index_src_prev_codon_to_codon : {edge_index_src_prev_codon_to_codon}")
                    # print(f"edge_index_dst_prev_codon_to_codon : {edge_index_dst_prev_codon_to_codon}")
                    # sys.exit()


                current_node_id+=1
        
        df_codon_node = pd.DataFrame(codon_node_list)
        df_node_features = df_node_features.append(df_codon_node)
        #print(f"df_node_features : {df_node_features.shape}")
        

        np_index_src = np.concatenate([edge_index_src_base_to_codon + edge_index_dst_base_to_codon + edge_index_src_prev_codon_to_codon + edge_index_dst_prev_codon_to_codon])
        np_index_dst = np.concatenate([edge_index_dst_base_to_codon + edge_index_src_base_to_codon + edge_index_dst_prev_codon_to_codon + edge_index_src_prev_codon_to_codon])
        # print(f"np_index_src : {np_index_src}")
        # print(f"np_index_dst : {np_index_dst}")
        
        # print(f"np_index_src : {np_index_src.shape}")
        # print(f"np_index_dst : {np_index_dst.shape}")

        np_codon_edge_index = np.array([np_index_src, np_index_dst])
        #print(f"np_codon_edge_index : {np_codon_edge_index.shape}")

        edge_index = np.concatenate([edge_index, np_codon_edge_index], axis=1)
        #print(f"edge_index : {edge_index.shape}")

        df_edge_feataure_codon = pd.DataFrame(edge_feataure_src_base_to_codon+edge_feataure_dst_base_to_codon+edge_feataure_src_prev_codon_to_codon+edge_feataure_dst_prev_codon_to_codon)

        #print(f"df_edge_feataure_codon : {df_edge_feataure_codon.shape}")
        #print(f"df_total_edge_features : {df_total_edge_features.shape}")

        df_total_edge_features = df_total_edge_features.append(df_edge_feataure_codon)
        

        #print(f"df_total_edge_features : {df_total_edge_features.shape}")
        #print(f"df_total_edge_features : {df_total_edge_features.columns}")
        #print(f"df_total_edge_features : {df_total_edge_features.index}") 

        #sys.exit()

        return df_node_features, df_total_edge_features, edge_index
    

class MapE2NxN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MapE2NxN, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print(f"mape2nn init:{x.shape}")
        x = self.linear1(x)
        #print(f"mape2nn l1:{x.shape}")
        x = self.dropout(x)
        #print(f"mape2nn dr:{x.shape}")
        x = self.linear2(x)
        #print(f"mape2nn l2:{x.shape}")
        return x



class MyDeeperGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features,num_classes,
                node_hidden_channels = 128, #* 2,
                edge_hidden_channels = 16,
                hidden_channels3=32,
                num_layers = 10,
                ):
        super(MyDeeperGCN, self).__init__()

        self.node_encoder = ChebConv(num_node_features, node_hidden_channels, K=5)
        self.edge_encoder = nn.Linear(num_edge_features, edge_hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = NNConv(node_hidden_channels, node_hidden_channels,
                          MapE2NxN(edge_hidden_channels,
                                   node_hidden_channels * node_hidden_channels,
                                   hidden_channels3))
            norm = nn.LayerNorm(node_hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+',
                                 dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = nn.Linear(node_hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.1)

        self.postition = PositionEncode(node_hidden_channels)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(node_hidden_channels, 8, 256, dropout=0.2, activation='relu'),
            2
        )

    def forward(self, input_X_list):

        node_features = input_X_list[0]
        pred_len = input_X_list[1][0]
        input_len = input_X_list[2][0]
        edge_index_batchdata = input_X_list[3]
        edge_features_batchdata = input_X_list[4]
        batch_size = node_features.shape[0]
        # print(f"node_features : {node_features.shape}")
        # print(f"edge_index_batchdata : {edge_index_batchdata.shape}")
        # print(f"edge_features_batchdata : {edge_features_batchdata.shape}")

        graph_batch = getGraphDataBatch(node_features, edge_index_batchdata, edge_features_batchdata)

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        # print(f"x : {x.shape}")
        # print(f"edge_index : {edge_index.shape}")
        # print(f"edge_attr : {edge_attr.shape}")

        # edge for paired nodes are excluded for encoding node
        seq_edge_index = edge_index#[:, edge_attr[:,-1] == 0]
        x = self.node_encoder(x, seq_edge_index)

        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)
        x = x.reshape(batch_size, -1, x.shape[1])
        x = x.permute(1,0,2).contiguous()
        #print(f"x.shape : {x.shape}")
        pos = self.postition(x)
        #print(f"pos.shape : {pos.shape}")
        x = x + pos
        x = self.transformer(x+pos)
        #print(f"x.shape : {x.shape}")
        x = x.permute(1,0,2).contiguous()
        x = x.reshape(-1, x.shape[-1])
        #print(f"x.shape : {x.shape}")

        
        x = self.lin(x)
        y = x.reshape(batch_size, -1, x.shape[1])[:, :pred_len, :]
        #print(y.shape)

        del graph_batch, x
        # torch.cuda.empty_cache()
        # gc.collect()

        return y

    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score

class Graph_DNN(nn.Module):
    def __init__(self,
                num_node_features,
                num_target,
                seq_dropout=0.2, 
                hidden_dim=128 * 2, 
                hidden_layers=3, 
                emb_dropout=0.2,
                ):
        
        super(Graph_DNN, self).__init__()
        
        
        self.seq_dropout = seq_dropout
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers


        self.save_param_epoch_=0
        self.save_param_val_score_=0

        self.emb_dropout = emb_dropout #nn.Dropout(emb_dropout)

        self.num_node_features = num_node_features

        self.conv1 = GCNConv(self.num_node_features, self.hidden_layers)
        self.conv2 = GCNConv(self.hidden_layers, num_target)
        

    

    def forward(self, input_X_list):
        node_features = input_X_list[0]
        pred_len = input_X_list[1][0]
        input_len = input_X_list[2][0]
        edge_index_batchdata = input_X_list[3]
        edge_features_batchdata = input_X_list[4]

        graph_batch = getGraphDataBatch(node_features, edge_index_batchdata, edge_features_batchdata)
        

        x = self.conv1(graph_batch.x, graph_batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, graph_batch.edge_index)
        
        x = x.reshape(-1, input_len, x.shape[1])
        y = x[:, :pred_len, :]

        
               

        return y

    
    def setSaveParams(self, epoch, val_score):
        self.save_param_epoch_=epoch
        self.save_param_val_score_=val_score





def check_graph(graph_data):
    print("**** show graph proparty ****")
    print(f"structure of graph : {graph_data}")
    print(f"key of graph : {graph_data.keys}")
    print(f"number of node : {graph_data.num_nodes}")
    print(f"number of edges : {graph_data.num_edges}")
    print(f"number of node features : {graph_data.num_node_features}")
    print(f"contain isolation node : {graph_data.contains_isolated_nodes()}")
    print(f"contain self loops : {graph_data.contains_self_loops()}")
    print()
    print("===== fratures of node : x =====")
    print(graph_data["x"])
    print("===== labels of node : y =====")
    print(graph_data["y"])
    print("===== structure of edge =====")
    print(graph_data["edge_index"])

def show_graph(graph_data, ppath_to_save_dir=PATH_TO_GRAPH_DIR):
    # networkxのグラフに変換
    nxg = to_networkx(graph_data)

    # 可視化のためのページランク計算
    pr = nx.pagerank(nxg)
    pr_max = np.array(list(pr.values())).max()

    # 可視化する際のノード位置
    draw_pos = nx.spring_layout(nxg, seed=0)
    


    # ノードの色設定
    cmap = plt.get_cmap('tab10')
    labels = graph_data.y.numpy()
    print(labels)
    colors = [cmap(l) for l in labels]
    print(colors)
    
    # 図のサイズ
    plt.figure(figsize=(10, 10))

    # 描画
    nx.draw_networkx_nodes(nxg, 
                        draw_pos,
                        node_size=[v / pr_max * 1000 for v in pr.values()],
                        node_color=colors, alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

    plt.title('KarateClub')
    path_to_save = os.path.join(str(ppath_to_save_dir), datetime.now().strftime("%Y%m%d%H%M%S") + "_graph.png")
    plt.savefig(path_to_save)
    plt.show(block=False) 
    plt.close()

class GCN(torch.nn.Module):
    def __init__(self, input_node_features, output_num_classes):
        super(GCN, self).__init__()
        hidden_size = 5
        self.conv1 = GCNConv(input_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def bpps():

    a = np.load(INPUT_DIR/"bpps/id_fff546103.npy")
    print(a)
    sys.exit()

    

def test_gcn():

    bpps()

    dataset = KarateClub()

    print("グラフ数:", len(dataset))  # グラフ数: 1
    print("クラス数:",dataset.num_classes)  # クラス数: 2

    data = dataset[0]  # 1つめのグラフ
    check_graph(data)
    #show_graph(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルのインスタンス生成
    model = GCN(dataset.num_node_features, dataset.num_classes)
    # print(model)

    # モデルを訓練モードに設定
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # learnig loop
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        #print(f"out : {out}, y : {data.y.shape}")
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    
    model.eval()

    # 推論
    _, pred = model(data).max(dim=1)

    print("結果：", pred)
    print("真値：", data["y"])



def test_gcn1():

    # ノード
    src = [0, 1, 2]  # 送信側
    dst = [1, 2, 1]  # 受信側

    # エッジ
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ノードの特徴量
    x0 = [1, 2]
    x1 = [3, 4]
    x2 = [5, 6]
    x = torch.tensor([x0, x1, x2], dtype=torch.float)

    # ラベル
    y0 = [1]
    y1 = [0]
    y2 = [1]
    y = torch.tensor([y0, y1, y2], dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index)
    check_graph(data)
    show_graph(data)





if __name__ == '__main__':
    test_gcn()