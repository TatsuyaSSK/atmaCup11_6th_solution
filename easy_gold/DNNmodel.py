
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from torch_geometric.nn import GCNConv, AGNNConv, ChebConv, NNConv, DeepGCNLayer
#from torch_geometric.data import Data, Batch
#from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
#from torch_geometric.utils import to_networkx

#import networkx as nx

print(torch.__version__)

def set_seed_torch(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed_torch(5)


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
        
        
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred, weight=None):
        # print(f"y_true : {y_true}")
        # print(f"y_pred : {y_pred}")
        # print(f"weight : {weight}")

        if weight is not None:
            loss = torch.sqrt(torch.mean(weight * (y_pred - y_true)**2) + self.eps)

        else:

            loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        #print(f"loss : {loss}")
        #sys.exit()
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, use_as_eval=False, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.use_as_eval = use_as_eval
        self.num_scored = num_scored
        self.weight=None

    def forward(self, y_true, y_pred):

        # score = 0
        # for i in range(self.num_scored):
        #     score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        # return score
        #if (self.use_as_eval==False):
            #print(f"y_true : {y_true}")
            #print(f"y_pred : {y_pred}")
        self.weight = y_true[..., -1]

        b_size = y_true.shape[0]
        total_score = 0
        for b in range(b_size):
            score = 0
            for i in range(self.num_scored):
                score += self.rmse(y_true=y_true[b, :, i], y_pred=y_pred[b, :, i], weight=self.weight[b, :]) / self.num_scored
            total_score += score / b_size

        if self.use_as_eval:
            total_score = total_score.detach().item()

        return total_score
        
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


        
        self.max_seq = dataset_params["max_seq"]
        self.pad_num = dataset_params["pad_num"]
        self.sequence_features_list = dataset_params["sequence_features_list"]
        self.continuous_features_list = dataset_params["continuous_features_list"]
        self.label_col = dataset_params["label_col"]
        self.row_index_name = df_train_X.index.name

        
        if train_flag:
            df_train_X[self.label_col]=df_train_y #[self.label_col]
        else:
            df_train_X[self.label_col]=0

        
        grp = df_train_X.reset_index().groupby("user_id", sort=False).tail(self.max_seq)
        
        grp_user = grp.groupby("user_id", sort=False)

        self.x_data=[]
        self.x_data_conti=[]
        self.y_data = []
        self.row_id_data = []
        self.padding_data = []

        agg_dict = {
            self.row_index_name:list,
            self.label_col:list, 
        }
        for col in self.sequence_features_list:
            agg_dict[col] = list
        for col in self.continuous_features_list:
            agg_dict[col] = list

        # for idx, row in grp_user.agg({
        #     self.row_index_name:list,
        #     "content_id":list, 
        #     self.label_col:list, 
        #     "task_container_id":list, 
        #     "part":list, 
        #     "prior_question_elapsed_time":list,
        #     }).reset_index().iterrows():
        for idx, row in grp_user.agg(agg_dict).reset_index().iterrows():


            num_elements = len(row["content_id"])

  
            np_data_list = []
            np_data_list_conti = []
            if num_elements >= self.max_seq:

                for col in self.sequence_features_list:

                    seq_l = row[col]
                    np_data_list.append(np.array(seq_l))

                for col in self.continuous_features_list:

                    conti_l = row[col]
                    np_data_list_conti.append(np.array(conti_l))

                row_id = row[self.row_index_name]
                self.row_id_data.append(np.array(row_id))

                pad_l = [False]*self.max_seq
                self.padding_data.append(np.array(pad_l))
                
                label_l=row[self.label_col]
                self.y_data.append(np.array(label_l, dtype=float))

                

            else:
                # we have to pad...
                num_padding = self.max_seq-num_elements
                padding = [self.pad_num]*num_padding

                for col in self.sequence_features_list:

                    seq_l = row[col]+padding
                    np_data_list.append(np.array(seq_l))


                for col in self.continuous_features_list:

                    conti_l = row[col]+padding
                    np_data_list_conti.append(np.array(conti_l))

                row_id = row[self.row_index_name]+ [-1]*num_padding
                self.row_id_data.append(np.array(row_id))

                pad_l = [False]*num_elements + [True]*num_padding
                self.padding_data.append(np.array(pad_l))

                
                label_l=row[self.label_col] #+ padding
                self.y_data.append(np.array(label_l, dtype=float))


            #np_data = np.stack(np_data_list)
            self.x_data.append(np_data_list)
            self.x_data_conti.append(np_data_list_conti)

            # if 26262584 in row_id:
            #     print(row_id)
            #     import pdb; pdb.set_trace()


           


       

        #import pdb; pdb.set_trace()
        



    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        # x_list = []

        # for col in self.sequence_features_list:
        #     x_list.append(self.x_data[idx][col])

        # y= self.data[idx]["answered_correctly"]

        return self.x_data[idx], self.x_data_conti[idx], self.row_id_data[idx], self.padding_data[idx], self.y_data[idx]




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
        




class Transformer_DNN(nn.Module):
    def __init__(self,
                ninp:int,
                nhead:int, 
                nhid:int, 
                nlayers:int,
                emb_dim_pairs_list,
                num_continuout_features,
                dropout:float =0.5,
                ):
        
        super(Transformer_DNN, self).__init__()

        #self.current_row_id=0

        self.ninp = ninp

        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model=ninp, 
                                                 nhead=nhead, 
                                                 dim_feedforward=nhid, 
                                                 dropout=dropout, 
                                                 activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.emb_layers = nn.ModuleList([nn.Embedding(m, ninp) for m, _ in emb_dim_pairs_list])
        self.categorical_proj = nn.Sequential(
            nn.Linear(ninp*len(emb_dim_pairs_list), ninp//2),
            nn.LayerNorm(ninp//2),
        )  


        self.conti_embed_layers = nn.ModuleList([nn.Linear(1,ninp,bias=False) for n in range(num_continuout_features)])
        self.continuous_proj = nn.Sequential(
            nn.Linear(ninp*num_continuout_features, ninp//2),
            nn.LayerNorm(ninp//2),
        )
        
        
        self.postition = PositionEncode(self.ninp)

        self.decoder = nn.Linear(ninp, 1)
        self.out_act = nn.Sigmoid()

        self.layer_normal = nn.LayerNorm(self.ninp)
        self.dropout = nn.Dropout(p=dropout)

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

        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

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

            #print(f"embedded_src : {embedded_src.shape}")


            embedded_src=embedded_src.transpose(0, 1)

            #import pdb; pdb.set_trace()

        #print(f"embedded_src : {embedded_src.shape}")



        #input_emb = embedded_src * np.sqrt(self.ninp)

        embedded_src += self.postition(embedded_src)
        input_emb = self.dropout(embedded_src)
        input_emb = self.layer_normal(input_emb)

        

        # row_id = src[-2]
        src_key_padding_mask = src[-1]

        src_mask = self.generate_square_subsequent_mask(input_emb.size(0)).cuda()
        output = self.transformer_encoder(src=input_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #output = input_emb
        #print(f"out enc: {output}")
        output = self.decoder(output)
        #print(f"out dec: {output}")

        output = self.out_act(output)
        #print(f"out act: {output}")

        output = output.transpose(1, 0)
        #print(f"out transpose: {output.shape}")

        output = output[~src_key_padding_mask]
        #print(f"out mask: {output.shape}")


        #self.current_row_id = row_id[~src_key_padding_mask]

        #import pdb; pdb.set_trace()
        #print(output)
            
        
        return output

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