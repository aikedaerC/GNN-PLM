import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.nn.conv import GatedGraphConv

import src.process as process
import src.prepare as prepare


torch.manual_seed(2020)

def encode_input(text, tokenizer):
    max_length = 512
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask

def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):
    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        self.bn1 = nn.BatchNorm1d(conv1d_1.out_channels)
        self.bn2 = nn.BatchNorm1d(conv1d_2.out_channels)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1) 
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.bn1(self.conv1d_1(concat)))) # (8, 205, 303) -> (8, 50, 301) -> (8,50,150) : (d_in-stride*N)=k_size; d_out = N + 1;
        Z = self.mp_2(self.bn2(self.conv1d_2(Z))) # (8,20,76)

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1]) # (1640, 200) -> (8, 205, 200)

        Y = self.mp_1(F.relu(self.bn1(self.conv1d_1(hidden))))
        Y = self.mp_2(self.bn2(self.conv1d_2(Y))) # (8,20,50)

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)


        res = self.fc1(Z) * self.fc2(Y)
        # res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Devign(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(Devign, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        
        self.count_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)
        x = self.conv(x, data.x)
        return x
    
    def load(self):
        self.load(self.path)

    def save(self):
        self.save(self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")

class BertGGCN(nn.Module):
    def __init__(self, CONFIG):
        super(BertGGCN, self).__init__()
        self.config = CONFIG
        self.gated_graph_conv_args = CONFIG.bertggcn.model.gated_graph_conv_args
        self.conv_args = CONFIG.bertggcn.model.conv_args
        self.emb_size = CONFIG.bertggcn.model.emb_size
        self.model_dir = CONFIG.bertggcn.model.model_dir
        self.device = CONFIG.device

        self.ggc = GatedGraphConv(**self.gated_graph_conv_args).to(self.device)
        self.conv = Conv(**self.conv_args,
                         fc_1_size=self.gated_graph_conv_args["out_channels"] + self.emb_size,
                         fc_2_size=self.gated_graph_conv_args["out_channels"]).to(self.device)
        # self.k = 0.1
        self.k = nn.Parameter(torch.tensor(0.1))  # Learnable k for fusion
        self.nb_class = 1

        # pretrained_weights = RobertaModel.from_pretrained(self.model_dir).embeddings.word_embeddings.weight

        # self.embed = nn.Embedding.from_pretrained(pretrained_weights,
                                                #   freeze=True,
                                                #   padding_idx=1)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.bert_model = RobertaModel.from_pretrained(self.model_dir).to(self.device)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, self.nb_class).to(self.device)

        self.count_parameters()

    def forward(self, gnn_data):
        import pdb;pdb.set_trace()
        gnn_data.x = self.reduce_embedding(gnn_data)
        x, edge_index, text = gnn_data.x, gnn_data.edge_index, gnn_data.func
        x = self.ggc(x, edge_index) # [batch_size*205, 200]
        x = self.conv(x, gnn_data.x)

        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))
        cls_prob = torch.sigmoid(cls_logit).view(-1)  # Ensure output is in the range [0, 1]
        sig = x * self.k + cls_prob * (1 - self.k)
        # sig = torch.sigmoid(pred)
        return sig


    def reduce_embedding(self, data):
        linear_layer = nn.Linear(data.x.size(1), 101).to(self.device)
        reduced_embedding = linear_layer(data.x)
        return reduced_embedding
    
    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")


class VulBertaGGCN(nn.Module):
    def __init__(self, CONFIG):
        super(VulBertaGGCN, self).__init__()
        self.config = CONFIG
        self.gated_graph_conv_args = CONFIG.vulberta.model.gated_graph_conv_args
        self.model_dir = CONFIG.vulberta.model.model_dir
        self.emb_size = CONFIG.vulberta.model.emb_size 
        self.batch_size = CONFIG.process.batch_size
        self.nodes_dim = CONFIG.embed.nodes_dim
        self.device = CONFIG.device
        self.ggc = GatedGraphConv(**self.gated_graph_conv_args).to(self.device)

        pretrained_weights = RobertaModel.from_pretrained(self.model_dir).embeddings.word_embeddings.weight

        self.embed = nn.Embedding.from_pretrained(pretrained_weights,
                                                  freeze=True,
                                                  padding_idx=1).to(self.device)
        self.conv0 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1) 

        self.conv1 = nn.Conv1d(in_channels=self.emb_size, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=self.emb_size, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=self.emb_size, out_channels=200, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200*4,256) #500
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,2) 

        self.count_parameters()


    def forward(self, data):
        # graph features
        x, edge_index, input_ids = data.x, data.edge_index, data.input_ids
        g = self.ggc(x, edge_index) # [bs*205, 200]  [26240, 200])   
        g = g.view(self.batch_size, self.nodes_dim, -1)
        g = g.permute(0,2,1)

        # plm embedding features
        input_ids = input_ids.view(self.batch_size, -1) # 1024 tokenizer max length
        x = self.embed(input_ids) # 131072,768
        x = x.permute(0,2,1)      # 128, 768, 1024

        x0 = F.relu(self.conv0(g))
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        x0 = F.max_pool1d(x0, x0.shape[2])
        x1 = F.max_pool1d(x1, x1.shape[2])
        x2 = F.max_pool1d(x2, x2.shape[2])
        x3 = F.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x0,x1,x2,x3],dim=1)
        x = x.flatten(1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # sig = torch.sigmoid(torch.flatten(logits))
        return logits
    

    def reduce_embedding(self, data):
        linear_layer = nn.Linear(data.x.size(1), 101).to(self.device)
        reduced_embedding = linear_layer(data.x)
        return reduced_embedding
    
    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")

