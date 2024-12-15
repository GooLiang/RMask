import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn import Parameter
from model_GAMLP import R_GAMLP

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
class SGA(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, hidden, nclass, n_layers, input_dropout, dropout):
        super(SGA, self).__init__()

        self.global_w = nn.Parameter(torch.Tensor(nfeat, 1))
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.global_w, gain=gain)
        self.input_drop = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = FeedForwardNet(
                    nfeat, hidden, nclass, n_layers, dropout
                )

    def forward(self, feature_list):
        feature_list = [self.input_drop(feature) for feature in feature_list]
        # # # ################share w ################################################################
        h = torch.stack(feature_list, dim=1)
        #global_vector = torch.softmax(torch.sigmoid(self.w(h)).squeeze(), dim=-1)
        global_vector = torch.softmax(torch.sigmoid(torch.matmul(h, self.global_w)).squeeze(2), dim=-1)   #global_vector = torch.softmax(torch.sigmoid(torch.matmul(self.input_drop(h), self.global_w)).squeeze(2), dim=-1)
        output_r = 0
        for i, hidden in enumerate(feature_list):
            output_r = output_r + hidden.mul(self.dropout(global_vector[:, i].unsqueeze(1)))
        ###########################################################################################
        output_r = self.output(output_r)
        return output_r
    
###SIGN###
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

###from https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py###
class SIGN(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden,
        out_feats,
        num_hops,
        n_layers,
        dropout,
        input_drop,
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()
###SIGN###


###GBP###
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

class GnnBP(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, bias):
        super(GnnBP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x

###GBP###


###SSGC-MLP
class SSGC_MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SSGC_MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        # self.input_drop = nn.Dropout(0.1)
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        # x = self.input_drop(x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
###SSGC-MLP


class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def get_model(args, model_opt, nfeat, nclass, degree, ff_layer, input_dropout, nhid=64, dropout=0, use_weight=True, cuda=True, device = "cuda:0"):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC" or model_opt == "SSGC" or model_opt == 'SGA_SSGC' or model_opt == 'RW_SSGC':
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == "SGA_SSGC" or model_opt == "RW_SIGN" and use_weight:
        print("using SGA")
        model = SGA(nfeat=nfeat,
                    hidden=nhid,
                    nclass=nclass,
                    n_layers = ff_layer,
                    input_dropout = input_dropout,
                    dropout = dropout)
    elif model_opt == 'SIGN' or model_opt == 'SGA_SIGN' or model_opt == 'RW_SIGN':
        print("using SIGN")
        model = SIGN(
            in_feats=nfeat,
            hidden=nhid,
            out_feats=nclass,
            num_hops = degree,
            n_layers = ff_layer,
            dropout=dropout,
            input_drop=input_dropout,
        )
    elif model_opt == 'GBP' or model_opt == 'RW_GBP':
        print("in GnnBP")
        model = GnnBP(nfeat=nfeat,
            nlayers=ff_layer,
            nhidden=nhid,
            nclass=nclass,
            dropout=dropout,
            bias = args.bns)
    elif model_opt =='GAMLP' or model_opt == 'RW_GAMLP':
        model = R_GAMLP(nfeat, nhid, nclass,args.degree+1,
                 dropout, args.input_dropout,args.att_drop,args.alpha, args.ff_layer,args.act,args.pre_process,args.residual,args.pre_dropout,args.bns)
    elif model_opt == 'RW_SSGC_large':
        model = SSGC_MLP(nfeat, nhid, nclass,
                ff_layer, dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    model.to(device)
    return model
