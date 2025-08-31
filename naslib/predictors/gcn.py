



import itertools
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from naslib.utils import AverageMeterGroup
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform




def normalize_adj(adj):
    
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.linear = nn.Linear(out_features * 2, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        
        out = torch.cat((output1, output2), dim=2)
        out = self.ln(self.linear(out))
        
        out = self.dropout(out)
        return out

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class NeuralPredictorModel(nn.Module):
    def __init__(
            self, initial_hidden=-1, gcn_hidden=128, gcn_layers=1, linear_hidden=256, weighted_pooling = False, device="cpu"
    ):
        super().__init__()
        self.gcn = [
            DirectedGraphConvolution(
                initial_hidden if i == 0 else gcn_hidden, gcn_hidden
            )
            for i in range(gcn_layers)
        ]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)
        self.device = device
        self.weighted_pooling = weighted_pooling

    def forward(self, inputs):
        numv, adj, out = (
            inputs["num_vertices"],
            inputs["adjacency"],
            inputs["operations"],
        )
        adj = adj.to(self.device)
        numv = numv.to(self.device)
        out = out.to(self.device)
        gs = adj.size(1)  

        adj_with_diag = normalize_adj(
            adj + torch.eye(gs, device=adj.device)
        )  
        for layer in self.gcn:
            out = layer(out, adj_with_diag)

        if self.weighted_pooling:
            def weighted_graph_pooling(out, weights):
                
                weighted_features = out * weights.unsqueeze(-1)  
                graph_representation = weighted_features.sum(dim=1) / weights.sum(dim=1, keepdim=True)
                return graph_representation

            
            weights = torch.ones(out.shape[0], out.shape[1]).to(self.device)
            weights[:, -1] = 10  
            graph_representation = weighted_graph_pooling(out, weights)
            out = self.fc1(graph_representation)
        else:
            out = graph_pooling(out, numv)
        out = F.gelu(out)
        
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out









































































































































class GCNPredictor(Predictor):
    def __init__(self, encoding_type="gcn", ss_type=None, hpo_wrapper=False, config = None, device = "cuda:0"):
        self.encoding_type = encoding_type
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {
            "gcn_hidden": 144,
            "batch_size": 7,
            "lr": 1e-4,
            "wd": 3e-4,
        }
        
        
        self.hyperparams = None
        self.config = config
        self.device = device
        
        
        
        
        
        
        
        
        
    def get_model(self, **kwargs):
        if self.ss_type == "nasbench101":
            initial_hidden = 5
        elif self.ss_type == "nasbench201":
            initial_hidden = 7
        elif self.ss_type == "nasbench301":
            initial_hidden = 9
        elif self.ss_type == "nlp":
            initial_hidden = 8
        elif self.ss_type == "transbench101":
            initial_hidden = 7
        else:
            raise NotImplementedError()
        predictor = NeuralPredictorModel(initial_hidden=initial_hidden, gcn_hidden=self.config.model_fit_para.gcn_hidden,
                                         gcn_layers=self.config.model_fit_para.gcn_layers,
                                         linear_hidden=self.config.model_fit_para.mlp_hidden, weighted_pooling = self.config.model_fit_para.weighted_pooling, device=self.device)
        return predictor

    def fit(self, xtrain, ytrain, train_info=None, epochs=500):

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        gcn_hidden = self.hyperparams["gcn_hidden"]
        batch_size = self.hyperparams["batch_size"]
        lr = self.hyperparams["lr"]
        wd = self.hyperparams["wd"]

        
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        

        ytrain_normed = (ytrain - self.mean) / self.std
        
        train_data = []
        for i, arch in enumerate(xtrain):
            encoded = arch.encode(encoding_type=self.encoding_type)
            encoded["val_acc"] = float(ytrain_normed[i])
            train_data.append(encoded)
        train_data = np.array(train_data)

        self.model = self.get_model(gcn_hidden=gcn_hidden)
        data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for _ in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            for _, batch in enumerate(data_loader):
                target = batch["val_acc"].float().to(self.device)
                prediction = self.model(batch)
                loss = criterion(prediction, target)
                
                if True:  
                    m = 0.1
                    
                    '''
                    y = list(map(lambda y_i: 1 if y_i == True else -1, graph_batch.y[0: -1] > graph_batch.y[1:]))
                    pairwise_ranking_loss = torch.nn.HingeEmbeddingLoss(margin=m)(pred[0:-1] - pred[1:],
                                                                                  target=torch.from_numpy(np.array(y)))
                    '''
                    pairwise_ranking_loss = []
                    sort_idx = torch.argsort(target, descending=True)
                    for idx, idx_y_i in enumerate(sort_idx):
                        for idx_y_i_p1 in sort_idx[idx + 1:]:
                            pairwise_ranking_loss.append(torch.max(torch.tensor(0.0, dtype=torch.float),
                                                                   m - (prediction[idx_y_i] - prediction[idx_y_i_p1])))
                    pairwise_ranking_loss = torch.mean(torch.stack(pairwise_ranking_loss))

                    loss += pairwise_ranking_loss

                loss.backward()
                optimizer.step()
                mse = accuracy_mse(prediction, target)
                meters.update(
                    {"loss": loss.item(), "mse": mse.item()}, n=target.size(0)
                )

            lr_scheduler.step()
        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=1000):
        test_data = np.array(
            [
                arch.encode(encoding_type=self.encoding_type)
                for arch in xtest
            ]
        )
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return pred * self.std + self.mean

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                "gcn_hidden": int(loguniform(64, 200)),
                "batch_size": int(loguniform(5, 32)),
                "lr": loguniform(0.00001, 0.1),
                "wd": loguniform(0.00001, 0.1),
            }

        self.hyperparams = params
        return params
