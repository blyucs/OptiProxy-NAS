import logging
import math
import pickle

import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import importlib
import matplotlib.pyplot as plt
from naslib.utils import scatter_plot
from ops import Ops_Model

from adj_ops_301 import Adj_Ops_Model

from adj import Adj_Model
from naslib.utils.encodings import EncodingType
from naslib import utils
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace,
)
from naslib.predictors.surrogate_lib import (
    GCNPredictor,
)
from naslib.search_spaces.nasbench201.conversions import *
from naslib.search_spaces.HW_NAS_Bench.hw_nas_bench_api import HWNASBenchAPI as HWAPI
from naslib.search_spaces.HW_NAS_Bench.nasbench201 import api
from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_str

class Search_Model(torch.nn.Module):

    def __init__(self, logger, config, seed, dataset_api, search_device, surrogate_device):
        super(Search_Model, self).__init__()

        self.search_device = search_device
        self.surrogate_device = surrogate_device
        self.config = config
        self.dataset_api = dataset_api
        self.search_space = self.get_search_space()
        self.surrogate_model_init()

        if self.config.search_space == "nasbench101":
            self.ops_model = Ops_Model(config=config, device=self.search_device)  
            self.adj_model = Adj_Model(config, dataset_api=self.dataset_api, logger=logger,
                                       device=self.search_device)  
            self.encoding = importlib.import_module('naslib.search_spaces.nasbench101.encodings')
            self.encode_func = self.encoding.encode_gcn
        elif self.config.search_space == "nasbench201":
            self.ops_model = Ops_Model(config=config, device=self.search_device)  
            self.encoding = importlib.import_module('naslib.search_spaces.nasbench201.encodings')
            self.encode_func = self.encoding.encode_gcn_nasbench201
        elif self.config.search_space == "nasbench301":
            self.adj_ops_model = Adj_Ops_Model(config, dataset_api=self.dataset_api, logger=logger,
                                               device=self.search_device)  
            self.encoding = importlib.import_module('naslib.search_spaces.nasbench301.encodings')
            self.encode_func = self.encoding.encode_gcn

        self.dataset = config.dataset
        

        self.gcn_criterion = nn.MSELoss().to(self.surrogate_device)
        self.fit_batch_size = config.model_fit_para.batch_size
        self.ranking_loss_reg_coe = config.model_fit_para.ranking_loss_reg_coe
        
        self.lr_beta = config.search_para.lr_beta
        
        self.model_epochs = config.model_fit_para.model_epochs
        self.search_epochs = config.search_para.search_epochs
        self.input_batch_size = config.search_para.input_batch_size
        self.num_vertices = config.num_vertices
        self.logger = logger
        self.gcn_history_spec_dict = []
        self.num_sample_each_step = config.search_para.num_sample_each_step
        self.verify_ratio = config.search_para.verify_ratio
        self.mean = np.float64(config.mean)
        self.std = np.float64(config.std)
        self.latency_mean = np.float64(config.latency_mean)
        self.latency_std = np.float64(config.latency_std)
        self.object_scaler = torch.Tensor(self.config.search_para.object_scaler).to(self.surrogate_device)
        self.object_scaler_step = torch.tensor(self.config.search_para.object_scaler_step).to(self.surrogate_device)
        self.seed = seed
        
        self.accuracies = []
        self.best_acc = 0.0
        self.best_acc_array = [0]
        self.gcn_history_dict = np.array([])
        self.gcn_history_spec_dict = []
        self.steps = []
        self.best_compact = []
        self.arch_compact = torch.Tensor([])
        self.over_latency = 0
        self.arch_pred = torch.Tensor([])
        self.max_flops = self.config.condition.max_flops
        self.max_params = self.config.condition.max_params
        self.max_latency = self.config.condition.max_latency
        self.hw_api = HWAPI("./naslib/search_spaces/HW_NAS_Bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
        
        nb_201_index_file = open("./naslib/search_spaces/HW_NAS_Bench/nb201_arch_to_index.pickle",'rb')
        self.nasbench_api = pickle.load(nb_201_index_file)
        self.prepare_dataset()

    def prepare_dataset(self):
        dataset_name = {'cifar10': 'cifar10-valid', 'cifar100': 'cifar100', 'ImageNet16-120': 'ImageNet16-120'}
        self.dataapi_name = dataset_name[self.dataset]
        f = open(
            "./naslib/data/nb201_"+str(self.dataapi_name)+f"_val_test_{self.config.dataset_ver}.pickle",
            "rb")
        self.data_metric = pickle.load(f)
        filted_dict = {}
        for k,v in self.data_metric.items():
            if isinstance(k, tuple):
                filted_dict[k] = v
        self.data_metric = filted_dict
        batch = {}
        import itertools
        specs = torch.Tensor(list(itertools.product(range(1, 6), repeat=6)))
        specs = torch.cat((torch.ones(15625, 1) * 0, specs), dim=1)
        specs = torch.cat((specs, torch.ones(15625, 1) * 6), dim=1)
        one_hot_mat = torch.nn.functional.one_hot(torch.Tensor(specs).long(), 7)
        adj = torch.tensor([[0., 1., 1., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 0.]]).repeat(15625, 1, 1)
        num_vertices = torch.ones(15625) * 8
        batch['num_vertices'] = num_vertices.float().cuda()
        batch['adjacency'] = adj.float().cuda()
        batch['operations'] = one_hot_mat.float().cuda()
        arch_dict = {}
        specs = torch.Tensor(list(itertools.product(range(0, 5), repeat=6)))
        pareto_front = []
        for i in range(len(specs)):
            one_dic = {}
            one_dic['num_vertices'] = np.array(8)
            one_dic['adjacency'] = adj[i].float().numpy()
            one_dic['operations'] = one_hot_mat[i].float().numpy()
            one_dic['operations'] = one_hot_mat[i].float().numpy()
            
            one_dic['val_acc'] = self.query_valid_acc(tuple(specs[i].int().numpy()), -2)
            arch_dict[tuple(specs[i].int().tolist())] = one_dic
        
        for i in range(len(specs)):
            k = tuple(specs[i].int().tolist())
            latency = self.query_hw_latency(k)
            pareto_front.append((-self.data_metric[k][self.dataapi_name]['eval_acc1es'][-2],
                                 latency))
            arch_dict[k]['latency'] = latency
            self.data_metric[k]['arch_encode'] = arch_dict[k]

        pareto_front = np.array(tuple(pareto_front))
        self.std = pareto_front[:,0].std()
        self.mean = -pareto_front[:,0].mean()
        self.latency_std = pareto_front[:,1].std()
        self.latency_mean = pareto_front[:,1].mean()
        pareto_front_index = self.pareto(pareto_front, return_mask=False)
        m = 0
        self.pareto_front = {str(tuple(specs[i].int().numpy())): pareto_front[i] for i in pareto_front_index}

        best_k = ''
        for k, v in self.data_metric.items():
            if self.query_valid_acc(k) > m and self.query_flops(k) < self.max_flops and self.query_hw_latency(
                    k) < self.max_latency and self.query_params(k) < self.max_params:
                m = self.query_valid_acc(k)
                best_k = k
        self.config.max_acc = float(m)

        self.logger.info('-------------best acc {}'.format(m))
        self.logger.info(best_k)
    def pareto(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def surrogate_model_init(self):
        supported_predictors = {
            "gcn": GCNPredictor(encoding_type=EncodingType.GCN, ss_type=self.config.search_space, hpo_wrapper=True,
                                config=self.config, device=self.surrogate_device),
        }
        predictor = supported_predictors[self.config.predictor]
        surrogate_model = predictor.get_model().to(self.surrogate_device)
        self.surrogate_model = surrogate_model
        self.gcn_optimizer = optim.AdamW(self.surrogate_model.parameters(), lr=self.config.model_fit_para.lr)
        

        self.gcn_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.gcn_optimizer,
                                                                 self.config.model_fit_para.lr_gamma)

    def latency_model_init(self):
        supported_predictors = {
            "gcn": GCNPredictor(encoding_type=EncodingType.GCN, ss_type=self.config.search_space, hpo_wrapper=True,
                                config=self.config, device=self.surrogate_device),
        }
        predictor = supported_predictors[self.config.predictor]
        surrogate_model = predictor.get_model().to(self.surrogate_device)
        self.latency_model = surrogate_model
        self.latency_optimizer = optim.AdamW(self.surrogate_model.parameters(), lr=self.config.model_fit_para.lr)
        

        self.latency_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.gcn_optimizer,
                                                                     self.config.model_fit_para.lr_gamma)

    def get_search_space(self):
        supported_search_spaces = {
            "nasbench101": NasBench101SearchSpace(),
            "nasbench201": NasBench201SearchSpace(),
            "nasbench301": NasBench301SearchSpace(),
            "nlp": NasBenchNLPSearchSpace(),
        }

        """
        If the API did not evaluate *all* architectures in the search space, 
        set load_labeled=True
        """
        
        search_space = supported_search_spaces[self.config.search_space]
        return search_space

    def compute_pairwise_ranking_loss(self, prediction, target, margin):
        
        sort_idx = torch.argsort(target, descending=True)

        sorted_predictions = prediction[sort_idx]

        pairwise_diff = sorted_predictions.unsqueeze(1) - sorted_predictions.unsqueeze(0)

        mask = torch.triu(torch.ones_like(pairwise_diff), diagonal=1).bool()

        pairwise_ranking_loss = torch.max(torch.tensor(0.0, dtype=torch.float), margin - pairwise_diff[mask])

        if pairwise_ranking_loss.numel() == 0:
            return torch.tensor(0.0)  
        else:
            return pairwise_ranking_loss.mean()

    def fit_surrogate_model(self, step_idx):
        self.surrogate_model.train()
        self.gcn_optimizer.param_groups[0]['lr'] = self.config.model_fit_para.lr
        if self.config.search_space == 'nasbench101':
            schedule = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
            
            data_list = [(data, data["val_acc"]) for data in self.gcn_history_dict]
            
            top_data_list = sorted(data_list, key=lambda x: x[1], reverse=True)[:schedule[step_idx]]
            
            top_data = [data[0] for data in top_data_list]
            
            data_loader = DataLoader(top_data, batch_size=self.fit_batch_size, shuffle=True)
        else:
            data_loader = DataLoader(self.gcn_history_dict, batch_size=self.fit_batch_size, shuffle=True)

        for gcn_step in range(self.model_epochs):
            for _, batch in enumerate(data_loader):
                self.gcn_optimizer.zero_grad()
                acc_target = batch["low_fi_acc"].float().to(self.surrogate_device)
                latency_target = batch["latency"].float().to(self.surrogate_device)
                prediction = self.surrogate_model(batch)
                loss_acc = self.gcn_criterion(prediction[:, 0], acc_target)
                loss_latency = self.gcn_criterion(prediction[:, 1], latency_target)
                loss = loss_acc + loss_latency
                if self.ranking_loss_reg_coe:
                    pairwise_ranking_loss = self.compute_pairwise_ranking_loss(prediction[:, 0], acc_target, 0.1)
                    loss += self.ranking_loss_reg_coe * pairwise_ranking_loss
                if self.ranking_loss_reg_coe:
                    pairwise_ranking_loss = self.compute_pairwise_ranking_loss(prediction[:, 1], latency_target, 0.1)
                    loss += self.ranking_loss_reg_coe * pairwise_ranking_loss
                loss.backward()
                self.gcn_optimizer.step()

            if gcn_step % self.config.model_fit_para.log_step == 0:
                self.logger.info(
                    f"Seed {self.seed}, model fit epoch: {gcn_step}, acc_loss: {loss_acc.item()}, latency_loss: {loss_latency.item()}")
            if gcn_step % self.config.model_fit_para.lr_step == 0:
                self.gcn_lr_scheduler.step()
                

        
        with torch.no_grad():
            data_loader = DataLoader(self.gcn_history_dict, batch_size=1000, shuffle=True)
            for _, data in enumerate(data_loader):
                prediction = self.surrogate_model(data).cpu().detach().numpy() * self.std + self.mean
                target = data["val_acc"] * self.std + self.mean
            fig = scatter_plot(prediction[:, 0], np.array(target), xlabel='Predicted', ylabel='True',
                               title=f'seed_{self.seed}')

            plt.axis([self.config.ylim_s, self.config.ylim_e, self.config.ylim_s, self.config.ylim_e])
            
            fig.savefig(os.path.join(self.config.save, f'pred_vs_true_seed_{self.seed}_{step_idx}.jpg'))
            

    def sample_spec_one_item_101(self, input_idx):
        sample_try_cnt = 0
        while True:
            compact = self.ops_model.sample_compact_from_alpha(input_idx)  
            cur_adj = self.adj_model.sample_descret_one_item(
                input_idx).cpu().detach().numpy()  
            ops_sample_str_list = [self.encoding.OPS[idx] for idx in compact]
            ops = [self.encoding.INPUT] + ops_sample_str_list + [self.encoding.OUTPUT]
            model_spec = self.dataset_api["api"].ModelSpec(matrix=cur_adj.astype(np.int64), ops=ops)
            if self.dataset_api["nb101_data"].is_valid(model_spec):
                break
            sample_try_cnt += 1
            if sample_try_cnt > 10 and sample_try_cnt % 10 == 0:
                self.logger.info(f"sample try count (fail): {sample_try_cnt}")

        return model_spec

    def sample_spec_101(self, input_idx):
        while True:
            
            model_spec = self.sample_spec_one_item_101(input_idx)

            exists = False
            for item in self.gcn_history_spec_dict:
                if np.array_equal(item['matrix'], model_spec.matrix) and item['ops'] == model_spec.ops:
                    exists = True
                    break
            if not exists:
                break

        return model_spec

    def sample_and_filter(self, num_samples=0, first_epoch=0):
        num_verify = num_samples * self.verify_ratio
        sample_num_each_alpha = math.ceil(num_verify / (self.input_batch_size))
        num_alpha_sample = num_samples
        gcn_spec_dict = [[] for _ in range(self.config.search_para.num_input_group)]
        gcn_encode_dict = [[] for _ in range(self.config.search_para.num_input_group)]
        top_gcn_spec_dict = []

        unique_preds = torch.Tensor().to(self.surrogate_device)

        for input_idx in range(self.input_batch_size):
            for _ in range(sample_num_each_alpha):
                if self.config.search_space == "nasbench101":
                    model_spec = self.sample_spec_101(input_idx)
                    dic = self.encoding.encode_gcn(
                        {"matrix": model_spec.matrix, "ops": model_spec.ops})
                    dic['num_vertices'] = np.array(dic['num_vertices'])
                    gcn_encode_dict.append(dic)
                    spec = {"matrix": model_spec.matrix, "ops": model_spec.ops}
                    gcn_spec_dict.append(spec)

                elif self.config.search_space == "nasbench201":
                    compact = self.ops_model.sample_compact_from_alpha(input_idx)
                    
                    while (compact in self.gcn_history_spec_dict):
                        compact = self.ops_model.sample_compact_from_alpha(input_idx)

                    dic = self.query_arch(tuple(compact))
                    gcn_encode_dict[input_idx % self.config.search_para.num_input_group].append(dic)
                    gcn_spec_dict[input_idx % self.config.search_para.num_input_group].append(compact)
                    
                elif self.config.search_space == "nasbench301":
                    compact = self.adj_ops_model.sample_compact_from_alpha_301(
                        input_idx)  
                    
                    while compact in self.gcn_history_spec_dict:
                        compact = self.adj_ops_model.sample_compact_from_alpha_301(input_idx)
                    
                    
                    dic = self.encoding.encode_gcn(compact)
                    dic['num_vertices'] = np.array(dic['num_vertices'])
                    gcn_encode_dict.append(dic)
                    gcn_spec_dict.append(compact)
                elif self.config.search_space == "nlp":
                    compact = self.adj_ops_model.sample_compact_from_alpha_nlp(
                        input_idx)  
                    
                    while compact in self.gcn_history_spec_dict:
                        compact = self.adj_ops_model.sample_compact_from_alpha_nlp(input_idx)
                    
                    
                    dic = self.encoding.encode_gcn(compact, max_nodes=self.config.search_para.num_nodes)
                    dic['num_vertices'] = np.array(dic['num_vertices'])
                    gcn_encode_dict.append(dic)
                    gcn_spec_dict.append(compact)
        for i in range(self.config.search_para.num_input_group):
            with torch.no_grad():
                data_loader = DataLoader(gcn_encode_dict[i], batch_size=1000, shuffle=False)
                predictions_list = []
                for _, data in enumerate(data_loader):
                    if self.config.search_space != "nasbench101":
                        if first_epoch:
                            prediction = torch.rand(len(data['val_acc'])) * self.std + self.mean
                        else:
                            gcn_spec_tensor = torch.Tensor(gcn_spec_dict).squeeze(0)
                            gcn_history_spec_tensor = torch.Tensor(self.gcn_history_spec_dict)
                            
                            
                            cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                            similarity = torch.cat(
                                [cossim(i, gcn_history_spec_tensor).mean().unsqueeze(-1) for i in gcn_spec_tensor])
                            
                            prediction_tensor = self.surrogate_model(data)
                            prediction = (prediction_tensor * self.object_scaler).sum(dim=1)
                            prediction[prediction_tensor[:, 1] > (self.max_latency*self.config.condition.max_latency_scaler-self.latency_mean) / self.latency_std] = 0
                            prediction = prediction - self.config.search_para.similarity_penalty_ratio * similarity.to(
                                self.surrogate_device)
                    else:
                        prediction = self.surrogate_model(data) * self.std + self.mean
                    
                    predictions_list.append(prediction)

            self.config.search_para.similarity_penalty_ratio *= self.config.search_para.similarity_penalty_decay
            
            
            predictions_tensor = torch.cat(predictions_list, dim=0).unsqueeze(0)
            min_tensor, max_model_idx = torch.min(predictions_tensor, dim=0)
            scores = min_tensor

            unique_pred = scores.unique(sorted=True, dim=0)
            unique_idx = (torch.cat([(scores == x_u).nonzero()[0] for x_u in unique_pred]))
            gcn_spec_dict[i] = np.array(gcn_spec_dict[i])[unique_idx.detach().cpu().numpy()]

            unique_pred = scores[unique_idx[-int(num_alpha_sample / self.config.search_para.num_input_group):]]
            unique_preds = torch.cat((unique_preds, unique_pred.to(self.surrogate_device)))

            per_alpha_spec_dict = gcn_spec_dict[i][
                                  -int(num_alpha_sample / self.config.search_para.num_input_group):].tolist()
            top_gcn_spec_dict += per_alpha_spec_dict
            self.gcn_history_spec_dict = self.gcn_history_spec_dict + per_alpha_spec_dict

            
            unique_preds = torch.cat((unique_preds, torch.Tensor([0]).to(unique_preds.device)))

        return top_gcn_spec_dict, unique_preds

    def query_valid_acc(self, spec, epoch=-2):
        return self.data_metric[spec][self.dataapi_name]['eval_acc1es'][epoch]

    def query_valid_acc_mean(self, spec, epoch=-2, window=3):
        return np.array(self.data_metric[spec][self.dataapi_name]['eval_acc1es'][epoch - window:epoch]).mean()

    def query_flops(self, spec):
        return self.data_metric[spec][self.dataapi_name]['cost_info']['flops']

    def query_params(self, spec):
        return self.data_metric[spec][self.dataapi_name]['cost_info']['params']

    def query_latency(self, spec):
        return self.data_metric[spec][self.dataapi_name]['cost_info']['latency']

    def query_hw_latency(self, spec):
        arch_index = self.nasbench_api[spec]
        latency = self.hw_api.query_by_index(arch_index, self.config.dataset)[self.config.hw_target_name]
        if latency > self.max_latency:
            self.over_latency += 1
        return latency

    def query_arch(self, spec):
        dic = self.data_metric[spec]['arch_encode']
        
        dic['latency'] = np.array(self.query_hw_latency(spec))
        return self.data_metric[spec]['arch_encode']

    def query_dict(self, spec_dict, acc_pred):
        for i in range(len(spec_dict)):
            spec = spec_dict[i]
            acc = self.query_valid_acc(tuple(spec), epoch=self.config.search_para.low_fi_epoch)
            latency = self.query_hw_latency(tuple(spec))
            if self.config.search_space == "nasbench101":
                self.logger.info(
                    f"Seed {self.seed}, \n spec_adj = \n {spec['matrix']}, \n spec_ops = {spec['ops']}, \n acc = {acc}, acc_pred: {acc_pred[i]}")
            else:
                self.logger.info(f"Seed {self.seed}, spec = {spec}, acc = {acc}, acc_pred: {acc_pred[i]}")

            if self.config.search_space == "nasbench101" or self.config.search_space == "nasbench301":
                dic = self.encode_func(spec)
            elif self.config.search_space == "nlp":
                dic = self.encode_func(spec, max_nodes=self.config.search_para.num_nodes)
            elif self.config.search_space == "nasbench201":
                
                dic = self.query_arch(tuple(spec))
                dic['low_fi_acc'] = ((self.query_valid_acc_mean(tuple(spec),
                                                                self.config.search_para.low_fi_epoch,
                                                                self.config.search_para.low_fi_mean_window) - self.mean) / self.std)

                dic['latency'] = ((self.query_hw_latency(tuple(spec)) - self.latency_mean) / self.latency_std)
            else:
                raise NotImplementedError()

            self.gcn_history_dict = np.hstack((self.gcn_history_dict, dic))

            if acc > self.best_acc and latency < self.max_latency:
                self.best_acc = acc
                self.logger.info(f"Seed {self.seed}, Sample count: {len(self.gcn_history_dict)}, best acc: {acc}")
                self.best_compact = spec
            if self.best_acc >= self.config.max_acc:
                print("global optimal reached")
                exit()
            self.steps.append(len(self.gcn_history_dict))
            self.accuracies.append(acc)
            self.best_acc_array.append(self.best_acc)

    def sampling_and_query_init(self):
        top_gcn_spec_dict, acc_pred = self.sample_and_filter(num_samples=self.config.search_para.num_sample_init, first_epoch=1)
        self.query_dict(top_gcn_spec_dict, acc_pred)

    def sampling_and_query(self):
        top_gcn_spec_dict, acc_pred = self.sample_and_filter(num_samples=self.config.search_para.num_sample_each_step,
                                                             first_epoch=0)
        self.over_latency = 0
        self.query_dict(top_gcn_spec_dict, acc_pred)

        if self.over_latency / self.config.search_para.num_sample_each_step > self.config.condition.over_latency_ratio:
            self.object_scaler[0] -= self.object_scaler_step
            self.object_scaler[1] += self.object_scaler_step
            print(self.object_scaler)
        else:
            self.object_scaler[0] += self.object_scaler_step
            self.object_scaler[1] -= self.object_scaler_step
            print(self.object_scaler)


    def prepare_input(self):

        if self.config.search_space == "nasbench101":
            ops_onehot_batch = self.ops_model()
            adj_batch = self.adj_model()
            
            
            
        elif self.config.search_space == "nasbench201":
            ops_onehot_batch = self.ops_model()
            matrix_base = np.array(
                [
                    [0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32
            )
            adj_batch = torch.tensor(matrix_base).repeat(self.input_batch_size, 1, 1)  
        elif self.config.search_space == "nasbench301":
            adj_batch, ops_onehot_batch = self.adj_ops_model()
        elif self.config.search_space == "nlp":
            adj_batch, ops_onehot_batch = self.adj_ops_model()
        else:
            raise NotImplementedError

        mask = torch.tensor(np.array([i < self.num_vertices for i in range(self.num_vertices)],
                                     dtype=np.float32)).repeat(self.input_batch_size, 1)  

        dic = {
            "num_vertices": torch.tensor([self.num_vertices] * self.input_batch_size),
            "adjacency": adj_batch,
            "operations": ops_onehot_batch,
            "mask": mask,
            "val_acc": torch.tensor([0.0] * self.input_batch_size),
        }
        return dic

    def search(self, step_idx):
        pred_values = []
        self.surrogate_model.eval()

        if self.config.search_space == "nasbench101":
            self.ops_model.alpha_optimizers.param_groups[0]['lr'] = self.config.search_para.lr
            self.adj_model.adj_optimizers.param_groups[0]['lr'] = self.config.search_para.lr_beta

            self.adj_model.temp_init()
            self.ops_model.temp_init()
            ops_epochs = 15  
            adj_epochs = 15  
            current_optimizer = 'ops'
            epoch_count = 0
            for arch_step in range(self.search_epochs):
                self.ops_model.alpha_optimizers.zero_grad()
                self.adj_model.adj_optimizers.zero_grad()
                input_dic = self.prepare_input()
                pred = self.surrogate_model(input_dic)
                pred_values.append(pred.sum().item())
                loss = -pred.sum()
                loss.backward(retain_graph=True)
                if current_optimizer == 'ops':
                    if epoch_count < ops_epochs:
                        self.ops_model.alpha_optimizers.step()
                        self.ops_model.temp_scheduler.step()
                        self.logger.info(f"search loss: {loss}")
                        epoch_count += 1
                    else:
                        current_optimizer = 'adj'
                        epoch_count = 0  
                else:  
                    if epoch_count < adj_epochs:
                        
                        self.adj_model.adj_optimizers.step()
                        self.adj_model.temp_scheduler.step()
                        self.logger.info(f"search pred.sum: {pred.sum()}")
                        epoch_count += 1
                    else:
                        current_optimizer = 'ops'
                        epoch_count = 0  

                
                

                # if arch_step == self.search_epochs - 1:
                #     self.logger.info(f"Seed {self.seed}, gumbel alphas:  {self.ops_model.node_attr_alpha}")
                #
                #
                #
                #
                #     self.logger.info(f"Seed {self.seed}, adj data: \n{self.adj_model.get_cur_adj()}")

        elif self.config.search_space == 'nasbench301':
            ops_epochs = 20  
            adj_epochs = 20  
            self.adj_ops_model.adj_optimizers.param_groups[0]['lr'] = self.config.search_para.lr_beta
            self.adj_ops_model.ops_optimizers.param_groups[0]['lr'] = self.config.search_para.lr_beta
            self.adj_ops_model.temp_init()
            current_optimizer = 'ops'
            epoch_count = 0
            for arch_step in range(self.search_epochs):
                self.adj_ops_model.ops_optimizers.zero_grad()
                self.adj_ops_model.adj_optimizers.zero_grad()
                input_dic = self.prepare_input()
                pred = self.surrogate_model(input_dic)
                pred_values.append(pred.sum().item())
                loss = -pred.sum()
                loss.backward(retain_graph=True)
                if current_optimizer == 'ops':
                    if epoch_count < ops_epochs:
                        self.adj_ops_model.ops_optimizers.step()
                        self.logger.info(f"search loss: {loss}")
                        epoch_count += 1
                    else:
                        
                        current_optimizer = 'adj'
                        epoch_count = 0  
                else:  
                    if epoch_count < adj_epochs:
                        
                        self.adj_ops_model.adj_optimizers.step()
                        self.logger.info(f"search pred.sum: {pred.sum()}")
                        epoch_count += 1
                    else:

                        
                        current_optimizer = 'ops'
                        epoch_count = 0  

                self.adj_ops_model.temp_scheduler.step()

                if arch_step % self.config.search_para.lr_step == 0:
                    self.adj_ops_model.adj_lr_scheduler.step()
                    self.adj_ops_model.ops_lr_scheduler.step()

                if arch_step == self.search_epochs - 1:
                    self.logger.info(f"Seed {self.seed}, \n normal adj: \n {self.adj_ops_model.normal_adj_para}, \
                      reduction adj: \n {self.adj_ops_model.reduction_adj_para}\n")
                    self.logger.info(f"Seed {self.seed}, \n normal alpha: \n {self.adj_ops_model.normal_ops_alpha}, \
                      reduction alpha: \n {self.adj_ops_model.reduction_ops_alpha}\n")

                    
                if arch_step % self.config.search_para.log_step == 0:
                    self.logger.info(f"Seed {self.seed}, arch epoch: {arch_step}, loss: {loss.item()}")
        elif self.config.search_space == 'nlp':  
            ops_epochs = 15  
            adj_epochs = 15  
            self.adj_ops_model.adj_optimizers.param_groups[0]['lr'] = self.config.search_para.lr_beta
            self.adj_ops_model.ops_optimizers.param_groups[0]['lr'] = self.config.search_para.lr_beta
            self.adj_ops_model.temp_init()
            current_optimizer = 'ops'
            epoch_count = 0
            for arch_step in range(self.search_epochs):
                self.adj_ops_model.ops_optimizers.zero_grad()
                self.adj_ops_model.adj_optimizers.zero_grad()
                input_dic = self.prepare_input()
                pred = self.surrogate_model(input_dic)
                pred_values.append(pred.sum().item())
                loss = -pred.sum()
                loss.backward(retain_graph=True)
                if current_optimizer == 'ops':
                    if epoch_count < ops_epochs:
                        self.adj_ops_model.ops_optimizers.step()
                        self.logger.info(f"search loss: {loss}")
                        epoch_count += 1
                    else:
                        current_optimizer = 'adj'
                        epoch_count = 0  
                else:  
                    if epoch_count < adj_epochs:
                        
                        self.adj_ops_model.adj_optimizers.step()
                        self.logger.info(f"search pred.sum: {pred.sum()}")
                        epoch_count += 1
                    else:

                        current_optimizer = 'ops'
                        epoch_count = 0  

                self.adj_ops_model.temp_scheduler.step()

                if arch_step % self.config.search_para.lr_step == 0:
                    self.adj_ops_model.adj_lr_scheduler.step()
                    self.adj_ops_model.ops_lr_scheduler.step()

                if arch_step == self.search_epochs - 1:
                    self.logger.info(f"Temp {self.adj_ops_model.temp_scheduler.curr_temp}\n")
                    self.logger.info(f"Seed {self.seed}, \n normal adj: \n {self.adj_ops_model.adj_para}\n")
                    self.logger.info(f"Seed {self.seed}, \n normal alpha: \n {self.adj_ops_model.ops_alpha}\n")

                    
                if arch_step % self.config.search_para.log_step == 0:
                    self.logger.info(f"Seed {self.seed}, arch epoch: {arch_step}, loss: {loss.item()}")

        elif self.config.search_space == 'nasbench201':

            self.ops_model.alpha_optimizers.param_groups[0]['lr'] = self.config.search_para.lr
            self.ops_model.temp_init()
            print(self.ops_model.temp_scheduler.curr_temp)
            for arch_step in range(self.search_epochs):
                self.ops_model.alpha_optimizers.zero_grad()
                input_dic = self.prepare_input()
                pred = self.surrogate_model(input_dic)
                loss = -pred[:, 0].sum() + pred[:,
                                           1].sum() - self.config.search_para.std_ratio * self.ops_model.node_attr_alpha.std(
                    dim=0).sum()
                loss.backward()
                self.ops_model.alpha_optimizers.step()
                self.ops_model.temp_scheduler.step()

                # if arch_step == self.search_epochs - 1:
                #     self.logger.info(
                #         f"Seed {self.seed}, gumbel alphas:  {self.ops_model.node_attr_alpha}")
                if arch_step % self.config.search_para.lr_step == 0:
                    self.ops_model.gumbel_lr_scheduler.step()
                    
                if arch_step % self.config.search_para.log_step == 0:
                    self.logger.info(f"Seed {self.seed}, arch epoch: {arch_step}, loss: {loss.item()}")
            self.config.search_para.std_ratio *= self.config.search_para.std_decay

            with torch.no_grad():
                self.ops_model.node_attr_alpha += 0
        return pred_values

    def hifi_query(self, current_step):
        all_hifi_acc_list = []
        all_hifi_spec_list = []
        for i in range(current_step):
            hifi_acc_list = []
            hifi_spec_list = []
            values, top_indexes = torch.topk(torch.Tensor(self.accuracies), i * 5 + self.config.search_para.low_fi_top_k)
            for j in top_indexes:
                spec = self.gcn_history_spec_dict[j]
                acc = self.query_valid_acc(tuple(spec))
                hifi_acc_list.append(acc)
                hifi_spec_list.append(spec)
            all_hifi_acc_list.append(hifi_acc_list)
            all_hifi_spec_list.append(hifi_spec_list)
        return all_hifi_spec_list, all_hifi_acc_list



