
import logging
import math

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
from adj_ops_nlp import Adj_Ops_Nlp_Model
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
from naslib.predictors import (
    GCNPredictor,
)


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
        elif self.config.search_space == "nlp":
            self.adj_ops_model = Adj_Ops_Nlp_Model(config, dataset_api=self.dataset_api, logger=logger,
                                                   device=self.search_device)  
            self.encoding = importlib.import_module('naslib.search_spaces.nasbenchnlp.encodings')
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
        self.seed = seed
        self.accuracies = []
        self.best_acc = 0.0
        self.best_acc_array = [0]
        self.gcn_history_dict = np.array([])
        self.gcn_history_spec_dict = []
        self.steps = []
        self.best_compact = []
        self.arch_compact = torch.Tensor([])
        self.arch_pred = torch.Tensor([])

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
                target = batch["val_acc"].float().to(self.surrogate_device)
                prediction = self.surrogate_model(batch)
                loss = self.gcn_criterion(prediction, target)
                if self.ranking_loss_reg_coe:  
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
                    if len(pairwise_ranking_loss) == 0:
                        pairwise_ranking_loss = torch.tensor(0.0)  
                    else:
                        pairwise_ranking_loss = torch.mean(torch.stack(pairwise_ranking_loss))

                    loss += self.ranking_loss_reg_coe * pairwise_ranking_loss
                loss.backward()
                self.gcn_optimizer.step()

            if gcn_step % self.config.model_fit_para.log_step == 0:
                self.logger.info(f"Seed {self.seed}, model fit epoch: {gcn_step}, loss: {loss.item()}")
            if gcn_step % self.config.model_fit_para.lr_step == 0:
                self.gcn_lr_scheduler.step()
                

        
        with torch.no_grad():
            data_loader = DataLoader(self.gcn_history_dict, batch_size=1000, shuffle=True)
            for _, data in enumerate(data_loader):
                prediction = self.surrogate_model(data).cpu().detach().numpy() * self.std + self.mean
                target = data["val_acc"] * self.std + self.mean
            fig = scatter_plot(prediction, np.array(target), xlabel='Predicted', ylabel='True',
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
        gcn_spec_dict = []
        gcn_encode_dict = []

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
                    
                    while compact in self.gcn_history_spec_dict:
                        compact = self.ops_model.sample_compact_from_alpha(input_idx)
                    arch = self.search_space.clone()
                    arch.set_spec(compact)
                    dic = self.encoding.encode_gcn_nasbench201(arch)
                    dic['num_vertices'] = np.array(dic['num_vertices'])
                    gcn_encode_dict.append(dic)
                    gcn_spec_dict.append(compact)
                    
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
        with torch.no_grad():
            data_loader = DataLoader(gcn_encode_dict, batch_size=1000, shuffle=False)
            predictions_list = []
            for _, data in enumerate(data_loader):
                if self.config.search_space != "nasbench101":
                    if first_epoch:
                        prediction = torch.rand(len(data['val_acc'])) * self.std + self.mean
                    else:
                        prediction = self.surrogate_model(data) * self.std + self.mean
                else:
                    prediction = self.surrogate_model(data) * self.std + self.mean

                predictions_list.append(prediction)

        
        predictions_tensor = torch.stack(predictions_list, dim=0)
        min_tensor, max_model_idx = torch.min(predictions_tensor, dim=0)
        scores = min_tensor

        unique_pred = scores.unique(sorted=True, dim=0)
        unique_idx = (torch.cat([(scores == x_u).nonzero()[0] for x_u in unique_pred]))
        gcn_spec_dict = np.array(gcn_spec_dict)[unique_idx.detach().cpu().numpy()]
        num_alpha_sample = num_samples
        top_gcn_spec_dict = []
        
        unique_pred = scores[unique_idx[-num_alpha_sample:]]
        top_gcn_spec_dict += gcn_spec_dict[-num_alpha_sample:].tolist()
        self.gcn_history_spec_dict = self.gcn_history_spec_dict + top_gcn_spec_dict

        return top_gcn_spec_dict, unique_pred

    def query_dict(self, spec_dict, acc_pred):
        for i in range(len(spec_dict)):
            spec = spec_dict[i]
            arch = self.search_space.clone()
            arch.set_spec(spec)
            acc = arch.query(metric=Metric.VAL_ACCURACY, dataset=self.dataset, dataset_api=self.dataset_api,
                             sample_mean=self.config.sample_mean)
            if self.config.search_space == "nasbench101":
                self.logger.info(
                    f"Seed {self.seed}, \n spec_adj = \n {spec['matrix']}, \n spec_ops = {spec['ops']}, \n acc = {acc}, acc_pred: {acc_pred[i]}")
            else:
                self.logger.info(f"Seed {self.seed}, spec = {spec}, acc = {acc}, acc_pred: {acc_pred[i]}")
            acc = (acc - self.mean) / self.std  
            if self.config.search_space == "nasbench101" or self.config.search_space == "nasbench301":
                dic = self.encode_func(spec)
            elif self.config.search_space == "nlp":
                dic = self.encode_func(spec, max_nodes=self.config.search_para.num_nodes)
            elif self.config.search_space == "nasbench201":
                dic = self.encode_func(arch)
            else:
                pass
            dic['num_vertices'] = np.array(dic['num_vertices'])
            dic['val_acc'] = np.array(acc)

            self.gcn_history_dict = np.hstack((self.gcn_history_dict, dic))
            acc = acc * self.std + self.mean

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_arch = arch
                self.logger.info(f"Seed {self.seed}, Sample count: {len(self.gcn_history_dict)}, "
                                 f"best_acc: {acc}")
                self.best_compact = spec
            if self.best_acc >= self.config.max_acc:
                self.best_arch_test_acc = self.best_arch.query(metric=Metric.TEST_ACCURACY, dataset=self.dataset,
                                                               dataset_api=self.dataset_api,
                                                               sample_mean=self.config.sample_mean)
                self.logger.info(f"Seed {self.seed}, Sample count: {len(self.gcn_history_dict)}, "
                                 f"best arch's corr test_acc: {self.best_arch_test_acc} best_acc: {acc}")
                print("global optimal reached")
                # exit()
            self.steps.append(len(self.gcn_history_dict))
            self.accuracies.append(acc)
            self.best_acc_array.append(self.best_acc)

    def sampling_and_query_init(self):
        top_gcn_spec_dict, acc_pred = self.sample_and_filter(num_samples=self.config.search_para.num_sample_init, first_epoch=1)
        self.query_dict(top_gcn_spec_dict, acc_pred)

    def sampling_and_query(self):
        top_gcn_spec_dict, acc_pred = self.sample_and_filter(num_samples=self.config.search_para.num_sample_each_step, first_epoch=0)
        self.query_dict(top_gcn_spec_dict, acc_pred)

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
                loss = -pred.sum()
                loss.backward(retain_graph=True)
                self.ops_model.alpha_optimizers.step()
                self.ops_model.temp_scheduler.step()

                # if arch_step == self.search_epochs - 1:
                #     self.logger.info(
                #         f"Seed {self.seed}, gumbel alphas:  {self.ops_model.node_attr_alpha}")
                if arch_step % self.config.search_para.lr_step == 0:
                    self.ops_model.gumbel_lr_scheduler.step()
                    
                if arch_step % self.config.search_para.log_step == 0:
                    self.logger.info(f"Seed {self.seed}, arch epoch: {arch_step}, loss: {loss.item()}")

        return pred_values
