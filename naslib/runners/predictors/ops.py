import torch
import logging
import torch
import os
import numpy as np
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.utils.encodings import EncodingType
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from pyDOE import lhs

class Ops_Model(torch.nn.Module):
    def __init__(self, config =None, device = "cpu"):
        super(Ops_Model, self).__init__()
        self.config = config
        self.device = device
        self.num_nodes = config.search_para.num_nodes
        self.num_ops = config.search_para.num_ops
        self.arch_batch_size = config.search_para.input_batch_size
        self.alpha_init()
        self.optim_init()

    def alpha_init(self):
        if self.arch_batch_size != 1:
            lhs_samples = lhs(self.num_nodes * self.num_ops, samples=self.arch_batch_size, criterion='m')
        else:
            lhs_samples = lhs(self.num_nodes * self.num_ops, samples=self.arch_batch_size)
        scaled_lhs_samples = self.config.search_para.lhs_lower + (lhs_samples * self.config.search_para.lhs_range)
        
        
        node_attr_alpha_values = scaled_lhs_samples.reshape(self.arch_batch_size, self.num_nodes, self.num_ops)
        node_attr_alpha = Parameter(torch.tensor(node_attr_alpha_values, dtype=torch.float32).to(self.device),
                                    requires_grad=True)
        self.node_attr_alpha = node_attr_alpha

    def alpha_optimizer_init(self):
        self.alpha_optimizers = optim.AdamW([self.node_attr_alpha], lr=self.config.search_para.lr)
        self.gumbel_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.alpha_optimizers, self.config.search_para.lr_gamma)

    def temp_init(self):
        
        temp_total_epochs = self.config.search_para.search_epochs
        
        ops_epochs_ratio = 1.0
        self.temp_scheduler = Temp_Scheduler(total_epochs=temp_total_epochs * ops_epochs_ratio,
                                             curr_temp=self.config.search_para.base_temp,
                                             base_temp=self.config.search_para.base_temp,
                                             temp_min=self.config.search_para.min_temp)

    def optim_init(self):
        self.temp_init()
        self.alpha_optimizer_init()

    def get_gumbel_dist(self, logit_alpha):
        
        u = torch.zeros_like(logit_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        
        
        r = softmax((logit_alpha + (-((-(u.log())).log()))) / self.temp_scheduler.curr_temp)
        return r

    def get_indices(self, r):
        
        r_hard = torch.argmax(r, dim=1)
        
        r_hard_one_hot = F.one_hot(r_hard, num_classes=r.size(1)).float()  
        
        r_re = (r_hard_one_hot - r).detach() + r
        
        return r

    def forward(self):
        batch_results = []
        for alpha in self.node_attr_alpha:
            if self.config.search_space == 'nasbench201':
                pre_node = torch.zeros((1, self.num_ops + 2)).to(self.device)
                pre_node[0, 0] = 1  
                gumbel_distribution = self.get_gumbel_dist(alpha)
                arch_indices = self.get_indices(gumbel_distribution)
                left_expanded = torch.cat([torch.zeros(self.num_nodes, 1).to(self.device), arch_indices], dim=-1)
                expanded = torch.cat([left_expanded, torch.zeros(self.num_nodes, 1).to(self.device)], dim=-1) 
                tail_node = torch.zeros((1, self.num_ops + 2)).to(self.device)
                tail_node[0, -1] = 1  
                result = torch.cat([pre_node, expanded, tail_node], dim=0)
                batch_results.append(result)
            elif self.config.search_space == 'nasbench101':
                pre_node = torch.zeros((1, self.num_ops + 2)).to(self.device)
                pre_node[0, 1] = 1  
                gumbel_distribution = self.get_gumbel_dist(alpha)
                arch_indices = self.get_indices(gumbel_distribution)
                left_expanded = torch.cat([torch.zeros(self.num_nodes, 1).to(self.device), arch_indices], dim=-1)
                
                expanded = torch.cat([ torch.zeros(self.num_nodes, 1).to(self.device), left_expanded], dim=-1)  
                tail_node = torch.zeros((1, self.num_ops + 2)).to(self.device)
                tail_node[0, 0] = 1  
                result = torch.cat([pre_node, expanded, tail_node], dim=0)
                batch_results.append(result)
            else:
                raise NotImplementedError

        return torch.stack(batch_results)

    def get_compact_from_alpha(self):
        cell_compact = []
        
        for i in range(len(self.node_attr_alpha)):
            op_idx = self.node_attr_alpha[i].argmax().item()  
            cell_compact.append(op_idx)
        return cell_compact

    def sample_compact_from_alpha(self, alpha_idx):
        cell_compact = []
        
        for i in range(len(self.node_attr_alpha[alpha_idx])):
            if self.config.search_space == 'nasbench201':
                op_idx = torch.multinomial(0.3*torch.rand(len(self.node_attr_alpha[alpha_idx][i])).to(self.node_attr_alpha.device) + \
                                           F.softmax(self.node_attr_alpha[alpha_idx][i], dim=-1), 1).item()  
            else:
                op_idx = torch.multinomial(F.softmax(self.node_attr_alpha[alpha_idx][i], dim=-1), 1).item()  
            cell_compact.append(op_idx)
        return cell_compact


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp
