import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from ops import Temp_Scheduler
from naslib.search_spaces.nasbench301.encodings import *
from naslib.utils import get_dataset_api
from pyDOE import lhs
import torch.optim as optim
import torch.nn.functional as F
import random


class Adj_Ops_Nlp_Model(Module):
    def __init__(self, config, dataset_api, logger, device = 'cpu'):
        super(Adj_Ops_Nlp_Model, self).__init__()
        self.config = config
        
        self.num_nodes = config.search_para.num_nodes + 1
        self.num_ops = config.search_para.num_ops
        self.device = device
        self.logger = logger
        self.batch_size = config.search_para.input_batch_size
        self.dataset_api  = dataset_api

        self.para_init()
        self.optim_init()
        self.temp_init()
        self.hidden_state_init()

    def hidden_state_init(self):
        numbers = list(range(5, 9))
        self.hidden_state = random.sample(numbers, 3)
        self.hidden_state.sort()

    def fixed_zero_mask_init(self):
        matrix = np.ones((self.num_nodes, self.num_nodes))
        matrix[np.triu_indices(self.num_nodes, k=1)] = 0

        return matrix

    def adj_init(self, num_vertices):
        fixed_zero_tensor = torch.tensor(self.fixed_zero_matrix, dtype=torch.bool).to(self.device)

        lhs_samples = lhs(num_vertices * (num_vertices - 1) // 2, samples=self.batch_size)

        scaled_samples = -0.1 + 0.2 * lhs_samples

        
        adj_values = torch.zeros((self.batch_size, num_vertices, num_vertices), dtype=torch.float32).to(self.device)
        upper_indices = np.triu_indices(num_vertices, k=1)
        for b in range(self.batch_size):
            adj_values[b, upper_indices[0], upper_indices[1]] = torch.tensor(scaled_samples[b], dtype=torch.float32).to(self.device)

        
        trainable_matrix = Parameter(adj_values, requires_grad=True)

        
        trainable_matrix.data[:, fixed_zero_tensor] = 0

        
        gradient_mask = torch.logical_not(fixed_zero_tensor)

        
        trainable_matrix.register_hook(lambda grad: grad * gradient_mask[None, :, :])

        return trainable_matrix

    def alpha_init(self, num_nodes, num_ops):
        
        lhs_samples = lhs(num_nodes * num_ops, samples = self.batch_size)
        
        scaled_samples = 0.0 + (lhs_samples * 2.0)
        
        node_alpha = scaled_samples.reshape([self.batch_size, num_nodes, num_ops])
        
        return Parameter(torch.tensor(node_alpha, dtype=torch.float32).to(self.device), requires_grad=True)

    def para_init(self):
        self.fixed_zero_matrix = self.fixed_zero_mask_init()
        
        self.adj_para = self.adj_init(self.num_nodes)  
        
        self.ops_alpha = self.alpha_init(self.num_nodes, self.num_ops)  

    def temp_init(self):
        
        
        temp_total_epochs = self.config.search_para.search_epochs
        
        
        adj_epochs_ratio = 1.0
        self.temp_scheduler = Temp_Scheduler(
            total_epochs = temp_total_epochs * adj_epochs_ratio,
            curr_temp=self.config.search_para.base_temp,
            base_temp=self.config.search_para.base_temp,
            temp_min=self.config.search_para.min_temp,)

    def optimizer_init(self):
        edge_parameters = [self.adj_para]
        ops_parameters = [self.ops_alpha]

        self.adj_optimizers = optim.AdamW(edge_parameters, lr=self.config.search_para.lr_beta)
        self.ops_optimizers = optim.AdamW(ops_parameters, lr=self.config.search_para.lr_beta)

        self.adj_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.adj_optimizers, self.config.search_para.lr_gamma)
        self.ops_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.ops_optimizers, self.config.search_para.lr_gamma)

    def optim_init(self):
        self.temp_init()
        self.optimizer_init()

    def sample_compact_from_alpha_nlp(self, index):
        fixed_zero_mask = torch.tensor(self.fixed_zero_matrix, dtype=torch.bool).to(self.device)

        
        adj_para = torch.sigmoid(self.adj_para[index])
        adj_para = adj_para.masked_fill(fixed_zero_mask, 0)
        s = adj_para.cpu().detach().numpy()
        sampled_adj = np.random.binomial(1, s)
        
        edge_indices = np.nonzero(sampled_adj)
        
        edge_compact = [(int(from_idx), int(to_idx)) for from_idx, to_idx in zip(edge_indices[0], edge_indices[1])]

        
        ops_compact = []
        for i in range(len(self.ops_alpha[index])):
            op_idx = torch.multinomial(F.softmax(self.ops_alpha[index][i], dim=-1), 1).item()
            ops_compact.append(op_idx)

        cell_compact = (edge_compact, ops_compact, self.hidden_state)

        return cell_compact

    def gumbel_sigmoid_sample(self, logits):
        size = logits.size()
        uniform = torch.rand(size).to(self.device)
        gumbel_noise = -torch.log(-torch.log(uniform))
        y = logits + gumbel_noise
        return torch.sigmoid(y / self.temp_scheduler.curr_temp)

    def transform_adj_matrix(self, adj_dist):
        fixed_zero_mask = torch.tensor(self.fixed_zero_matrix, dtype=torch.bool).to(self.device)
        activated_adj_dist = self.gumbel_sigmoid_sample(adj_dist)
        activated_adj_dist = activated_adj_dist.masked_fill(fixed_zero_mask, 0)

        return activated_adj_dist

    def get_cur_adj_item(self, index):
        mat = self.transform_adj_matrix(self.adj_para[index])
        return mat

    def get_cur_adj(self):
        cur_adj_array = []
        for i in range(self.batch_size):
            cur_adj_array.append(self.get_cur_adj_item(i))
        return torch.stack(cur_adj_array)

    def get_gumbel_dist(self, log_alpha, temp):
        
        u = torch.zeros_like(log_alpha).uniform_().to(self.device)
        softmax = torch.nn.Softmax(-1)
        r = softmax((log_alpha + (-((-(u.log())).log()))) / temp)
        return r

    def get_indices(self, r):
        
        r_hard = torch.argmax(r, dim=1)
        
        r_hard_one_hot = F.one_hot(r_hard, num_classes=r.size(
            1)).float()  
        
        r_re = (r_hard_one_hot - r).detach() + r
        
        return r     

    def get_cur_ops_item(self, log_alpha_dict):
        gumbel_distribution = self.get_gumbel_dist(log_alpha_dict, self.temp_scheduler.curr_temp)
        arch_indices = self.get_indices(gumbel_distribution)
        return arch_indices

    def get_cur_ops(self):
        cur_ops_array = []
        for i in range(self.batch_size):
            cur_ops_array.append(self.get_cur_ops_item(self.ops_alpha[i]))
        return torch.stack(cur_ops_array)

    def forward(self):
        adj_arr = self.get_cur_adj()
        ops_arr = self.get_cur_ops()
        return adj_arr, ops_arr
        
