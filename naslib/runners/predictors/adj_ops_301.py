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

fixed_zero_matrix = np.array(
    [
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.float32
)  

fixed_one_matrix = np.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32
) 

class Adj_Ops_Model(Module):
    def __init__(self, config, dataset_api, logger, device = 'cpu'):
        super(Adj_Ops_Model, self).__init__()
        self.config = config
        
        self.num_nodes = config.search_para.num_nodes
        self.num_ops = config.search_para.num_ops
        self.device = device
        self.logger = logger
        self.batch_size = config.search_para.input_batch_size
        self.dataset_api  = dataset_api

        
        self.normal_adj_para = self.adj_init(self.num_nodes + 3)   
        self.reduction_adj_para = self.adj_init(self.num_nodes + 3)

        
        self.normal_ops_alpha = self.alpha_init(self.num_nodes, self.num_ops)  
        self.reduction_ops_alpha = self.alpha_init(self.num_nodes, self.num_ops)

        self.optim_init()
        self.temp_init()

    def adj_init(self, num_vertices):
        
        fixed_zero_tensor = torch.tensor(fixed_zero_matrix, dtype=torch.bool).to(self.device)
        fixed_one_tensor = torch.tensor(fixed_one_matrix, dtype=torch.bool).to(self.device)

        
        lhs_samples = lhs(num_vertices * (num_vertices - 1) // 2, samples=self.batch_size)

        scaled_samples = 0.3 + 0.2 * lhs_samples  

        adj_values = torch.zeros((self.batch_size, num_vertices, num_vertices), dtype=torch.float32).to(self.device)
        upper_indices = np.triu_indices(num_vertices, k=1)
        for b in range(self.batch_size):
            adj_values[b, upper_indices[0], upper_indices[1]] = torch.tensor(scaled_samples[b], dtype=torch.float32).to(self.device)

        
        trainable_matrix = Parameter(adj_values, requires_grad=True)

        
        trainable_matrix.data[:, fixed_one_tensor] = 1.0
        trainable_matrix.data[:, fixed_zero_tensor] = 0

        
        gradient_mask = torch.logical_not(torch.logical_or(fixed_zero_tensor, fixed_one_tensor))

        
        trainable_matrix.register_hook(lambda grad: grad * gradient_mask[None, :, :])

        return trainable_matrix

    def alpha_init(self, num_nodes, num_ops):
        
        lhs_samples = lhs(num_nodes * num_ops, samples = self.batch_size)
        
        scaled_samples = 0.0 + (lhs_samples * 2.0)
        
        node_alpha = scaled_samples.reshape([self.batch_size, num_nodes, num_ops])
        
        return Parameter(torch.tensor(node_alpha, dtype=torch.float32).to(self.device), requires_grad=True)

    def temp_init(self):
        
        
        temp_total_epochs = self.config.search_para.search_epochs
        
        
        adj_epochs_ratio = 1.0
        self.temp_scheduler = Temp_Scheduler(
            total_epochs = temp_total_epochs * adj_epochs_ratio,
            curr_temp=self.config.search_para.base_temp,
            base_temp=self.config.search_para.base_temp,
            temp_min=self.config.search_para.min_temp,)

    def optimizer_init(self):
        edge_parameters = [self.normal_adj_para] + [self.reduction_adj_para]
        ops_parameters = [self.normal_ops_alpha] + [self.reduction_ops_alpha]

        self.adj_optimizers = optim.AdamW(edge_parameters, lr=self.config.search_para.lr_beta)
        self.ops_optimizers = optim.AdamW(ops_parameters, lr=self.config.search_para.lr_beta)

        self.adj_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.adj_optimizers, self.config.search_para.lr_gamma)
        self.ops_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.ops_optimizers, self.config.search_para.lr_gamma)

    def optim_init(self):
        self.temp_init()
        self.optimizer_init()

    def sample_compact_from_alpha_301(self, index):
        fixed_zero_mask = torch.tensor(fixed_zero_matrix, dtype=torch.bool).to(self.device)
        fixed_one_mask = torch.tensor(fixed_one_matrix, dtype=torch.bool).to(self.device)
        def extract_cell(adj_para, ops_alpha):
            
            adj_para = torch.sigmoid(adj_para)
            adj_para = adj_para.masked_fill(fixed_zero_mask, 0)
            adj_para = adj_para.masked_fill(fixed_one_mask, 1)
            sampled_adj = adj_para.cpu().detach().numpy()
            
            cell = []

            for i in [0, 1]:
                
                op_idx = torch.multinomial(F.softmax(ops_alpha[i], dim=-1), 1).item()
                cell.append((i, op_idx))

            edge_dist_4 = sampled_adj[0:2, 4]
            op_idx_4 = torch.multinomial(F.softmax(ops_alpha[2], dim=-1), 1).item()
            
            edge_dist_5 = np.zeros(2)
            edge_dist_5[0] = sampled_adj[1, 5]
            edge_dist_5[1] = np.mean(sampled_adj[2:4, 5])
            op_idx_5 = torch.multinomial(F.softmax(ops_alpha[3], dim=-1), 1).item()

            valid = False
            attempts = 0
            while not valid:
                edge_idx_4 = torch.multinomial(F.softmax(torch.tensor(edge_dist_4, dtype=torch.float32), dim=-1),
                                               1).item()
                edge_idx_5 = torch.multinomial(F.softmax(torch.tensor(edge_dist_5, dtype=torch.float32), dim=-1),
                                                1).item()
                if edge_idx_4 < edge_idx_5:
                    valid = True
                attempts += 1
                if attempts > 10 and attempts % 10 == 0:  
                    self.logger.info(f"Failed to find valid edge indices (O4,O5) after {attempts} attempts")

            cell.append((edge_idx_4, op_idx_4))
            cell.append((edge_idx_5, op_idx_5))

            edge_dist_6 = np.zeros(3)
            edge_dist_6[0] = sampled_adj[0, 6]
            edge_dist_6[1] = sampled_adj[1, 6]
            edge_dist_6[2] = np.mean(sampled_adj[2:4, 6])
            op_idx_6 = torch.multinomial(F.softmax(ops_alpha[4], dim=-1), 1).item()

            
            edge_dist_7 = np.zeros(3)
            edge_dist_7[0] = sampled_adj[1, 7]
            edge_dist_7[1] = np.mean(sampled_adj[2:4, 7])
            edge_dist_7[2] = np.mean(sampled_adj[4:6, 7])
            op_idx_7 = torch.multinomial(F.softmax(ops_alpha[5], dim=-1), 1).item()

            valid = False
            attempts = 0
            while not valid:
                edge_idx_6 = torch.multinomial(F.softmax(torch.tensor(edge_dist_6, dtype=torch.float32), dim=-1),
                                               1).item()
                edge_idx_7 = torch.multinomial(F.softmax(torch.tensor(edge_dist_7, dtype=torch.float32), dim=-1),
                                                1).item()
                if edge_idx_6 < edge_idx_7:
                    valid = True
                attempts += 1
                if attempts > 10 and attempts % 10 == 0:  
                    self.logger.info(f"Failed to find valid edge indices (O6,O7) after {attempts} attempts")

            cell.append((edge_idx_6, op_idx_6))
            cell.append((edge_idx_7, op_idx_7))

            
            edge_dist_8 = np.zeros(4)
            edge_dist_8[0] = sampled_adj[0, 8]
            edge_dist_8[1] = sampled_adj[1, 8]
            edge_dist_8[2] = np.mean(sampled_adj[2:4, 8])
            edge_dist_8[3] = np.mean(sampled_adj[4:6, 8])
            op_idx_8 = torch.multinomial(F.softmax(ops_alpha[6], dim=-1), 1).item()

            
            edge_dist_9 = np.zeros(4)
            edge_dist_9[0] = sampled_adj[1, 9]
            edge_dist_9[1] = np.mean(sampled_adj[2:4, 9])
            edge_dist_9[2] = np.mean(sampled_adj[4:6, 9])
            edge_dist_9[3] = np.mean(sampled_adj[6:8, 9])
            op_idx_9 = torch.multinomial(F.softmax(ops_alpha[7], dim=-1), 1).item()
            
            valid = False
            attempts = 0
            while not valid:
                edge_idx_8 = torch.multinomial(F.softmax(torch.tensor(edge_dist_8, dtype=torch.float32), dim=-1),
                                               1).item()
                edge_idx_9 = torch.multinomial(F.softmax(torch.tensor(edge_dist_9, dtype=torch.float32), dim=-1),
                                                1).item()
                if edge_idx_8 < edge_idx_9:
                    valid = True
                attempts += 1
                if attempts > 10 and attempts % 10 == 0:  
                    self.logger.info(f"Failed to find valid edge indices (O8,O9) after {attempts} attempts")

            cell.append((edge_idx_8, op_idx_8))
            cell.append((edge_idx_9, op_idx_9))

            return cell

        normal_cell = extract_cell(self.normal_adj_para[index], self.normal_ops_alpha[index])
        reduction_cell = extract_cell(self.reduction_adj_para[index], self.reduction_ops_alpha[index])

        cell_compact = (normal_cell, reduction_cell)

        return cell_compact

    def gumbel_sigmoid_sample(self, logits):
        size = logits.size()
        uniform = torch.rand(size).to(self.device)
        gumbel_noise = -torch.log(-torch.log(uniform))
        y = logits + gumbel_noise
        return torch.sigmoid(y / self.temp_scheduler.curr_temp)

    def transform_adj_matrix(self, adj_dist):
        fixed_zero_mask = torch.tensor(fixed_zero_matrix, dtype=torch.bool).to(self.device)
        fixed_one_mask = torch.tensor(fixed_one_matrix, dtype=torch.bool).to(self.device)

        activated_adj_dist = self.gumbel_sigmoid_sample(adj_dist)

        activated_adj_dist = activated_adj_dist.masked_fill(fixed_zero_mask, 0)
        activated_adj_dist = activated_adj_dist.masked_fill(fixed_one_mask, 1)

        return activated_adj_dist
        

    def get_cur_adj_item(self, index):
        matrices = []

        mat = self.transform_adj_matrix(self.normal_adj_para[index])
        matrices.append(mat)

        mat = self.transform_adj_matrix(self.reduction_adj_para[index])
        matrices.append(mat)

        mat_length = len(matrices[0][0])
        merged_length = mat_length * 2

        matrix_final = torch.zeros((merged_length, merged_length)).to(self.device)
        matrix_final[:mat_length, :mat_length] = matrices[0]
        matrix_final[mat_length:, mat_length:] = matrices[1]

        
        matrix_final[mat_length - 1, mat_length] = 1
        matrix_final[ mat_length - 1, mat_length + 1] = 1

        return matrix_final

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
        pre_node = torch.zeros((2, len(OPS) + 2)).to(self.device)
        pre_node[0:2, 0] = 1   
        gumbel_distribution = self.get_gumbel_dist(log_alpha_dict, self.temp_scheduler.curr_temp)
        arch_indices = self.get_indices(gumbel_distribution)
        left_expanded = torch.cat([torch.zeros(self.config.search_para.num_nodes, 1).to(self.device), arch_indices], dim=1)
        expanded = torch.cat([left_expanded, torch.zeros(self.config.search_para.num_nodes, 1).to(self.device)], dim=1)
        tail_node = torch.zeros((1, len(OPS) + 2)).to(self.device)
        tail_node[:, -1] = 1  
        result = torch.cat([pre_node, expanded, tail_node], dim=0)
        return result

    def get_cur_ops(self, log_alpha_arr):
        cur_ops_array = []
        for i in range(self.batch_size):
            cur_ops_array.append(self.get_cur_ops_item(log_alpha_arr[i]))
        return torch.stack(cur_ops_array)

    def get_normal_reduction_ops(self):
        normal_ops = self.get_cur_ops(self.normal_ops_alpha)
        reduction_ops = self.get_cur_ops(self.reduction_ops_alpha)
        ops_onehot_batch = torch.cat((normal_ops, reduction_ops), dim=1)
        return ops_onehot_batch

    def forward(self):
        
        adj_arr = self.get_cur_adj()
        ops_arr = self.get_normal_reduction_ops()
        return adj_arr, ops_arr
        
