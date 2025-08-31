import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from ops import Temp_Scheduler
from naslib.search_spaces.nasbench101.graph import *
from naslib.utils import get_dataset_api
from pyDOE import lhs
import torch.optim as optim

class Adj_Model(Module):
    def __init__(self, config, dataset_api, logger, device = 'cpu'):
        super(Adj_Model, self).__init__()
        self.config = config
        self.nnodes = config.num_vertices
        self.device = device
        self.logger = logger
        self.batch_size = config.search_para.input_batch_size
        self.dataset_api  = dataset_api
        self.adj_init()
        self.optim_init()

    def adj_init(self):
        adj_elements_num = int(self.nnodes * (self.nnodes - 1) / 2)
        lhs_samples = lhs(adj_elements_num, samples=self.batch_size)
        
        
        scaled_lhs_samples = 0.1 + (lhs_samples * 0.2)
        adj_values = scaled_lhs_samples.reshape(self.batch_size, adj_elements_num)
        self.adj = Parameter(torch.tensor(adj_values, dtype=torch.float32),
                                    requires_grad=True)

    def temp_init(self):
        
        temp_total_epochs = self.config.search_para.search_epochs
        
        adj_epochs_ratio = 1.0
        self.temp_scheduler = Temp_Scheduler(
            total_epochs = temp_total_epochs * adj_epochs_ratio,
            curr_temp=self.config.search_para.base_temp,
            base_temp=self.config.search_para.base_temp,
            temp_min=self.config.search_para.min_temp,)

    def adj_optimizer_init(self):
        self.adj_optimizers = optim.Adam([self.adj], lr=self.config.search_para.lr_beta)

    def optim_init(self):
        self.temp_init()
        self.adj_optimizer_init()

    def attack_once(self, loss, lr):
        adj_grad = torch.autograd.grad(loss, self.adj)[0]
        self.adj.data.add_(lr * adj_grad)

    def sample_descret_one_item(self, i):

        cur_adj = torch.sigmoid(self.adj.data[i] + 0.3)
        s = cur_adj.cpu().detach().numpy()
        sampled = np.random.binomial(1, s)
        m = torch.zeros((self.nnodes, self.nnodes))
        triu_indices = torch.triu_indices(row=self.nnodes, col=self.nnodes, offset=1)
        m[triu_indices[0], triu_indices[1]] = torch.tensor(sampled).float()  
        return m

    def gumbel_sigmoid_sample(self, logits):
        size = logits.size()
        uniform = torch.rand(size)
        gumbel_noise = -torch.log(-torch.log(uniform))
        y = logits + gumbel_noise
        return torch.sigmoid(y / self.temp_scheduler.curr_temp)

    def apply_softmax_adj(self, adj):
        
        segments = [6, 5, 4, 3, 2, 1]
        start_idx = 0
        softmaxed_parts = []

        
        for size in segments:
            end_idx = start_idx + size
            segment = adj[start_idx:end_idx]
            
            if size > 1:
                softmaxed_segment = F.softmax(segment, dim=0)
            else:
                softmaxed_segment = segment  
            
            scale_factor = segment.sum() / softmaxed_segment.sum()
            rescaled_softmaxed_segment = softmaxed_segment * scale_factor
            softmaxed_parts.append(rescaled_softmaxed_segment)
            start_idx = end_idx

        
        return torch.cat(softmaxed_parts, dim=0)

    def adj_propagate_item(self, i):

        gumbel_adj = self.gumbel_sigmoid_sample(self.adj[i])

        m = torch.zeros((self.nnodes, self.nnodes))
        triu_indices = torch.triu_indices(row=self.nnodes, col=self.nnodes, offset=1)
        m[triu_indices[0], triu_indices[1]] = gumbel_adj   

        return m

    def get_cur_adj(self):
        cur_adj_array = []
        for i in range(self.batch_size):
            m = torch.zeros((self.nnodes, self.nnodes))
            triu_indices = torch.triu_indices(row=self.nnodes, col=self.nnodes, offset=1)
            m[triu_indices[0], triu_indices[1]] = self.adj.data[i]
            cur_adj_array.append(m)
        return torch.stack(cur_adj_array)

    def forward(self):
        cur_adj_array = []
        for i in range(self.batch_size):
            cur_adj_array.append(self.adj_propagate_item(i))
        return torch.stack(cur_adj_array)
