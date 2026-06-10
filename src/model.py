import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os
from scipy import sparse
import random    
from torch_geometric.nn import GATConv

    
class LatentTime(torch.nn.Module):
    def __init__(self, num_cells, num_genes, num_layers=3):
        super(LatentTime, self).__init__()
        
        self.num_features = num_genes

        self.d_model = 512
        self.num_layers = num_layers
        
        self.time_input_linear = nn.Linear(self.num_features, self.d_model)
        self.time_input_bn = nn.BatchNorm1d(self.d_model)
          
        modules = []
        for i in range(self.num_layers):
            module = [
                nn.Linear(self.d_model, self.d_model),
                nn.BatchNorm1d(self.d_model),
                nn.LeakyReLU(),
                nn.Linear(self.d_model, self.d_model),
                nn.BatchNorm1d(self.d_model),
                nn.LeakyReLU(),
            ]
            modules.extend(module)
        self.moduleList = torch.nn.ModuleList(modules)
        
        self.time_output_linear = nn.Linear(self.d_model, 1)
        self.time_output_bn = nn.BatchNorm1d(1)
        
    def forward(self, x):
        x = self.time_input_linear(x)
        x = self.time_input_bn(x)
        x = F.leaky_relu(x)
        
        for i in range(self.num_layers):
            residual = x
            n = 6*i
            x = self.moduleList[n + 0](x)
            x = self.moduleList[n + 1](x)
            x = self.moduleList[n + 2](x)
            x = self.moduleList[n + 3](x)
            x = self.moduleList[n + 4](x)
            x = self.moduleList[n + 5](x)
        
        x = self.time_output_linear(x)
        x = self.time_output_bn(x)
#         x = F.sigmoid(x)

        weight_self = x[:, 0:2000]
        weight_self = -F.softplus(weight_self)
        latent_time = x[:, -1]
        latent_time = F.sigmoid(latent_time)
        x = torch.cat([weight_self, latent_time.unsqueeze(dim=1)], dim=1)
        
        return x












class VelocityGAT(torch.nn.Module):
    def __init__(self, in_channels, num_TFs, heads=8, num_layers=4, dropout=0.1):
        super(VelocityGAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = 128
        self.out_channels = num_TFs+6
        self.num_layers = num_layers

        
        self.input_layer = GATConv(self.in_channels, self.hidden_channels, heads=heads, dropout=dropout)
        self.output_layer = GATConv(self.hidden_channels * heads, self.out_channels, heads=1, dropout=dropout)
        
        self.input_linear = nn.Linear(self.in_channels, self.hidden_channels*4)
        self.input_bn = nn.BatchNorm1d(self.hidden_channels*4)
        modules = []
        for i in range(self.num_layers):
            module = [
                nn.Linear(self.hidden_channels*4, self.hidden_channels*4),
                nn.BatchNorm1d(self.hidden_channels*4),
                nn.ELU(),
                nn.Linear(self.hidden_channels*4, self.hidden_channels*4),
                nn.BatchNorm1d(self.hidden_channels*4),
                nn.ELU(),
            ]
            modules.extend(module)
        self.moduleList = torch.nn.ModuleList(modules)
        self.output_linear = nn.Linear(self.hidden_channels*4, self.out_channels)
        self.output_bn = nn.BatchNorm1d(self.out_channels)

        
    def forward(self, data, offset=-1):
        x_input = data.x
        edge_index = data.edge_index
        
        
        x = self.input_layer(x_input, edge_index)
        x= self.output_layer(x, edge_index)
        
        if self.num_layers>0:
            y = self.input_linear(x_input)
            y = self.input_bn(y)
            y = F.elu(y)
            for i in range(self.num_layers):
                n = 6*i
                residual = y
                y = self.moduleList[n + 0](y)
                y = self.moduleList[n + 1](y)
                y = self.moduleList[n + 2](y)
                y = self.moduleList[n + 3](y)
                y = self.moduleList[n + 4](y)
                y = self.moduleList[n + 5](y)
                y += residual
            y = self.output_linear(y)
            
            x = x+y
        
        
        scale_factor = x[:, 0:1].squeeze(dim=1)
        weight_TFs = x[:, 1:-5]
        weight_self = x[:, -5:-4].squeeze(dim=1)
        a = x[:, -4:-3].squeeze(dim=1)
        b = x[:, -3:-2].squeeze(dim=1)
        c = x[:, -2:-1].squeeze(dim=1)
        d = x[:, -1]

        scale_factor = F.sigmoid(scale_factor)*2
        #weight_TFs = F.elu(weight_TFs)
        weight_self = -F.sigmoid(weight_self)
        a = F.softplus(a)
        b = F.sigmoid(b)
        if offset == 0:
            c = F.tanh(c)
        elif offset == 1:
            c = F.sigmoid(c)
        else:
            c = -F.sigmoid(c)
        #d = F.elu(d)
        
        x = torch.cat([scale_factor.unsqueeze(dim=1), weight_TFs, weight_self.unsqueeze(dim=1), a.unsqueeze(dim=1), 
               b.unsqueeze(dim=1), c.unsqueeze(dim=1), d.unsqueeze(dim=1)], dim=1)
        return x