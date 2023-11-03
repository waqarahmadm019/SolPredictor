#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 11:24:09 2023

@author: waqar
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from typing import Optional
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
import torch.nn.functional as F
# from torch import Tensor
from torch.nn import GRUCell, Linear
# import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool,ResGatedGraphConv
# from torch_geometric.typing import Adj, OptTensor
# from torch_geometric.utils import softmax


# from torch_geometric.nn.inits import glorot

class SolPredictor(torch.nn.Module):
    
    def __init__(
        self,
        
        in_channels=9,
        hidden_channels=45,
        out_channels=1,
        edge_dim=3,
        num_layers=2,
        num_timesteps=8,
        dropout: float = 0.127,
    ):
        super().__init__()
        # self.device = device
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        self.lin1 = Linear(in_channels, hidden_channels)
        # self.gate_conv = ResGatedGraphConv(hidden_channels, 1)
        self.gate_conv = ResGatedGraphConv(hidden_channels, hidden_channels)
        # GATEConv(hidden_channels, hidden_channels, edge_dim,
        #                           dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)
        
        
        # self.device
        # self.simple_gru = GRUAttention()
        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            # conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
            #                add_self_loops=False, negative_slope=0.01)
            conv = ResGatedGraphConv(hidden_channels, hidden_channels)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))
        
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        # self.layernorm400 = torch.nn.LayerNorm(400)
        # self.layernorm200 = torch.nn.LayerNorm(200)
        # self.layernorm45 = torch.nn.LayerNorm(45)
        # self.tanh = torch.nn.Tanh()
        # self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.glu= torch.nn.GLU()
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        # self.elu= torch.nn.ELU()
        self.lin2 = Linear(hidden_channels, out_channels)
        # self.lin3 = Linear(90, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()
        # self.lin3.reset_parameters()
        
        # self.lin_fp1.reset_parameters()
        # self.lin_fp2.reset_parameters()

    def forward(self, data, **kwargs):
        """"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr,data.batch
        
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index)) #, edge_attr
        # h_clone = h.clone().detach()
        # h_out = global_add_pool(h_clone, batch).relu_()
        # h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        
        return self.lin2(out)


    def jittable(self) -> 'SolPredictor':
        self.gate_conv = self.gate_conv.jittable()
        self.atom_convs = torch.nn.ModuleList(
            [conv.jittable() for conv in self.atom_convs])
        self.mol_conv = self.mol_conv.jittable()
        return self


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')