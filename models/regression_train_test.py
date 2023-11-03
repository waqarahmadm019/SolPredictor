#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:24:08 2023

@author: waqar
"""
import torch
import torch.nn.functional as F
from math import sqrt

@torch.no_grad()
def SolPredictions(loader, model, device):
    total_preds = torch.Tensor()
    for data in loader:
        data = data.to(device)
        out = model(data)
        total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
        
    return total_preds.numpy().flatten()
