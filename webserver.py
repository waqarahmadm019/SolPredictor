#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:13:47 2023

@author: waqar
"""

import torch
import os
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from models.solubilityPredictor_resgatedgraphconv import SolPredictor
from models.regression_train_test import SolPredictions
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch_geometric.utils import from_smiles
savepath = 'solubilitypredictor_resgatedgraphconv11/'

from torch_geometric.data import InMemoryDataset #, Data
seed = 120
import time
if torch.cuda.is_available():  
    device = "cuda:4"
    torch.cuda.manual_seed_all(seed)
    # print("cuda:4")
else:  
    device = "cpu" 
    # print(torch.cuda.is_available())
device = "cpu"    
class Molecule_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', y=None, transform=None,
                 pre_transform=None,smiles=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
#             print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            # print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    
    
    def process(self, smiles):
       
        data_list = []
        for i in range(len(smiles)):
            smile = smiles[i]
            data = from_smiles(smile)
            data.x = (data.x).type(torch.FloatTensor)
            data.edge_attr = (data.edge_attr).type(torch.FloatTensor)
            data.smile_fingerprint = None
            graph = data
            
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def Predictions(smilesList):
    transform = T.Compose([T.NormalizeFeatures(['x', 'edge_attr'])])
    
    test_data_set = 'test_data_set'+str(time.time_ns())
    test_data = Molecule_data(root='data', dataset=test_data_set,y=None,
                               smiles=smilesList, transform=transform)
    noveltest_loader  = DataLoader(test_data,batch_size=64,shuffle=True)
    model = SolPredictor().to(device)
    model_file_name = 'saved_models/' + savepath +'model_1.model'
    
    checkpoint = torch.load(model_file_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    predictions = SolPredictions(noveltest_loader, model, 'cpu')

    # print(predictions)
    return predictions


smileslist = ['COC1=CC(=NNC(=O)c2ccncc2)CC(C21Oc1c(C2=O)c(OC)cc(c1Cl)OC)C',
              'FC1=CC(OC(F)(F)F)=CC=C1C2=CC=C(CO[C@H]3COC4=NC([N+]([O-])=O)=CN4C3)N=C2']


predictions = Predictions(smileslist)
print(predictions)
