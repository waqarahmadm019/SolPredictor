#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:18:38 2022

@author: waqar
"""
from tqdm import tqdm
import pandas as pd
# import numpy as np
# import torch.nn.functional as F
# from rdkit import Chem
# from rdkit.Chem import Draw
# import networkx as nx
# from torch_geometric.datasets import MoleculeNet
# import matplotlib.pyplot as plt
# from sklearn import model_selection, preprocessing, metrics, decomposition

import torch_geometric.transforms as T
# import torch
import os
# import json,pickle
# from collections import OrderedDict
# from rdkit.Chem import MolFromSmiles


# from torch_geometric.data import InMemoryDataset, Data
# from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold, KFold #, train_test_split
# from tqdm.notebook import tqdm
# import torch
# import torch.nn as nn
# import Data_Prep.Graph_Data as gd
from Data_Prep.Graph_Data import Molecule_data
# import selfies as sf
# from math import sqrt
# from models.attenFP_v1 import AttentionConvNet
# from sklearn.utils.class_weight import compute_class_weight
# from torchmetrics import Dice, Accuracy

def createFoldsData(n_splits, smilesColumn, labelColumn, savepath):
    iy = 0
    folds = n_splits
    for fold in tqdm(range(folds)):
        print("fold : ", fold)
        df_train = pd.read_csv('New_fold/'+savepath+'fold_'+str(iy)+'_'+'x_train.csv')
        df_test  = pd.read_csv('New_fold/'+savepath+'fold_'+str(iy)+'_'+'x_test.csv')
        smiles = df_train[smilesColumn]
#         codIds = df_train['CODID']
        herg_labels = df_train[labelColumn]
        herg_labels = herg_labels.to_numpy()

        smiles_test = df_test[smilesColumn]
#         codIds_test = df_test['CODID']
        herg_labels_test = df_test[labelColumn]
        herg_labels_test = herg_labels_test.to_numpy()


        # smile_graph = {}
        herg_arr = []
        smiles_array = []

        for i,smile in enumerate(smiles):
            # print(f'smile: {smile}')
            # selfie = sf.encoder(smile)
            # selfieTosmile = sf.decoder(selfie)
            # print(f'selfieTosmile: {selfieTosmile}')
            # g = gd.smile_to_graph(smile)
            # if g != None:
            # smile_graph[smile] = g
            herg_arr.append(herg_labels[i])
            smiles_array.append(smile)

        # smile_graph_test = {}
        herg_arr_test = []
        smiles_array_test = []

        for i,smile in enumerate(smiles_test):
            # g = gd.smile_to_graph(smile)
            # if g != None:
            # smile_graph_test[smile] = g
            herg_arr_test.append(herg_labels_test[i])
            smiles_array_test.append(smile)
        print("train fold : ", fold )
        train_data = Molecule_data(root='data/'+savepath, dataset='train_data_set_fold_'+str(iy),y=herg_arr,
                                   smiles=smiles_array)
        
        print("test fold : ", fold )
        test_data = Molecule_data(root='data/'+savepath, dataset='test_data_set_fold_'+str(iy),y=herg_arr_test,
                                   smiles=smiles_array_test)

        iy+=1

#smilesColumn as string required to get smiles data from dataframe
#labelColumn get concerned labels data from dataframe
#splitStrategy='KFold' or splitStrategy='StratifiedKFold' for imbalanced dataset
def createFoldsCsv(df, n_splits, smilesColumn, labelColumn, savepath, splitStrategy='KFold'):
    if splitStrategy=='StratifiedKFold':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    
    ix = 0
    train1 = df
    smiles = df[smilesColumn]
    # codIds = df['CODID']
    labels = df[labelColumn]
    for train_index, test_index in (kf.split(smiles, labels)):
        train_X, test_X = smiles[train_index], smiles[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
        test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
        print('>>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        print ("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train,X_test=train1.iloc[train_index], train1.iloc[test_index]
        if not os.path.exists('New_fold/'+savepath):
            os.makedirs('New_fold/'+savepath)
        X_train.to_csv('New_fold/'+savepath+'fold_'+str(ix)+'_'+'x_train.csv',index=False)
        X_test.to_csv('New_fold/'+savepath+'fold_'+str(ix)+'_'+'x_test.csv',index=False)
        ix+=1
    createFoldsData(n_splits, smilesColumn, labelColumn, savepath=savepath)


def createData(traincsv, testcsv, smilesColumn, labelColumn):
    # iy = 0
    # folds = n_splits
    # for fold in tqdm(range(folds)):
    df_train = pd.read_csv('New_fold/'+traincsv)
    df_test  = pd.read_csv('New_fold/'+testcsv)
    smiles = df_train[smilesColumn]
#         codIds = df_train['CODID']
    herg_labels = df_train[labelColumn]
    herg_labels = herg_labels.to_numpy()

    smiles_test = df_test[smilesColumn]
#         codIds_test = df_test['CODID']
    herg_labels_test = df_test[labelColumn]
    herg_labels_test = herg_labels_test.to_numpy()


    # smile_graph = {}
    herg_arr = []
    smiles_array = []

    for i,smile in enumerate(smiles):
        # print(f'smile: {smile}')
        # selfie = sf.encoder(smile)
        # selfieTosmile = sf.decoder(selfie)
        # print(f'selfieTosmile: {selfieTosmile}')
        # g = gd.smile_to_graph(smile)
        # if g != None:
        # smile_graph[smile] = g
        herg_arr.append(herg_labels[i])
        smiles_array.append(smile)

    # smile_graph_test = {}
    herg_arr_test = []
    smiles_array_test = []

    for i,smile in enumerate(smiles_test):
        # g = gd.smile_to_graph(smile)
        # if g != None:
        # smile_graph_test[smile] = g
        herg_arr_test.append(herg_labels_test[i])
        smiles_array_test.append(smile)

    train_data = Molecule_data(root='data', dataset='train_data_set',y=herg_arr,
                               smiles=smiles_array)

    test_data = Molecule_data(root='data', dataset='test_data_set',y=herg_arr_test,
                               smiles=smiles_array_test)

    

#smilesColumn as string required to get smiles data from dataframe
#labelColumn get concerned labels data from dataframe

def createCsv(traindf, testdf, smilesColumn, labelColumn):
    
    traindf.to_csv('New_fold/x_train.csv',index=False)
    testdf.to_csv('New_fold/x_test.csv',index=False)
    createData(smilesColumn, labelColumn)
    
# whole data set as training dataset
def createTestData(path,filename,datasetname, smilesColumn, labelColumn):
    
    df_test = pd.read_csv(path + '/' + filename)
    smiles_test = df_test[smilesColumn]
    solubility_test = df_test[labelColumn]
    solubility_test = solubility_test.to_numpy()
    # smile_graph_test = {}
    solubility_arr_test = []
    smiles_array_test = []

    for i,smile in enumerate(smiles_test):
        solubility_arr_test.append(smiles_test[i])
        smiles_array_test.append(smile)
    
    transform = T.Compose([T.NormalizeFeatures(['x', 'edge_attr'])])
    noveltest_data = Molecule_data(root='data/', dataset=datasetname,
                                   y=solubility_test,smiles=smiles_array_test,
                                   transform=transform)
    return noveltest_data
#     iy+=1