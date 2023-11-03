import numpy as np
from rdkit import Chem
from typing import Dict, List
import torch
import deepchem as dc
from rdkit.Chem import MolFromSmiles

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import networkx as nx
import os
# from skipatom import AtomVectors
# import selfies as sf
import os
from rdkit.Chem import MACCSkeys
from rdkit import Chem
# from torch_geometric.utils.smiles import from_smiles
from torch_geometric.utils import from_smiles
from torch.nn.functional import normalize
import pickle
import pandas as pd
# from gensim.models import word2vec
# from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
# from torch_geometric.utils.smiles import fromsmiles 
# model = AtomVectors.load("../data/atom2vec.dim20.model")

def generate_features_fromsmile(smile):
    return from_smiles(smile)
    
def smiles_features(mol):
    
    symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg',    #list of all elements in the dataset
        'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 
        'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 
        'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 
        'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 
        'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C',
        'Re','Ta','Ir','Be','Tl']

    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]

    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    features = []
    xs = []
    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.
        #comment degree from 6 to 8
        degree = [0.] * 8
        degree[atom.GetDegree()] = 1.
        formal_charge = atom.GetFormalCharge()
        radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = [0.] * len(hybridizations)
        hybridization[hybridizations.index(
            atom.GetHybridization())] = 1.
        aromaticity = 1. if atom.GetIsAromatic() else 0.
        hydrogens = [0.] * 5
        hydrogens[atom.GetTotalNumHs()] = 1.
        chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
        chirality_type = [0.] * 2
        if atom.HasProp('_CIPCode'):
            chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
    
        x = torch.tensor(symbol + degree + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type)
        xs.append(x)
    
        features = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    
        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[stereos.index(bond.GetStereo())] = 1.
    
        edge_attr = torch.tensor(
            [single, double, triple, aromatic, conjugation, ring] + stereo)
    
        edge_attrs += [edge_attr, edge_attr]
    
    if len(edge_attrs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    return features, edge_index, edge_attr

def smiles_features_skipatom(mol):
    # symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu',import deepchem as dc 'Hg', 'Co', 'Mg',    #list of all elements in the dataset
    #     'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 
    #     'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 
    #     'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 
    #     'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 
    #     'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C',
    #     'Re','Ta','Ir','Be','Tl']

    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]

    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    features = []
    xs = []
    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.
        #comment degree from 6 to 8
        degree = [0.] * 8
        degree[atom.GetDegree()] = 1.
        formal_charge = atom.GetFormalCharge()
        radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = [0.] * len(hybridizations)
        hybridization[hybridizations.index(
            atom.GetHybridization())] = 1.
        aromaticity = 1. if atom.GetIsAromatic() else 0.
        hydrogens = [0.] * 5
        hydrogens[atom.GetTotalNumHs()] = 1.
        chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
        chirality_type = [0.] * 2
        if atom.HasProp('_CIPCode'):
            chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
        # atom_embedding = model.vectors[model.dictionary["Se"]].tolist()
        x = torch.tensor(atom_embedding + degree + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type)
        torch.add(x, .0001)
        xs.append(x)
    
        features = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    
        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[stereos.index(bond.GetStereo())] = 1.
    
        edge_attr = torch.tensor(
            [single, double, triple, aromatic, conjugation, ring] + stereo)
    
        edge_attrs += [edge_attr, edge_attr]
    
    if len(edge_attrs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    return features, edge_index, edge_attr    

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def create_char_to_idx(smiles_arr,
                   max_len: int = 250,
                   smiles_field: str = "smiles") -> Dict[str, int]:
    """Creates a dictionary with character to index mapping.
    Parameters
    ----------
    filename: str
        Name of the file containing the SMILES strings
    max_len: int, default 250
        Maximum allowed length of the SMILES string
    smiles_field: str, default "smiles"
        Field indicating the SMILES strings int the file.
    Returns
    -------
    Dict[str, int]
        A dictionary mapping characters to their integer indexes.
    """
    dict_path = "Data_Prep/vocabulary.pkl"
    
    char_to_idx = {}
    if (os.path.isfile(dict_path)):
        print('Vocabulary already exists, loading vocabulary....')
        file = open(dict_path, "rb")
        char_to_idx = pickle.load(file)
        file.close()
        
    else:
        PAD_TOKEN = "<pad>"
        OUT_OF_VOCAB_TOKEN = "<unk>"
        # smiles_df = pd.read_csv(filename)
        char_set = set()
        for smile in smiles_arr:
            # for smile in smiles_df[smiles_field]:
            # if len(smile) <= max_len:
            char_set.update(set(smile))

        unique_char_list = [PAD_TOKEN, OUT_OF_VOCAB_TOKEN]
        unique_char_list += list(char_set)
        char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}
        file = open(dict_path, "wb")
        pickle.dump(char_to_idx, file)
        file.close()
    return char_to_idx
    

# def smile_to_graph(smile):
#     mol = Chem.MolFromSmiles(smile)
#     if mol == None:
#         return None
#     # fp = MACCSkeys.GenMACCSKeys(mol)
#     # c_size = mol.GetNumAtoms()
#     data = generate_features_fromsmile(smile)
    
#     model = word2vec.Word2Vec.load('models/model_300dim.pkl')
#     # df['mol'] = df[smilesColumn].apply(lambda x: Chem.MolFromSmiles(x))
#     sentence = MolSentence(mol2alt_sentence(mol, 1))
#     # df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
#     mol2vec = sentences2vec(sentence, model, unseen='UNK')
    
#     # edges = []
#     # for bond in mol.GetBonds():
#     #     edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     # g = nx.Graph(edges).to_directed()
#     # edge_index = []
#     # for e1, e2 in g.edges:
#     #     edge_index.append([e1, e2])
#     # Alternative, easier way to convert it to a bitstring. 
#     # smile_fingerprint = [int(x) for x in fp.ToBitString()]
#     # print(len(smile_fingerprint))
#     return c_size, features, edge_index, edge_attr,mol2vec





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
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(y,smiles)
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

    
    
    def process(self, y,smiles):
       
        data_list = []
        data_len = len(y)
        incorrect_smiles = []
        # featurizer = dc.feat.Mol2VecFingerprint()
        char_to_idx = create_char_to_idx(smiles)
        featurizer = dc.feat.SmilesToSeq(char_to_idx,max_len= 210, pad_len= 0)
        
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smile = smiles[i]
            label = y[i]                    
            if len(smile)==1:
                print(f'smile : {smile}')
            smile_fingerprint = featurizer.featurize(smile)
            data = from_smiles(smile)
            data.x = (data.x).type(torch.FloatTensor)
            data.edge_attr = (data.edge_attr).type(torch.FloatTensor)
            if data.edge_attr.shape[0] == 0:
                print('edge attr : ', data.edge_attr.shape)
                print('smile : ', smile)
                incorrect_smiles.append(smile)
                continue
            # if data.edge_attr.size()
            data.y = torch.FloatTensor([label])
            smile_fingerprint = torch.tensor(smile_fingerprint, dtype=torch.float)
            smile_fingerprint = normalize(smile_fingerprint, p=2.0, dim = 0)
            # print(smile_fingerprint)
            data.smile_fingerprint = smile_fingerprint
            # data.mol2vec = mol2vec
            # GCNData = Data(x=torch.Tensor(features),
            #                     edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            #                     y=torch.FloatTensor([labels]))
            graph = data
            
            # graph.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        df = pd.DataFrame(incorrect_smiles, columns = ['SMILES'])
        df.to_csv('incorrect_smiles.csv')
        torch.save((data, slices), self.processed_paths[0])