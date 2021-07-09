'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2021-07-09 12:31:00
'''
import torch
from torch.utils.data import Dataset

import math
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

class NISTDataset(Dataset):
    def __init__(self, supp, in_dim=1024, out_dim=2000, radius=2):
        self.supp = supp
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius

    def __len__(self):
        return len(self.supp)

    def __getitem__(self, idx):
        mol = self.supp[idx]
        X = torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.in_dim)), dtype=torch.int8)
        Y = self.create_Y(mol, precise=1.0, ms_range=self.out_dim)
        return X, Y

    def create_Y(self, mol, precise, ms_range):
        spectrum = torch.zeros(math.ceil(ms_range / precise))
        for item in mol.GetProp("MASS SPECTRAL PEAKS").split("\n"):
            record = item.split()
            index = int(float(record[0]) / precise)
            spectrum[index] = float(record[1])
        # normalization by sqrt and max intensity
        spectrum = torch.sqrt(spectrum)
        return spectrum / torch.max(spectrum)

class GNPSDataset(Dataset): 
    def __init__(self, supp, in_dim=1024, out_dim=2000, radius=2):
        self.supp = supp
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius

        self.point_sets = []
        self.ms_sets = []
        for _, spectrum in enumerate(self.supp):
            # name = spectrum['params']['name']
            smiles = spectrum['params']['smiles']
            mol = Chem.MolFromSmiles(smiles)
            self.point_sets.append(mol)
            
            x = spectrum['m/z array'].tolist()
            y = spectrum['intensity array'].tolist()
            self.ms_sets.append(self.generate_ms(x, y))

        assert len(self.point_sets) == len(self.ms_sets)

    def __getitem__(self, idx): 
        mol = self.point_sets[idx]
        X = torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.in_dim)), dtype=torch.int8)
        Y = self.ms_sets[idx]
        return X, Y 

    def __len__(self):
        return len(self.point_sets)
    
    def generate_ms(self, x, y):
        '''
        Input:  x   [list denotes the x-coordinate of peaks] (removed parent_ion)
                y   [list denotes the y-coordinate of peaks]
        Return: ms  [list denotes the mass spectra]
        '''
        ms = [0] * self.out_dim # add "0" to y data
        for i, j in enumerate(x): 
            ms[int(float(j)+0.5)] += np.sqrt(float(y[i])) # smooth out larger values using sqrt
            
            # if float(y[i]) == 0: # smooth out larger values using log
            #     continue 
            # ms[int(float(j)+0.5)] += np.log(float(y[i]))
        return ms / np.max(ms) # Normalization, scale the ms to [0, 1]