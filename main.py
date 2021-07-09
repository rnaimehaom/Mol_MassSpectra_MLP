'''
Date: 2021-07-08 16:49:04
LastEditors: yuhhong
LastEditTime: 2021-07-09 14:20:12
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*')
from pyteomics import mgf

from dataset import NISTDataset, GNPSDataset
from model import MLP

def reg_criterion(output, target):
    t = nn.CosineSimilarity()
    return torch.mean(1 - t(output, target))

def train(model, device, loader, optimizer):
    accuracy = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        optimizer.zero_grad()
        model.train()
        pred = model(x)
        
        loss = reg_criterion(pred, y)
        loss.backward()
        optimizer.step()

        accuracy += F.cosine_similarity(pred, y, dim=1).mean().item()

    return accuracy / (step + 1)

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    acc = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x)

        acc.append(F.cosine_similarity(y, pred, dim=1).mean().item())
        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())
        
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    return y_true, y_pred, np.sum(acc)/len(loader)

def batch_filter(supp, out_dim=2000, data_type='sdf'): 
    for _, item in enumerate(supp):
        if data_type == 'mgf':
            smiles = item.get('params').get('smiles')
            if len(smiles) == 0:
                continue
            if len(item.get('m/z array')) == 0 or item.get('m/z array').max() > out_dim: 
                continue
            
        elif data_type == 'sdf':
            mol = item
            if mol is None:
                continue
            if not mol.HasProp("MASS SPECTRAL PEAKS"):
                continue
            if mol.GetProp("SPECTRUM TYPE") != "MS2":
                continue
        yield item

def load_data(data_path, data_type, in_dim, out_dim, radius, num_workers, batch_size): 
    if data_type == 'sdf':
        supp=Chem.SDMolSupplier(data_path)
        dataset = NISTDataset([item for item in batch_filter(supp, out_dim, data_type)], in_dim=in_dim, out_dim=out_dim, radius=radius)
    elif data_type == 'mgf': 
        supp=mgf.read(data_path)
        dataset = GNPSDataset([item for item in batch_filter(supp, out_dim, data_type)], in_dim=in_dim, out_dim=out_dim, radius=radius)
    else:
        print('Data Type Error. Please chooes a dataset from [sdf | mgf].')
        exit()
    
    print('Load {} data from {}.'.format(len(dataset), data_path))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    return data_loader 

def main_mlp():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_mlp_layers', type=int, default=6,
                        help='number of mlp layers (default: 6)')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--in_dim', type=int, default=1024,
                        help='input dimensionality (default: 1024)')
    parser.add_argument('--emb_dim', type=int, default=1600,
                        help='embedding dimensionality (default: 1600)')
    parser.add_argument('--out_dim', type=int, default=2000,
                        help='output dimensionality (default: 2000)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--radius', type=int, default=2,
                        help='radius (default: 2)')

    parser.add_argument('--train_data_path', type=str, default = '', help='path to training data')
    parser.add_argument('--test_data_path', type=str, default = '', help='path to test data')
    parser.add_argument('--data_type', type=str, default = '', help='type of dataset (sdf or mgf)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_path', type=str, default = '', help='path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default = '', help='path to resume checkpoint')
    args = parser.parse_args()
    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    train_loader = load_data(data_path=args.train_data_path, data_type=args.data_type, in_dim=args.in_dim, out_dim=args.out_dim, radius=args.radius, num_workers=args.num_workers, batch_size=args.batch_size)
    valid_loader = load_data(data_path=args.test_data_path, data_type=args.data_type, in_dim=args.in_dim, out_dim=args.out_dim, radius=args.radius, num_workers=args.num_workers, batch_size=args.batch_size)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = MLP(num_mlp_layers=args.num_mlp_layers, in_dim=args.in_dim, emb_dim=args.emb_dim, out_dim=args.out_dim, drop_ratio=args.drop_ratio).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    if args.resume_path is not '':
        # print(torch.load(args.resume_path).keys())
        model.load_state_dict(torch.load(args.resume_path)['model_state_dict'])
        best_valid_acc = torch.load(args.resume_path)['best_val_acc']

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    if args.checkpoint_path is not '':
        checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok = True)

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_acc = 0

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_acc = train(model, device, train_loader, optimizer)
        
        print('Evaluating...')
        _, _, valid_acc = eval(model, device, valid_loader)

        print({'Train': train_acc, 'Validation': valid_acc})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_acc, epoch)
            writer.add_scalar('train/mae', train_acc, epoch)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if args.checkpoint_path is not '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_acc': best_valid_acc, 'num_params': num_params}
                torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()
            
        print(f'Best cosine similarity so far: {best_valid_acc}')

    if args.log_dir is not '':
        writer.close()

if __name__ == "__main__":
    main_mlp()