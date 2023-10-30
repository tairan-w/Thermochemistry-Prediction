import sys
import math
import os
import random
from typing import Union, List, Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class MoleculeDatapoint:
  def __init__(self, smiles: str, targets: List[float]):
    self.smiles = smiles
    self.targets = targets
    
class MoleculeDataset:
  def __init__(self, data: List[MoleculeDatapoint]):
    self.data = data
    
  def smiles(self) -> List[str]:
    return [d.smiles for d in self.data]
  
  def targets(self) -> List[float]:
    return [d.targets for d in self.data]
  
  def shuffle(self, seed: int = None):
    if seed is not None:
      random.seed(seed)
    random.shuffle(self.data)
  
  def __len__(self) -> int:
    return len(self.data)
  
  def __getitem__(self, item) -> MoleculeDatapoint:
    return self.data[item]
    
    
    def get_data(split: str) -> MoleculeDataset:
  data_path = 'delaney_{}.csv'.format(split)
  with open(data_path) as f:
    f.readline()
    data = []
    for line in f:
      line = line.strip().split(',')
      smiles, targets = line[0], line[1:]
      targets = [float(target) for target in targets]
      data.append(MoleculeDatapoint(smiles, targets))
      
  return MoleculeDataset(data)
  train_data, test_data = get_data('train'), get_data('test')
     

ELEM_LIST = range(100)
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
ATOM_FDIM = 100 + len(HYBRID_LIST) + 6 + 5 + 4 + 7 + 5 + 3 + 1
BOND_FDIM = 6 + 6
MAX_NB = 12

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetAtomicNum() - 1, ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + onek_encoding_unk(int(atom.GetImplicitValence()), [0,1,2,3,4,5,6])
            + onek_encoding_unk(int(atom.GetTotalNumHs()), [0,1,2,3,4])
            + onek_encoding_unk(int(atom.GetHybridization()), HYBRID_LIST)
            + onek_encoding_unk(int(atom.GetNumRadicalElectrons()), [0,1,2])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def mol2graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append( atom_features(atom) )
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds) 
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b

    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2

    return fatoms, fbonds, agraph, bgraph, scope

def index_select_ND(source, dim, index):  # convenience method for selecting indices, used in MPN
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)
     

num_epochs = 30
batch_size = 50
lr = .001
     

class MPN(nn.Module):
  def __init__(self):
    super(MPN, self).__init__()
    self.hidden_size = 300
    self.depth = 3
    self.dropout = 0

    self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.hidden_size, bias=False)
    self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.W_o = nn.Linear(ATOM_FDIM + self.hidden_size, self.hidden_size)
    self.W_pred = nn.Linear(self.hidden_size, 1)

  def forward(self, mol_graph) -> torch.FloatTensor:
    fatoms,fbonds,agraph,bgraph,scope = mol_graph

    binput = self.W_i(fbonds)
    message = F.relu(binput)

    for i in range(self.depth - 1):
        nei_message = index_select_ND(message, 0, bgraph)
        nei_message = nei_message.sum(dim=1)
        nei_message = self.W_h(nei_message)
        message = F.relu(binput + nei_message)
        if self.dropout > 0:
            message = F.dropout(message, self.dropout, self.training)

    nei_message = index_select_ND(message, 0, agraph)
    nei_message = nei_message.sum(dim=1)
    ainput = torch.cat([fatoms, nei_message], dim=1)
    atom_hiddens = F.relu(self.W_o(ainput))
    if self.dropout > 0:
        atom_hiddens = F.dropout(atom_hiddens, self.dropout, self.training)

    mol_vecs = []
    for st,le in scope:
        mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
        mol_vecs.append(mol_vec)

    mol_vecs = torch.stack(mol_vecs, dim=0)
    return self.W_pred(mol_vecs)
    
     

model = MPN()
optimizer = optim.Adam(model.parameters(), lr=lr)
     

def param_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
     

print(model)
print('Number of parameters = {:,}'.format(param_count(model)))

def train_epoch(model: nn.Module,
                optimizer: optim.Optimizer,
                data: MoleculeDataset,
                batch_size: int,
                epoch: int) -> float:
  model.train()
  data.shuffle(seed=epoch)
  
  total_loss = 0
  num_batches = 0
  
  data_size = len(data) // batch_size * batch_size  # drop final, incomplete batch
  for i in range(0, data_size, batch_size):
    batch = MoleculeDataset(data[i:i + batch_size])
    mol_graph, targets = mol2graph(batch.smiles()), batch.targets()
    
    targets = torch.FloatTensor(targets)
    
    optimizer.zero_grad()
    preds = model(mol_graph)
    loss = F.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()
    
    total_loss += math.sqrt(loss.item())
    num_batches += 1
    
  avg_loss = total_loss / num_batches
  
  return avg_loss
     

num_epochs = 30
for epoch in range(num_epochs):
  train_loss = train_epoch(model, optimizer, train_data, batch_size, epoch)
  print('Epoch {}: Train loss = {:.4f}'.format(epoch, train_loss))
  
  def rmse(targets: List[float], preds: List[float]) -> float:
    return math.sqrt(mean_squared_error(targets, preds))
     

def evaluate(model: nn.Module, data: MoleculeDataset, batch_size: int):
    model.eval()
    
    all_preds = []
    with torch.no_grad():
      for i in range(0, len(data), batch_size):
        batch = MoleculeDataset(data[i:i + batch_size])
        mol_graph = mol2graph(batch.smiles())
                
        preds = model(mol_graph)
        all_preds.extend(preds)
    
    return rmse(data.targets(), all_preds)
     

test_rmse = evaluate(model, test_data, batch_size)
print('Test rmse = {:.4f}'.format(test_rmse))