from rdkit import Chem
import numpy as np
from atom import get_atom_features
from bond import get_bond_features

def mol_to_graph(mol):
    """
    Converts an RDKit molecule object to a graph representation.
    :param mol: RDKit molecule object.
    :return: A dictionary containing atom features, bond features, and adjacency information.
    """
    if mol is None:
        return None

    atom_features = []
    bond_features = []
    adj_list = []

    # Generate atom features
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    # Generate bond features and adjacency list
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        adj_list.append((start_idx, end_idx))
        adj_list.append((end_idx, start_idx))

        bond_feat = get_bond_features(bond)
        bond_features.append(bond_feat)
        bond_features.append(bond_feat)  # undirected graph (both directions)

    atom_features = np.array(atom_features, dtype=np.float32)
    bond_features = np.array(bond_features, dtype=np.float32)
    adj_list = np.array(adj_list, dtype=np.int64)

    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'adj_list': adj_list
    }

# Function to convert SMILES to graph

def smiles_to_graph(smiles):
    """
    Converts SMILES to graph representation using RDKit.
    :param smiles: SMILES string.
    :return: Graph representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_graph(mol)
