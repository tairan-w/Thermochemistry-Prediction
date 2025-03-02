from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import numpy as np


# Function to calculate atomic features

def get_atom_features(atom):
    features = []

    # Atom type (one-hot)
    atom_type = [0] * 100
    atom_type[atom.GetAtomicNum() - 1] = 1
    features.extend(atom_type)

    # Formal charge (one-hot)
    formal_charge = [0] * 5
    charge = atom.GetFormalCharge() + 2  # range [-2, 2]
    if 0 <= charge < 5:
        formal_charge[charge] = 1
    features.extend(formal_charge)

    # Chirality (one-hot)
    chirality = [0] * 4
    chirality_tag = atom.GetChiralTag().real
    if 0 <= chirality_tag < 4:
        chirality[chirality_tag] = 1
    features.extend(chirality)

    # Hydrogen atoms (one-hot)
    num_hydrogens = [0] * 5
    h_count = min(atom.GetTotalNumHs(), 4)
    num_hydrogens[h_count] = 1
    features.extend(num_hydrogens)

    # Hybridization (one-hot)
    hybridization = [0] * 5
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4
    }
    hybrid_type = hybrid_map.get(atom.GetHybridization(), 2)  # Default to SP3
    hybridization[hybrid_type] = 1
    features.extend(hybridization)

    # Aromaticity
    features.append(int(atom.GetIsAromatic()))

    # Atomic mass (normalized)
    features.append(atom.GetMass() / 100.0)

    # Number of bonds (one-hot)
    bond_count = [0] * 6
    num_bonds = min(len(atom.GetBonds()), 5)
    bond_count[num_bonds] = 1
    features.extend(bond_count)

    # Electronegativity (normalized by 4)
    electronegativity = Descriptors.MolLogP(atom.GetOwningMol()) / 4.0
    features.append(electronegativity)

    # Ionization energy (normalized by 15.76 eV for H)
    ionization_energy = Descriptors.MolMR(atom.GetOwningMol()) / 15.76
    features.append(ionization_energy)

    # Partial charge (using Gasteiger method)
    Chem.rdPartialCharges.ComputeGasteigerCharges(atom.GetOwningMol())
    partial_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
    features.append(partial_charge)

    # Donor/Acceptor (one-hot)
    is_donor = int(atom.GetAtomicNum() in [7, 8])
    is_acceptor = int(atom.GetAtomicNum() in [7, 8])
    features.extend([is_donor, is_acceptor])

    # Saturation degree (one-hot, max 5)
    saturation = [0] * 6
    single_bonds = sum(1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.SINGLE)
    saturation[min(single_bonds, 5)] = 1
    features.extend(saturation)

    # Atomic volume (normalized)
    atomic_volume = atom.GetAtomicNum() / 100.0
    features.append(atomic_volume)

    return np.array(features)
