from rdkit import Chem
import numpy as np

# Function to calculate bond features

def get_bond_features(bond):
    features = []

    # Bond type (one-hot)
    bond_type = [0] * 4
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    bond_type_idx = bond_type_map.get(bond.GetBondType(), 0)
    bond_type[bond_type_idx] = 1
    features.extend(bond_type)

    # Conjugated
    features.append(int(bond.GetIsConjugated()))

    # In ring
    features.append(int(bond.IsInRing()))

    # Stereo (one-hot)
    stereo = [0] * 6
    stereo_idx = int(bond.GetStereo())
    if 0 <= stereo_idx < 6:
        stereo[stereo_idx] = 1
    features.extend(stereo)

    # Rotatable bond
    features.append(int(bond.IsRotor()))

    # Ring size (one-hot, max 6)
    ring_size = [0] * 6
    if bond.IsInRing():
        ring_info = bond.GetOwningMol().GetRingInfo()
        for ring in ring_info.BondRings():
            if bond.GetIdx() in ring:
                size = min(len(ring), 6) - 1
                ring_size[size] = 1
                break
    features.extend(ring_size)

    # Hydrogen bonding (one-hot)
    hydrogen_bonding = [0] * 2
    if bond.GetBeginAtom().GetAtomicNum() in [7, 8] or bond.GetEndAtom().GetAtomicNum() in [7, 8]:
        hydrogen_bonding[1] = 1  # Yes
    else:
        hydrogen_bonding[0] = 1  # No
    features.extend(hydrogen_bonding)

    # Local environment (one-hot, max 4 bonds considered)
    local_env = [0] * 4
    for idx, neighbor in enumerate(bond.GetBeginAtom().GetBonds()):
        if idx < 4:
            local_env[idx] = neighbor.GetBondTypeAsDouble()
    features.extend(local_env)

    return np.array(features)
