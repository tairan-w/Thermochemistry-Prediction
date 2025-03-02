from rdkit.Chem import rdMolDescriptors, rdPartialCharges

@register_features_generator('extended_atom_features')
def extended_atom_features_generator(mol: Molecule) -> np.ndarray:
    """Generates extended atom features for a molecule."""
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    atom_features = []

    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),                            # Atom type
            atom.GetFormalCharge(),                          # Formal charge
            int(atom.GetChiralTag()),                        # Chirality
            atom.GetTotalNumHs(),                            # Hydrogen atoms
            int(atom.GetHybridization()),                    # Hybridization
            int(atom.GetIsAromatic()),                       # Aromaticity
            atom.GetMass() / 100,                            # Atomic mass (normalized)
            len(atom.GetBonds()),                            # Number of bonds
            rdMolDescriptors.CalcCrippenContribs(mol)[atom.GetIdx()][0],  # Partial charge
            float(rdMolDescriptors.CalcTPSA(mol)),           # TPSA
            rdPartialCharges.ComputeGasteigerCharges(mol),   # Gasteiger partial charges
        ]
        atom_features.extend(features)
    return np.array(atom_features)

@register_features_generator('extended_bond_features')
def extended_bond_features_generator(mol: Molecule) -> np.ndarray:
    """Generates extended bond features for a molecule."""
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    bond_features = []

    for bond in mol.GetBonds():
        features = [
            int(bond.GetBondTypeAsDouble()),                 # Bond type
            int(bond.GetIsConjugated()),                     # Conjugation
            int(bond.IsInRing()),                            # In ring
            int(bond.GetStereo()),                           # Stereochemistry
            int(bond.IsRotor()),                             # Rotatable bond
            rdMolDescriptors.CalcNumRotatableBonds(mol),     # Number of rotatable bonds
        ]
        bond_features.extend(features)
    return np.array(bond_features)


@MoleculeFeaturizerRegistry("extended_features")
class ExtendedFeaturizer(VectorFeaturizer[Mol]):
    """Generates extended molecular features using RDKit."""

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        atom_features = extended_atom_features_generator(mol)
        bond_features = extended_bond_features_generator(mol)

        # 合并原子和键的特征
        features = np.concatenate([atom_features, bond_features])
        return features

    def __len__(self) -> int:
        # 计算总特征长度，假设每种特征长度固定
        atom_feature_length = 12  # 假设每个原子有 12 个特征
        bond_feature_length = 6   # 假设每个键有 6 个特征
        max_atoms = 50            # 假设最多有 50 个原子
        max_bonds = 50            # 假设最多有 50 个键

        return atom_feature_length * max_atoms + bond_feature_length * max_bonds


@MoleculeFeaturizerRegistry("extended_rdkit_2d")
class ExtendedRDKit2DFeaturizer(VectorFeaturizer[Mol]):
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        # 添加扩展特征
        extended_features = ExtendedFeaturizer()(mol)
        features = np.concatenate([features, extended_features])
        return features

    def __len__(self) -> int:
        original_length = len(Descriptors.descList)
        extended_length = ExtendedFeaturizer().__len__()
        return original_length + extended_length
