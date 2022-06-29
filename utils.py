from rdkit import Chem



def remove_isotopes(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_block = Chem.MolToMolBlock(mol)
    new_mol_block = []
    for line in mol_block.split('\n'):
        if 'M  ISO' not in line:
            new_mol_block += [line]
    new_mol_block = '\n'.join(new_mol_block)
    new_mol = Chem.MolFromMolBlock(new_mol_block)
    new_smile = Chem.MolToSmiles(new_mol)
    return(new_smile)



if __name__ == "__main__":
    from rdkit.Chem import Draw
    from rdkit.Chem.rdCoordGen import AddCoords

    smile = r"Cc1c(CN[C@H]2CC[C@@H](F)C2)nn(C)c1-c1cc(F)c(F)c(F)c1"
    new_smile = remove_isotopes(smile)
    mol = Chem.MolFromSmiles(smile)
    new_mol = Chem.MolFromSmiles(new_smile)
    AddCoords(mol)
    AddCoords(new_mol)
    print("original:", smile)
    print("no isotopes:", new_smile)
    img = Draw.MolsToGridImage(mols=[mol, new_mol], legends=['original', 'no isotopes'])
    img.show()
