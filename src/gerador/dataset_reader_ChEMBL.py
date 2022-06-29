"""O tf fornece uma API própria para lidar com datasets: tf.data.Datasets. Tem uma série de vantagens como a execução lazy de funções, o que poupa memória para datasets grandes. No entanto, está em constante alteração e a documentação ainda é muito fraca (e técnica) pelo que a abordagem escolhida foi usar o pandas para preprocessar os datasets que depois são guardados em ficheiros .tfrecords. Além disso os dataframes do pandas permitem acompanhar o tamanho do dataset à medida que é processado. Estes podem depois ser lidos pelos scripts de cada modelo que assim poupa no pré-processamento, obtendo um objeto tf.data.Dataset com as vantagens associadas. A perda de desempenho com o pandas não parece ser decisiva. UPDATE: contando apenas o preprocessamento, a velocidade do pandas até é maior cerca de 1.5x, usando o tqdm para testes"""

from copyreg import pickle
from datetime import datetime
import os
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import nltk
import logging
import sys
import pickle
from nltk import tokenize as tk
import selfies as sf



# DATASET_TO_USE = "test_dataset.csv"
DATASET_TO_USE = "tudo_ro5_small_less400.csv"


def remove_isotopes(smile):
    if "[" not in smile: #aumenta muito a performance
        return smile
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


def pad_to_len(smile: np.ndarray, target_len):
    sz = len(smile)
    if sz > target_len:
        return np.nan
    return np.pad(smile, (0, target_len - len(smile)), 'constant')

def x_to_y(smile: np.ndarray):
    return np.delete(np.pad(smile, (0, 1), 'constant'), 0)

def encode(smile: str):
    pass

#TODO tentar implementar um subword tokenizer, poara ver se grupos como carboxilos e etc aparecem separados

class ElementarTokenizer():
    """Tokeniza os smiles tendo em conta:
        os elementos: CCClO -> C C Cl O;
        as cargas: C[N+]C -> C [N+] C;
        e em geral tudo dentro de brackets, como info estereoquimica e coisas como [nH]"""
    
    def __init__(self) -> None:
        tab_periodica = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
         'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
         'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
         'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
         'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
         'Po', 'At', 'Rn', 'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
         'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
         'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
         'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        self.mwtok = tk.MWETokenizer(tab_periodica, "")
        self.brackettok = tk.RegexpTokenizer('\[([^\[]*?)\]')
        
    def tokenize(self, smile: str):
        for exp in self.brackettok.tokenize(smile):
            self.mwtok.add_mwe("[" + exp + "]")
        return self.mwtok.tokenize(smile)

class RecSExprTokenizer(): # TODO UPDATE NAO É NECESSARIO FAZER ISTO, USA UMA REGEX SEMELHANTE À DE CIMA
    """Variante recursiva de nltk.tokenize.SExprTokenizer.
        Deve devolver uma lista de tokens cujos elementos são apenas as expressões englobadas pelos parenteses especificados em parens mais internas, no caso de haver parenteses nested. Exemplo: COc1cccc(S(=O)(=O)NC(=O)COC2CCCCC2)c1 -> COc1cccc"""
    def __init__(self, parens="()") -> None:
        self.parens = parens
        self.tok = tk.SExprTokenizer(parens=parens)
    
    def tokenize(self, smile: str):
        #caso base
        if len(self.tok.tokenize(smile.strip(self.parens))) == 1: #não ha mais parenteses nested
            return smile

    
    
    
    



if __name__ == "__main__":
    acc = int(time.time())
    dataset_dir = os.path.join("./datasets/", "dat" + str(acc))
    while(os.path.exists(dataset_dir)):
        acc += 1
        dataset_dir = os.path.join("./datasets/", "dat" + str(acc))
    os.mkdir(dataset_dir)

    file_handler = logging.FileHandler(filename=os.path.join(dataset_dir, "README.md"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', handlers=[file_handler,console_handler])
    logger = logging.getLogger("logger")

    logger.debug(datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    logger.info("original file: " + DATASET_TO_USE)
    
    tqdm.pandas()

    ds_train = pd.read_csv(os.path.join("./raw_data/", DATASET_TO_USE), sep=";")#, nrows=10000)

    # ds_train = ds_train.sample(frac=1)
    # ds_train = ds_train.take(200000)

    # Retirar compostos demasiado simples
    ds_train = ds_train.loc[ds_train["Heavy Atoms"] > 5]

    # Retirar sais
    ds_train = ds_train.loc[-ds_train["Smiles"].str.contains(".", regex=False)]

    ds_train = ds_train[["Smiles"]]
    

    # Trocar a presença de isótopos marcados nos smiles para os base. Só alteram o número de massa, logo não têm influência nas propriedades químicas
    print("A remover isótopos...")
    ds_train["Smiles"] = ds_train["Smiles"].progress_apply(remove_isotopes)

    # Remover duplicados que possam existir
    ds_train = ds_train.drop_duplicates(keep="first")

    # Passar para SELFIES
    print("A transformar em SELFIES...")
    ds_train["Selfies"] = ds_train["Smiles"].progress_apply(sf.encoder, strict=False)

    # Tokenizar
    print("A tokenizar...")
    tokenizer = ElementarTokenizer()
    ds_train["Tokenized Selfies"] = ds_train["Selfies"].progress_apply(tokenizer.tokenize)
    
    # Obter vocabulário
    vocab = {token: index + 2 for (index, token) in enumerate(ds_train["Tokenized Selfies"].explode().unique())} # tem que ser index + 2 para guardar o zero para o pad e o 1 para o G. O zero pode ser ignorado pela lstm e acelera
    vocab.update({'A': 0, 'G': 1})
    logger.info(vocab)
    
    # Acrescentar START e STOP
    print("A acrescentar START E STOP...")
    ds_train["Tokenized Selfies"] = ds_train["Tokenized Selfies"].progress_apply(np.pad, pad_width=(1, 1), mode='constant', constant_values=('G', 'A'))

    # Codificar para inteiros
    print("A codificar para inteiros...")
    ds_train["Encoded Selfies"] = ds_train["Tokenized Selfies"].progress_apply(lambda x: [vocab[t] for t in x])

    # Pad
    print("A acrescentar padding de zeros...")
    TARGET_LEN = 100
    ds_train["Encoded Selfies"] = ds_train["Encoded Selfies"].progress_apply(pad_to_len, target_len=TARGET_LEN)
    ds_train = ds_train.dropna()

    # Criar X e Y
    print("A criar X e Y...")
    ds_train["X"] = ds_train["Encoded Selfies"]
    ds_train["Y"] = ds_train["Encoded Selfies"].progress_apply(x_to_y)

    # Reduzir ao necessário e finalizar
    ds_train = ds_train[["X", "Y", "Selfies", "Smiles"]]

    # Guardar dataset
    logger.info(ds_train)
    
    with open(__file__, "r") as pycode:
        logger.debug("\n\n\n######################################################################\nCódigo usado para a geração:\n######################################################################\n\n\n")
        logger.debug(pycode.read())

    with open(os.path.join(dataset_dir, "vocab.pk"), "wb") as vocabpk:
        pickle.dump(vocab, vocabpk)
    ds_train.to_pickle(os.path.join(dataset_dir, "ds_train.pk"))