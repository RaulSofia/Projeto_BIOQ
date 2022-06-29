import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle as pk
import scipy as sp
import selfies as sf
from tqdm import tqdm

from mostrador.mostrador import Mostrador

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_TO_USE = "selfies_9017_3epochs_nomask_A0"
from_selfies = True


def predict_molecule(model, vocab, cap_size=100, temp=0.75, from_selfies=False):

    start_tok = "G"

    mol = [vocab[start_tok]] # parece haver problemas de performance ao usar o numpy.array()
    next = __predict_next_token(model, mol)
    mol += next
    while(len(mol) < cap_size and next != [0]):
        next = __predict_next_token(model, mol)
        mol += next
    return decode(vocab, mol, from_selfies)


def __predict_next_token(model: tf.keras.Model, mol):
    preds = model.predict(mol)[-1][0]
    tok = np.random.choice(len(preds), p=preds)
    return [tok]

def decode(vocab, mol:list, from_selfies):
    decode_dict = {v: k for k, v in vocab.items()}
    decoded_list = [decode_dict[x] for x in mol]
    smiles = "".join(decoded_list).strip("GA")
    if from_selfies:
        smiles = sf.decoder(smiles)
    return smiles
    
def predict_molecules(model, vocab, cap_size=100, temp=0.75, n=50, from_selfies=False):
    mols = []
    for i in tqdm(range(50)):
        mols.append(predict_molecule(model, vocab, cap_size, temp, from_selfies=from_selfies))
    return mols


if __name__ == '__main__':
    # modeldir = os.path.join("./modelos/", MODEL_TO_USE)
    # model = load_model(os.path.join(modeldir, "model"))
    # with open(os.path.join(modeldir, "vocab.pk"), "rb") as f:
    #     vocab = pk.load(f)
    # print(predict_molecules(model, vocab, from_selfies=from_selfies, n=50))
    disp = Mostrador()
    data = pd.read_csv("./raw_data/test_dataset.csv", sep=";")
    disp.add(data, subtitle="AlogP")
    disp.add(data, subtitle=["AlogP", "Molecular Weight", "Aromatic Rings"])
    # disp.add(data)
    # disp.add(data, title="AlogP", subtitle="Smiles")
    # print(disp.data[["_subtitulo", "_titulo"]])
    # print(data)
    # print(disp.data)
    disp.render()#sort_by="AlogP")
    disp.show()