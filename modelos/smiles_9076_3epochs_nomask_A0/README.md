21/06/2022, 03:41:42
original file: smiles_elemdivided_A0_all
total dataset size: 100000
Layer lstm will use cuDNN kernels when running on GPU.
Layer lstm_1 will use cuDNN kernels when running on GPU.
Layer lstm_2 will use cuDNN kernels when running on GPU.
Found untraced functions such as lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn while saving (showing 5 of 15). These functions will not be directly callable after loading.
Assets written to: ./modelos/model1655779302\model\assets



######################################################################
CÛdigo usado para a geraÁ„o:
######################################################################



import logging
import os
import sys
import time
from datetime import datetime


import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Softmax, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
from tqdm import tqdm


# DATASET_TO_USE = "smiles_chardivided_all"
DATASET_TO_USE = "smiles_elemdivided_A0_all"
# DATASET_TO_USE = "selfies_bracketdivided_all"
DATASET_SIZE = 100000

if __name__ == "__main__":
    acc = int(time.time())
    model_dir = os.path.join("./modelos/", "model" + str(acc))
    os.mkdir(model_dir)

    
    file_handler = logging.FileHandler(filename=os.path.join(model_dir, "README.md"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', handlers=[file_handler,console_handler])
    logger = logging.getLogger("logger")
    
    tqdm.pandas()
    
    logger.debug(datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    logger.info("original file: " + DATASET_TO_USE)
    


    ds_train = pd.read_pickle(os.path.join("./datasets/", DATASET_TO_USE, "ds_train.pk"))
    with open(os.path.join("./datasets/", DATASET_TO_USE, "vocab.pk"), "rb") as vocabpk:
        vocab = pickle.load(vocabpk)
    print(ds_train, vocab)

    def findp(vector):
        if 70 in vector:
            return True
        return False
    
    ds_train = ds_train.sample(frac=1)

    ds_train = ds_train.head(DATASET_SIZE)
    DATASET_SIZE = len(ds_train)
    logger.info("total dataset size: " + str(DATASET_SIZE))
    

    model = Sequential()
    model.add(Embedding(input_dim=len(vocab), output_dim=256, input_length=None))#, mask_zero=True))
    model.add(LSTM(512, return_sequences=True, dropout=0.2))
    model.add(LSTM(512, return_sequences=True, dropout=0.2))
    model.add(LSTM(512, return_sequences=True, dropout=0.2))
    model.add(Dense(len(vocab), activation="softmax"))

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()


    
    x = np.stack(ds_train["X"]) #por incrivel que pare√ßa, parece aprender mais rapido com ints do que com floats
    y = np.stack(ds_train["Y"])
    
    # Early Stopping e Checkpoints
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, mode = "min")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [callback]#, checkpoint]

    model.fit(x, y, epochs=3, batch_size=32, callbacks=callbacks_list)

    model.save(os.path.join(model_dir, "model"))
    with open(__file__, "r") as pycode:
        logger.debug("\n\n\n######################################################################\nC√≥digo usado para a gera√ß√£o:\n######################################################################\n\n\n")
        logger.debug(pycode.read())

    with open(os.path.join(model_dir, "vocab.pk"), "wb") as vocabpk:
        pickle.dump(vocab, vocabpk)


    preds = model.predict(np.array([1]))
    print(preds, np.sum(preds))

