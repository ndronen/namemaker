import string

from keras.layers import Input, Embedding, GRU, Dense, merge
from keras.models import Model
from keras.constraints import unitnorm
import keras.callbacks as cb

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

def Model(object):
    def __init__(self):
        pass

    def load_data(self):
        datapath = 'data/Ratings_Warriner_et_al.csv.gz'
        return pd.read_csv(datapat, encoding='utf8')

    def build_index(self, words):
        chars = set()
        for word in words:
            chars.update(word)

        chars = list(chars)
        chars.sort()
        chars.insert(0, '<ZERO>')

        index = dict(zip(chars, range(len(chars))))
        for metachar in ['^', '$']:
            if metachar not in index:
                index[metachar] = len(index)

        return index

    def build_config(self, df, index):
        md = {}
        md['maxlen'] = max(df.Word.apply(len)) + 2
        md['minlen'] = min(df.Word.apply(len)) + 2
        md['batch_size'] = 32
        md['n_embeddings'] = len(index)
        md['n_embed_dims'] = 25
        md['n_recurrent_units'] = 50
        md['n_dense_units'] = 10
        md['optimizer'] = 'adam'
        md['patience'] = 2
        return md

    def build_X(self, words, config, index):
        X = np.zeros((len(words), config['maxlen']), dtype='int32')
        for i,word in enumerate(words):
            for j,ch in enumerate(word):
                X[i,j] = index[ch]
        return X

    def build_y(self, values):
        mean = values.mean()
        sd = values.std()
        values -= mean
        values /= sd
        return values
    
    def build_model(self, config):
        input = Input(shape=(config['maxlen'],), dtype='int32')
    
        embed = Embedding(
                input_dim=config['n_embeddings'],
                output_dim=config['n_embed_dims'],
                input_length=config['maxlen'], 
                mask_zero=True,
                W_constraint=unitnorm())
        x = embed(input)
    
        x = GRU(config['n_recurrent_units'],
                return_sequences=False)(x)
    
        x = Dense(config['n_dense_units'], activation='relu')(x)
    
        output = Dense(1, activation='linear')(x)
    
        model = Model(input=input, output=output)
        model.compile(optimizer=config['optimizer'],
                loss='mean_squared_error')
    
        return model
    
    def build_callbacks(self, config):
        callbacks = []
        callbacks.append(cb.EarlyStopping(
            monitor='loss', patience=config['patience']))
        return callbacks
    
