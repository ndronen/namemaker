import string

from keras.layers import Input, Embedding, GRU, Dense, merge
from keras.models import Model
from keras.constraints import unitnorm
import keras.callbacks as cb

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

class Model(object):
    def __init__(self):
        pass

    def load_data(self):
        raise NotImplementedError()
    
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
        examples = []
        for word in words:
            for i in range(len(word)):
                for j in range(config['window_size']):
                    print(word, i, j, word[i+j])
    
    def build_y_language_model(self, X):
        pass
    
    def build_char_language_model(self, config):
        batch_shape = (config['batch_size'], config['maxlen'])
        input = Input(batch_shape=batch_shape, dtype='int32')
    
        embed = Embedding(
                input_dim=config['n_embeddings'],
                output_dim=config['n_embed_dims'],
                input_length=config['maxlen'], 
                mask_zero=True,
                W_constraint=unitnorm())
        x = embed(input)
    
        x = GRU(config['n_recurrent_units'],
                stateful=True,
                mask_zero=True)
    
        x = Dense(config['n_dense_units'], activation='relu')(x)
    
        output = Dense(config['n_embeddings'], activation='softmax')(x)
    
        model = Model(input=input, output=output)
        model.compile(optimizer=config['optimizer'],
                loss='categorical_crossentropy')
    
        return model
    
    
    def build_valence_model(self, config):
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
