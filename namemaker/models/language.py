import string
import unittest

from keras.layers import Input, Embedding, GRU, Dense, merge
from keras.models import Model
from keras.constraints import unitnorm
import keras.callbacks as cb

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

from .. import build_index

class Model(object):
    def __init__(self):
        pass

    def load_data(self, minlen=4):
        df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf')
        words = df.word.tolist()
        return [w for w in words if len(w) >= minlen]
    
    def build_config(self, words, index):
        md = {}
        md['batch_size'] = 32
        md['window_size'] = 4
        md['n_embeddings'] = len(index)
        md['n_embed_dims'] = 25
        md['n_recurrent_units'] = 50
        md['n_dense_units'] = 10
        md['optimizer'] = 'adam'
        md['patience'] = 2
        return md
    
    def build_Xy(self, words, config, index):
        char_seqs = []
        next_chars = [] 

        for word in words:
            for i in range(len(word)):
                try:
                    char_seq = word[i:i+config['window_size']]
                    next_char = word[i+config['window_size']]
                    char_seqs.append(char_seq)
                    next_chars.append(next_char)
                except IndexError:
                    break

        X = np.zeros((len(char_seqs), config['window_size']), dtype='int32')
        y = np.zeros((len(next_chars), 1), dtype='int32')

        for i,chs in enumerate(char_seqs):
            for j, ch in enumerate(chs):
                X[i,j] = index[ch]

        for i,next_char in enumerate(next_chars):
            y[i] = index[next_char]

        return X, y

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
    
    def build_callbacks(self, config):
        callbacks = []
        callbacks.append(cb.EarlyStopping(
            monitor='loss', patience=config['patience']))
        return callbacks

class TestLanguageModel(unittest.TestCase):
    def test_language_model(self):
        model = Model()
        words = ["^aardvark$"]
        index = build_index(words)
        rindex = dict((v,k) for k,v in index.items())
        config = model.build_config(words, index)
        config['window_size'] = 4

        X, y = model.build_Xy(words, config, index)

        expected_X = [ "^aar", "aard", "ardv", "rdva", "dvar", "vark"]
        expected_y = [ "d", "v", "a", "r", "k", "$" ]

        self.assertEqual(len(expected_X), len(X))
        self.assertEqual(len(expected_y), len(y))

        for i,char_seq in enumerate(expected_X):
            reconstructed = ''
            for idx in X[i]:
                reconstructed += rindex[idx]
            self.assertEqual(char_seq, reconstructed)

        for i,next_char in enumerate(expected_y):
            idx = y[i].item()
            self.assertEqual(next_char, rindex[idx])


if __name__ == '__main__':
    unittest.main()
