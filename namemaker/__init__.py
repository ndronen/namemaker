import string

from keras.layers import Input, Embedding, GRU, Dense, merge
from keras.models import Model
from keras.constraints import unitnorm
import keras.callbacks as cb

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

def load_data():
    return pd.read_csv('data/Ratings_Warriner_et_al.csv.gz', encoding='utf8')

def build_index(words):
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

def build_config(df, index):
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

def build_X_valence(words, config, index):
    X = np.zeros((len(words), config['maxlen']), dtype='int32')
    for i,word in enumerate(words):
        for j,ch in enumerate(word):
            X[i,j] = index[ch]
    return X

def build_y_valence(values):
    mean = values.mean()
    sd = values.std()
    values -= mean
    values /= sd
    return values

def build_X_language_model(words, config, index):
    examples = []
    for word in words:
        for i in range(len(word)):
            for j in range(config['window_size']):
                print(word, i, j, word[i+j])

def build_y_language_model(X):
    pass

def build_char_language_model(config):
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


def build_valence_model(config):
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

def build_callbacks(config):
    callbacks = []
    callbacks.append(cb.EarlyStopping(
        monitor='loss', patience=config['patience']))
    return callbacks

def changename(name, chars, random_state):
    chars = list(chars)
    # Don't change the length of the name.
    chars.remove(' ')
    for ch in string.punctuation:
        try:
            chars.remove(ch)
        except ValueError:
            pass

    # Choose an index to change.
    i = random_state.choice(len(name))
    # Choose a new character.
    newchar = random_state.choice(chars)

    # Replace character at index with new character.
    return name[:i] + newchar + name[i+1:]

def makenames(model, config, index, nnames, nchars=0, initname=None, random_state=17):
    assert nchars != 0 or initname is not None
    assert isinstance(nnames, int) and nnames > 0

    random_state = check_random_state(random_state)

    chars = list(index.keys())
    chars.sort()

    if initname is not None:
        name = initname
    else:
        # Choose nchars chars at random.
        name = random_state.choice(chars, nchars)
        name = ''.join(name)

    seen = set()
    names = []

    for i in range(nnames):
        X = build_X([name], config, index)
        y = model.predict(X)
        names.append((name,y))
        name = changename(name, chars, random_state)

    return names
