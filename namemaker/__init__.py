import string
from sklearn.utils import check_random_state

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
