import numpy as np

def turn_to_ints(musicstrings):
    musicints = []
    musicdict = dict((c, i) for (i, c) in enumerate(musicstrings))

    for string in musicstrings:
        notes = string.lower().strip().split()
        phrase = []
        for note in notes:
            if note in musicdict.keys():
                phrase.append(musicdict[note])
        musicints.append(phrases)

    musicarr = np.array(musicints)
    return musicarr
