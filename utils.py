import numpy as np
import random
import math

def noisify(set, diff=0.5):
    noisy = set + np.random.normal(0, diff, size=set.shape)
    return noisy

def clip(set):
    set = np.max(0, set)
    set = np.min(1, set)
    return set

def randomselect(set, number=5):
    total = len(set)
    randchoices = []
    randindices = []
    for i in range(number):
        total -= 1
        ind = random.randint(0, total)
        randindices.append(ind)
        randchoices.append(set[ind])
    return randchoices, randindices

def one_hot_encode(poggers):
    unique = list(set(poggers))
    udict = dict((ind,item) for ind, item in enumerate(sorted(unique)))
    out_hot = np.zeros((len(poggers), len(udict)))
    for pog, champ in enumerate(poggers):
        out_hot[pog, udict[champ]] = 1
    return out_hot

def divide_and_shuffle(set1, set2, size=0.5):
    together = list(zip(list(set1), list(set2)))
    random.shuffle(together)
    if size is not None:
        lenlarge = math.floor(len(together) * (1 - size))
        largenew = together[:lenlarge]
        smallnew = together[lenlarge:]
        newlargex = [large[0] for large in largenew]
        newlargey = [large[1] for large in largenew]
        newsmallx = [small[0] for small in smallnew]
        newsmally = [small[1] for small in smallnew]
        return np.array(newlargex), np.array(newsmallx), np.array(newlargey), np.array(newsmally)
