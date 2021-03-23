import numpy as np
import random
import math
import re

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

def substrings(main, size, step = 1):
    main = re.sub("\n+", " ", main)
    main = re.sub("[^a-zA-z 1-9]", "", main)
    corpus = main.split()
    length = len(corpus)
    slices = []
    #print(((length - (size-1))//step))
    for i in range(0, ((length - (size-1))//step)):
        idx = i * step
        try:
            slices.append(corpus[idx:idx+size])
        except Exception as e:
            print(e)
    return slices


if __name__ == "__main__":
    print(substrings(open("chugjug.txt", "r").read(), 3, step=2))
