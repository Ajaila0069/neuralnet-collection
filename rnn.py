import numpy as np
import re
from tqdm import tqdm

class rnn:

    def __init__(self, corpus, y=None, a_size=1):
        self.corpus = corpus
        self.corpus = [re.sub("[^a-zA-z 1-9]", "", i) for i in self.corpus]
        self.y = y
        self.a_size = a_size

        self.words = ""
        for sentence in self.corpus:
            self.words += sentence + " "

        self.unique = list(set(self.words.split()))
        self.x_size = len(self.unique)
        self.x = []
        for i in self.corpus:
            self.x.append(self.one_hot_encode(self.unique, i))

        self.parameters = self.init_params(self.x_size, self.a_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsig(self, x):
        y = sefl.sigmoid(x)
        return y * (1 - y)

    def one_hot_encode(self, unique, corpus):
        udict = dict((ind,item) for ind, item in enumerate(sorted(unique)))
        out_hot = np.zeros((len(corpus), len(udict)))
        for pog, champ in enumerate(corpus):
            out_hot[pog, udict[champ]] = 1
        return out_hot

    def init_params(self, x_size, a_size):
        parameters = {}

        parameters["WAX"] = np.random.randn(a_size, x_size)
        parameters["WAA"] = np.random.randn(a_size, a_size)
        parameters["BA"] = np.zeros((a_size, 1))

        parameters["WAI"] = np.random.randn(a_size, 1)
        parameters["BAI"] = np.zeros((1, 1))

        return parameters

    def update_parameters(self, gradients, lr):

        self.parameters['WAX'] += -lr * gradients['dWax']
        self.parameters['WAA'] += -lr * gradients['dWaa']
        self.parameters['WAI'] += -lr * gradients['dWai']
        self.parameters['BA'] += -lr * gradients['dBa']
        self.parameters['BAI'] += -lr * gradients['dBi']

    def cell_forward(self, x, a_prev):
        aa = np.dot(self.parameters["WAA"], a_prev)
        ax = np.dot(self.parameters["WAX"], x)
        a_raw = aa + ax + self.parameters["BA"]
        a_next = np.tanh(a_raw)
        return a_next

    def interp_forward(self, a_prev):
        z = np.dot(a_prev, self.parameters["WAI"]) + self.parameters["BAI"]
        a = self.sigmoid(z)
        return (a_prev, a, z)

    def net_forward(self, x, a_prev=None):
        if not a_prev:
            a_prev = np.zeros((1, self.a_size))

        caches = []
        for word in x:
            a_next = self.cell_forward(x, a_prev)
            cache = (x, a_prev, a_next)
            a_prev = a_next
            caches.append(cache)

        finalcache = self.interp_forward(a_prev)

        return caches, finalcache

    def cell_back(self, cache, da_prev):
        (x_t, a_prev, a_next) = cache
        gradients = {}

        dtanh = (1 - (a_next * a_next)) * da_prev
        dWAX = np.dot(x_t.T, dtanh)
        dWAA = np.dot(a_prev.T, a_next)
        dBA = np.sum(dtanh, keepdims=True, axis=-1)
        da = np.dot(self.parameters["WAA"].T, dtanh)

        gradients["dWax"], gradients["dWaa"], gradients["dBa"], gradients["da"] = dWAX, dWAA, dBA, da
        return gradients

    def interp_back(self, loss, a, z, a_prev):
        da = loss * self.dsig(z)
        dz = np.dot(da, self.parameters["WAI"].T)
        dw = np.dot(a_prev.T, da)
        db = np.sum(dz, keepdims=True)
        grads = {"dWai" : dw, "dBi" : db}
        return dz, grads

    def net_back(self, caches, finalcache, y):
        (a_prev, y_hat, z) = finalcache
        loss = y - y_hat
        da_prev, grads = self.interp_back(loss, y_hat, z, a_prev)

        gradsums = []

        for i, cache in reversed(enumerate(caches)):
            gradsums.append(self.cell_back(cache, da_prev))
            da_prev = gradsums[-1]["da"]

        grad_final = {"dWax" : 0, "dWaa" : 0, "dBa" : 0}

        for i in gradsums:
            grad_final["dWax"] += i["dWax"]
            grad_final["dWaa"] += i["dWaa"]
            grad_final["dBa"] += i["dBa"]

        grad_final.update(grads)

        return grad_final, loss

    def train(self, epochs):

        for i in range(epochs):
            losses = []
            for sentence, rating in tqdm(zip(self.x, self.y)):
                caches, finalcache = self.net_forward(x)
                grads, loss = self.net_back(caches, finalcache, y)
                self.update_parameters(grads)
                losses.append(loss**2)
            print(f"Epoch: {i} | Sum of Squared Losses: {sum(losses)}")


whereswaldo = rnn()
