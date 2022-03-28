import numpy as np
import re
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

class rnn:

    def __init__(self, corpus, hidden_size = 500, seq_len = 100, wpl = 50, lr = 0.01, decay = 1.005):

        self.preprocess_paragraph(corpus)

        self.chars_to_ix = {ch:i for i,ch in enumerate(self.unique)}
        self.ix_to_char = {i:ch for i, ch in enumerate(self.unique)}

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lr = lr
        self.dr = decay
        self.parameters = self.init_params()
        self.mem = self.init_mem()
        self.wpl = wpl

    def preprocess_paragraph(self, corpus):
        self.corpus = corpus
        self.corpus = re.sub("\n", " ", self.corpus)
        #self.corpus = re.sub("[^a-zA-z 1-9]", "", self.corpus)

        self.data_size = len(self.corpus.split(" "))
        self.unique = list(set(self.corpus.split(" ") + ["\n"]))
        self.vocab_size = len(self.unique)

    def preprocess_wordlist(self, corpus):
        self.corpus = corpus
        self.corpus = [re.sub("[^a-zA-z 1-9]", "", i) for i in self.corpus]

        self.data_size = len(corpus)
        self.unique = list(set(" ".join(corpus).split(" ") + ["\n"]))
        self.vocab_size = len(self.unique)

    def init_params(self):
        params = {}
        params["Wxh"] = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        params["Whh"] = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        params["Why"] = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        params["bh"] = np.zeros((self.hidden_size, 1))
        params["by"] = np.zeros((self.vocab_size, 1))
        return params

    def init_mem(self):
        mem = {}
        mem["Wxh"] = np.zeros((self.hidden_size, self.vocab_size))
        mem["Whh"] = np.zeros((self.hidden_size, self.hidden_size))
        mem["Why"] = np.zeros((self.vocab_size, self.hidden_size))
        mem["bh"] = np.zeros((self.hidden_size, 1))
        mem["by"] = np.zeros((self.vocab_size, 1))
        return mem

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def cell_forward(self, xt, a_prev):
        aa = np.dot(self.parameters["Whh"], a_prev)
        ax = np.dot(self.parameters["Wxh"], xt)
        a_raw = aa + ax + self.parameters["bh"]

        a_next = np.tanh(a_raw)
        yt_hat = self.softmax(np.dot(self.parameters["Why"], a_next) + self.parameters["by"])

        return a_next, yt_hat

    def net_forward(self, X, Y, a_prev):

        caches = {}
        x = {}
        a = {}
        y_hat = {}

        a[-1] = np.copy(a_prev)

        loss = 0

        for t in range(len(X)):
            x[t] = np.zeros((self.vocab_size, 1))
            if X[t] is not None:
                x[t][X[t]] = 1
            a[t], y_hat[t] = self.cell_forward(x[t], a[t-1])
            loss -= np.log(y_hat[t][Y[t], 0])

        cache = (x, a, y_hat)
        return cache, loss

    def cell_back(self, dy, x, a, a_prev, grads):
        grads["dWhy"] += np.dot(dy, a.T)
        grads["dby"] += dy
        da = np.dot(self.parameters["Why"].T, dy) + grads["da_next"]
        daraw = (1 - a * a) * da
        grads["db"] += daraw
        grads["dWxh"] += np.dot(daraw, x.T)
        grads["dWhh"] += np.dot(daraw, a_prev.T)
        grads["da_next"] = np.dot(self.parameters["Whh"].T, daraw)
        return grads

    def net_back(self, X, Y, cache):

        (x, a, y_hat) = cache

        grads = {}
        grads["dWhy"] = np.zeros_like(self.parameters["Why"])
        grads["dby"] = np.zeros_like(self.parameters["by"])
        grads["db"] = np.zeros_like(self.parameters["bh"])
        grads["dWxh"] = np.zeros_like(self.parameters["Wxh"])
        grads["dWhh"] = np.zeros_like(self.parameters["Whh"])
        grads["da_next"] = np.zeros_like(a[0])

        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            grads = self.cell_back(dy, x[t], a[t], a[t-1], grads)

        return grads, a

    def update_params(self, grads):

        for grad in grads.values():
            np.clip(grad, -5, 5, out=grad)

        self.mem["Wxh"] += grads["dWxh"] * grads["dWxh"]
        self.mem["Whh"] += grads["dWhh"] * grads["dWhh"]
        self.mem["Why"] += grads["dWhy"] * grads["dWhy"]
        self.mem["bh"] += grads["db"] * grads["db"]
        self.mem["by"] += grads["dby"] * grads["dby"]

        self.parameters["Wxh"] -= self.lr * grads["dWxh"] / np.sqrt(self.mem["Wxh"] + 1e-8)
        self.parameters["Whh"] -= self.lr * grads["dWhh"] / np.sqrt(self.mem["Whh"] + 1e-8)
        self.parameters["Why"] -= self.lr * grads["dWhy"] / np.sqrt(self.mem["Why"] + 1e-8)
        self.parameters["bh"] -= self.lr * grads["db"] / np.sqrt(self.mem["bh"] + 1e-8)
        self.parameters["by"] -= self.lr * grads["dby"] / np.sqrt(self.mem["by"] + 1e-8)

    def iter(self, ix, targets, h_prev):

        ix = [self.chars_to_ix[i] for i in ix]
        targets = [self.chars_to_ix[j] for j in targets]

        cache, loss = self.net_forward(ix, targets, h_prev)
        grads, a = self.net_back(ix, targets, cache)
        self.update_params(grads)

        return loss, a[len(ix)-1]

    def train(self, iters, sample_index=100):

        losses = []
        ixs = []

        a_prev = np.zeros((self.hidden_size, 1))

        for itera in tqdm(range(iters)):

            index = itera % (len(self.corpus.split()) - self.seq_len)

            x = self.corpus.split()[index:index+self.seq_len]
            y = x[1:] + [x[0]]

            loss, a_prev = self.iter(x, y, a_prev)

            losses.append(loss)
            ixs.append(itera)

            if itera % sample_index == 0:
                #self.lr /= self.dr
                print(loss)
                print(" ".join([self.ix_to_char[i] for i in self.sample(random.randrange(self.vocab_size))]))

        return losses, ixs

    def sample(self, seed, seq=None):

        if seq is None:
            seq = self.wpl
        indices = []
        x = np.zeros((self.vocab_size, 1))
        x[seed] = 1
        a_prev = np.zeros((self.hidden_size, 1))

        for i in range(seq):
            a_prev, probs = self.cell_forward(x, a_prev)
            #print(probs.flatten().shape)

            idx = np.random.choice(list(range(self.vocab_size)), p=probs.ravel())
            indices.append(idx)

            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

        indices.append(self.chars_to_ix['\n'])
        return indices


textname = 'chugjug.txt'

with open(textname, 'r') as t:
    text = t.read()
    text = text.lower()

iters = 50000
whereswaldo = rnn(text)
print("starting now")
y, x = whereswaldo.train(iters)
plt.plot(x, y)
plt.show()
