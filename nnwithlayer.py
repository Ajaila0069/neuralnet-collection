import numpy as np
import math
import random
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    iris = load_digits()
    x_train1, x_train2, y_train1, y_train2 = divide_and_shuffle(iris.data[:1600], one_hot_encode(iris.target)[:1600])

    hyperparams = {"lr" : 0.5,
                   "inpshape" : x_train1.shape[1],
                   "l1dims" : 80,
                   "l2dims" : 150,
                   "l3dims" : 230,
                   "l4dims" : 210,
                   "l5dims" : 180,
                   "l6dims" : 150,
                   "l7dims" : 120,
                   "l8dims" : 70,
                   "l9dims" : 50,
                   "l10dims" : 30,
                   "l11dims" : 20,
                   "l12dims" : y_train1.shape[1]}

class FCLayer:

    def __init__(self, rows, cols, lr):

        self.w = np.random.randn(rows, cols)
        self.b = np.zeros((1, cols))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigder(self, x):
        return x * (1 - x)

    def forward(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)

    def back(self, inp, out, error):
        weighted_error = error#np.dot(error, self.w.T)
        raw = weighted_error * self.sigder(out)
        dw = np.dot(inp.T, raw)
        self.w -= self.lr * dw
        db = np.sum(raw, axis=0)
        self.b -= self.lr * db
        weighted_error = np.dot(raw, self.w.T)
        return weighted_error

class ClassifyLayer:

    def __init__(self, rows, cols, lr):

        self.w = np.random.randn(rows, cols)
        self.b = np.zeros((1, cols))
        self.lr = lr

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)

    def soft_update(self, y, y_hat):
        return (y_hat - y)/y.shape[0]

    def forward(self, x):
        return self.softmax(np.dot(x, self.w) + self.b)

    def back(self, inp, out, y_proper):
        raw = y_proper
        dw = np.dot(inp.T, raw)
        self.w -= self.lr * dw
        db = np.sum(raw, axis=0, keepdims=True)
        self.b -= self.lr * db
        weighted_error = np.dot(raw, self.w.T)
        return weighted_error

class nnet:

    def __init__(self, x, y, activations):

        self.x = x
        self.y = y

        inputshape = self.x.shape[0]

        self.layers = [self.x]

        for active in activations:
            self.layers.append(np.zeros((inputshape, active.w.shape[1])))
        #self.layers = [self.x, np.zeros((inputshape, layers[0].w.shape[1])), np.zeros((inputshape, layers[1].w.shape[1])), np.zeros((inputshape, layers[2].w.shape[1]))]
        self.activations = activations

        for layer in self.layers:
            print(layer.shape)

    def net_forward(self, data):
        for index, activation in enumerate(self.activations):
            self.layers[index + 1] = activation.forward(self.layers[index])

    def cce(self, y_hat):
        loss = np.sum(-np.log(y_hat[np.arange(y_hat.shape[0]), self.y.argmax(axis=1)]))/y_hat.shape[0]
        return loss

    def mse(self, y_hat):
        loss = np.sum((self.y - y_hat)**2)/y_hat.shape[0]
        return loss

    def net_back(self):
        da = ((self.layers[-1] - self.y)/self.y.shape[0])
        for l in reversed(list(enumerate(self.activations))):
            (index, activation) = l
            da = activation.back(self.layers[index], self.layers[index + 1], da)
        return self.mse(self.layers[-1])

    def optimize(self):

        self.net_forward(self.x)
        loss = self.net_back()

        return loss

    def data_forward(self, data):
        layers = [data] + ([None] * (len(self.layers) - 1))
        for index, activation in enumerate(self.activations):
            layers[index + 1] = activation.forward(layers[index])
        return layers[-1]

    def train(self, its, monitor=0):

        for i in range(its):
            loss = self.optimize()

            if monitor != 0:
                if i % math.floor(its/monitor) == 0:
                    print("Iteration: ", i)
                    print("Current loss: ", loss)

    def predict(self, data):
        """
        for layer in self.layers:
            print(layer.shape)
        """
        final = self.data_forward(data)
        return final

def build_layers(params):
    paramnums = list(params.values())
    print(paramnums)
    lr = paramnums.pop(0)
    layers = []
    layerargs = [(paramnums[index], paramnums[index+1], lr) for index in range(len(paramnums) - 1)]
    for layerparamsind in range(len(layerargs) - 1):
        (indim, outdim, lr) = layerargs[layerparamsind]
        layers.append(FCLayer(indim, outdim, lr))
    layers.append(ClassifyLayer(layerargs[-1][0], layerargs[-1][1], layerargs[-1][2]))
    print(layers)
    return layers

if __name__ == "__main__":
    layers = build_layers(hyperparams)
    """
    [FCLayer(hyperparams["inpshape"], hyperparams["l1dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l1dims"], hyperparams["l2dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l2dims"], hyperparams["l3dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l3dims"], hyperparams["l4dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l4dims"], hyperparams["l5dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l5dims"], hyperparams["l6dims"], hyperparams["lr"]),
              FCLayer(hyperparams["l6dims"], hyperparams["l7dims"], hyperparams["lr"]),
              ClassifyLayer(hyperparams["l7dims"], hyperparams["l8dims"], hyperparams["lr"])]
    """

    whereswaldo = nnet(x_train1, y_train1, layers)

    print("starting now")
    whereswaldo.train(50000, 20)

    def get_acc(xset, yset):
        acc = 0
        for x, y in zip(xset, yset):
            s = whereswaldo.predict(x).argmax()
            if s == y.argmax():
                acc += 1
        return acc / xset.shape[0]

    print("Accuracy for first set: ", get_acc(x_train1, y_train1))
