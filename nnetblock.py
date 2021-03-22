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

iris = load_digits()
x_train1, x_train2, y_train1, y_train2 = divide_and_shuffle(iris.data[:1600], one_hot_encode(iris.target)[:1600])


class nnet:

    def __init__(self, x, y, lr=0.5):
        self.x = x
        self.y = y
        self.lr = lr

        l1dims = 20
        l2dims = 10
        l3dims = 50
        l4dims = 30

        self.w1 = np.random.randn(self.x.shape[1], l1dims)
        self.b1 = np.zeros((1, l1dims))
        self.w2 = np.random.randn(self.w1.shape[1], l2dims)
        self.b2 = np.zeros((1, l2dims))
        self.w3 = np.random.randn(self.w2.shape[1], l3dims)
        self.b3 = np.zeros((1, l3dims))
        self.w4 = np.random.randn(self.w3.shape[1], l4dims)
        self.b4 = np.zeros((1, l4dims))
        self.w5 = np.random.randn(self.w4.shape[1], self.y.shape[1])
        self.b5 = np.zeros((1, self.y.shape[1]))

    def load_swap(self, newx, newy):
        if newx.shape == self.x.shape:
            self.backupx = newx
        if newy.shape == self.y.shape:
            self.backupy = newy

    def switch(self):
        self.x, self.backupx = self.backupx, self.x
        self.y, self.backupy = self.backupy, self.y

    def shuffle(self):
        self.x, self.backupx, self.y, self.backupy = divide_and_shuffle(np.concatenate((self.x, self.backupx)), np.concatenate((self.y, self.backupy)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)

    def cce(self, y_hat):
        return (y_hat - self.y)/self.y.shape[0]

    def error(self, y_hat):
        loss = np.sum(-np.log(y_hat[np.arange(y_hat.shape[0]), self.y.argmax(axis=1)]))/y_hat.shape[0]
        return loss

    def think(self, x, weights, biases):
        return self.sigmoid(np.dot(x, weights) + biases)

    def classify(self, x, weights, biases):
        return self.softmax(np.dot(x, weights) + biases)

    def forward(self, data):
        l1 = self.think(data, self.w1, self.b1)
        l2 = self.think(l1, self.w2, self.b2)
        l3 = self.think(l2, self.w3, self.b3)
        l4 = self.think(l3, self.w4, self.b4)
        l5 = self.classify(l4, self.w5, self.b5)

        return [l5, l4, l3, l2, l1]

    def backstep(self, curr, inputs, outputs, weights):
        weighted_error = np.dot(curr, weights.T)
        raw = weighted_error * self.sigmoid_derivative(outputs)
        wgrad = np.dot(inputs.T, raw)
        bgrad = np.sum(raw, axis=0)
        return raw, wgrad, bgrad

    def back(self, layers):

        (l5, l4, l3, l2, l1) = layers

        final_raw = self.cce(l5)
        true_error = self.error(l5)

        l4raw, w4grad, b4grad = self.backstep(final_raw, l3, l4, self.w5)
        l3raw, w3grad, b3grad = self.backstep(l4raw, l2, l3, self.w4)
        l2raw, w2grad, b2grad = self.backstep(l3raw, l1, l2, self.w3)
        l1raw, w1grad, b1grad = self.backstep(l2raw, self.x, l1, self.w2)

        self.w5 -= self.lr * np.dot(l4.T, final_raw)
        self.b5 -= self.lr * np.sum(final_raw, axis=0, keepdims=True)
        self.w4 -= self.lr * w4grad
        self.b4 -= self.lr * b4grad
        self.w3 -= self.lr * w3grad
        self.b3 -= self.lr * b3grad
        self.w2 -= self.lr * w2grad
        self.b2 -= self.lr * b2grad
        self.w1 -= self.lr * w1grad
        self.b1 -= self.lr * b1grad

        return true_error

    def optimize(self):

        #pass the data through the network and read the layers
        layers = self.forward(self.x)

        #backpropogate to update gradients and read loss value
        loss = self.back(layers)

        return loss

    def train(self, its, monitor=0, switch=0, shuffle=0):

        for i in range(its):
            loss = self.optimize()

            if monitor != 0:
                if i % math.floor(its/monitor) == 0:
                    print("Iteration: ", i)
                    print("Current loss: ", loss)

            if switch != 0:
                if i % math.floor(its/switch) == 0:
                    print("Swapping training sets...")
                    self.switch()

            if shuffle != 0:
                if i % math.floor(its/switch) == 0:
                    print("Shuffling training sets...")
                    self.shuffle()

    def predict(self, data):
        out, _, _, _, _ = self.forward(data)
        return out

if __name__ == "__main__":
    whereswaldo = nnet(x_train1, y_train1, 0.8)
    whereswaldo.load_swap(x_train2, y_train2)
    print("starting now")
    whereswaldo.train(20000, 20, 40, 30)

    def get_acc(xset, yset):
        acc = 0
        for x, y in zip(xset, yset):
            s = whereswaldo.predict(x).argmax()
            if s == y.argmax():
                acc += 1
        return acc / xset.shape[0]

    print("Accuracy for first set: ", get_acc(x_train1, y_train1))
    print("Accuracy for second set: ", get_acc(x_train2, y_train2))
